# C:\dev\Captions_Formatter\Formatter_machine\isce\beam_search.py
"""Implements the caption segmentation search pipeline.

The functions and classes in this module combine the statistical scorer with
production heuristics to determine the optimal break sequence (`O`, `LB`,
`SB`) for a stream of enriched tokens.  The search now mirrors the behaviour
used in production: we prime the forward beam search with cached transition
scores, consult learned block scores every time we close a cue, and—when
enabled in :class:`isce.config.Config`—run a reverse beam search so decisions
from both timelines can be reconciled.  Additional passes re-run the search in
small windows for low quality cues and invoke post-processing helpers from
``isce.post_process``.
"""
from __future__ import annotations
import inspect
from dataclasses import dataclass, replace
from typing import List
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType, TokenRow, TransitionContext
from .scorer import Scorer
from .config import Config
from .utils import _build_token_rows
from .post_process import _block_ranges, reflow_tokens

FALLBACK_SB_PENALTY = 25.0
LOCAL_REFINEMENT_MIN_BEAM = 5
LOCAL_REFINEMENT_IMPROVEMENT = 0.5

@dataclass(frozen=True)
class PathState:
    """Represents one hypothesis (a path) in the beam search."""
    score: float
    line_num: int
    line_len: int
    block_start_idx: int
    breaks: tuple[BreakType, ...]

class Segmenter:
    """Stateful driver for the forward (or reverse) beam search.

    The segmenter walks tokens left-to-right, expanding the best scoring paths
    stored in :class:`PathState` objects.  Transition scores, line-length
    heuristics, and block-level quality metrics are blended together so that
    each hypothesis reflects both the immediate break decision and how that
    decision affects the remainder of the subtitle block.  The search mirrors
    the configuration passed in via :class:`isce.config.Config`, including
    single-word leniency sliders and the tunable fallback subtitle-break
    penalty used when no candidates survive pruning.

    Attributes:
        tokens: Ordered :class:`isce.types.Token` entries to segment.
        scorer: Shared :class:`isce.scorer.Scorer` used to rate decisions.
        cfg: Runtime configuration controlling beam width and heuristics.
        beam: Current best :class:`PathState` hypotheses.
        line_len_leniency: Slider derived factor for soft overage penalties.
        orphan_leniency: Slider derived factor for orphaned line penalties.
        transition_scores_cache: Pre-computed per-token transition scores so
            reverse and refinement passes can reuse identical inputs.
    """
    def __init__(self, tokens: List[Token], scorer: Scorer, cfg: Config):
        self.tokens = tokens
        self.scorer = scorer
        self.cfg = cfg
        self.beam: List[PathState] = []
        self.line_len_leniency = self.scorer.sl.get("line_length_leniency", 1.0)
        self.orphan_leniency = self.scorer.sl.get("orphan_leniency", 1.0)
        self.fallback_sb_penalty = float(self.scorer.sl.get("fallback_sb_penalty", FALLBACK_SB_PENALTY))
        self.lookahead_width = getattr(self.cfg, "lookahead_width", 0)
        self.short_line_penalty = float(self.scorer.sl.get("single_word_line_penalty", 0.0))
        allowed_proper_nouns = getattr(self.cfg, "allowed_single_word_proper_nouns", set())
        self.allowed_proper_nouns = {noun.strip().lower() for noun in allowed_proper_nouns}
        self._token_rows = _build_token_rows(self.tokens, self.cfg)
        signature = inspect.signature(self.scorer.score_transition)
        params = list(signature.parameters.values())
        self._transition_accepts_ctx = any(
            param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for param in params
        ) or len(params) >= 2
        self.last_path_score: float | None = None

    def _is_hard_ok_O(self, line_num: int, line_len: int, next_word_len: int) -> bool:
        """Checks if continuing a line (`O`) violates hard length constraints."""
        limit_key = f"line{line_num}"
        hard_limit = self.cfg.line_length_constraints.get(limit_key, {}).get("hard_limit", 42)
        return (line_len + 1 + next_word_len) <= hard_limit

    def _is_hard_ok_LB(self, state: PathState, current_idx: int) -> bool:
        """Checks if a line break (`LB`) is allowed at the current position."""
        if state.line_num != 1:
            return False
        # Ensure we do not emit multiple LB decisions within the same block.
        recent_breaks = state.breaks[state.block_start_idx : current_idx + 1]
        return "LB" not in recent_breaks

    def _count_chars(self, line_tokens: List[Token]) -> int:
        if not line_tokens:
            return 0
        return sum(len(token.w) for token in line_tokens) + max(0, len(line_tokens) - 1)

    def _is_allowed_single_word(self, token: Token) -> bool:
        if token.pos != "PROPN":
            return False
        stripped = token.w.rstrip(".,!?;:\"")
        return stripped.lower() in self.allowed_proper_nouns

    def _block_profiles(
        self, state: PathState, block_start_idx: int, end_idx: int
    ) -> tuple[List[Token], List[BreakType], List[List[Token]]]:
        """Return tokens, break labels, and per-line groupings for a block."""

        block_tokens = self.tokens[block_start_idx : end_idx + 1]
        block_breaks = list(state.breaks[block_start_idx:end_idx]) + ["SB"]
        lines: List[List[Token]] = []
        current_line: List[Token] = []
        for idx, token in enumerate(block_tokens):
            current_line.append(token)
            if block_breaks[idx] in ("LB", "SB"):
                lines.append(list(current_line))
                current_line = []
        if current_line:
            lines.append(list(current_line))
        return block_tokens, block_breaks, lines

    def _line_violations(self, lines: List[List[Token]]) -> List[str]:
        """Surface soft violations found within the candidate block lines."""

        violations: List[str] = []
        min_chars = self.cfg.min_chars_for_single_word_block
        for line_tokens in lines:
            if not line_tokens:
                continue
            is_single_word = len(line_tokens) == 1
            allowed_single = is_single_word and self._is_allowed_single_word(line_tokens[0])
            if is_single_word and not allowed_single:
                violations.append("single_word")
                continue
            if self._count_chars(line_tokens) < min_chars and not allowed_single:
                violations.append("short_line")
        return violations

    def _is_hard_ok_SB(self, state: PathState, current_idx: int) -> bool:
        """Checks if a block break (`SB`) violates hard constraints."""
        block_start_idx = state.block_start_idx
        start_token = self.tokens[block_start_idx]
        end_token = self.tokens[current_idx]
        chronological_start = min(start_token.start, end_token.start)
        chronological_end = max(start_token.end, end_token.end)
        duration = max(1e-6, chronological_end - chronological_start)
        if duration < self.cfg.min_block_duration_s:
            return False
        _, _, lines = self._block_profiles(state, block_start_idx, current_idx)
        if self._line_violations(lines):
            return False
        return True

    def _estimate_second_line(self, current_idx: int) -> tuple[int, int]:
        """Estimate how much content a hypothetical second line could hold."""

        if current_idx + 1 >= len(self.tokens):
            return 0, 0

        length = 0
        words = 0
        soft_target = self.cfg.line_length_constraints.get("line2", {}).get("soft_target", 37)

        for j in range(current_idx + 1, len(self.tokens)):
            token = self.tokens[j]
            if words > 0:
                length += 1
            length += len(token.w)
            words += 1

            if words >= 2:
                break
            if token.is_sentence_final or token.speaker_change or token.starts_with_dialogue_dash:
                break
            if length >= soft_target:
                break

        return length, words

    def _build_transition_context(self, state: PathState, current_idx: int) -> TransitionContext:
        """Construct the context describing the partially written block."""

        block_tokens = tuple(dict(t.__dict__) for t in self.tokens[state.block_start_idx : current_idx + 1])
        projected_chars: int | None = None
        projected_words: int | None = None
        if state.line_num == 1 and current_idx + 1 < len(self.tokens):
            projected_chars, projected_words = self._estimate_second_line(current_idx)

        return TransitionContext(
            pending_tokens=block_tokens,
            current_line_num=state.line_num,
            current_line_len=state.line_len,
            projected_second_line_chars=projected_chars,
            projected_second_line_words=projected_words,
        )

    def _score_transition(self, row: TokenRow, context: TransitionContext) -> dict[str, float]:
        if self._transition_accepts_ctx:
            return self.scorer.score_transition(row, context)
        return self.scorer.score_transition(row)

    def run(self) -> List[BreakType]:
        """
        Executes the main beam search algorithm.
        This method iterates through each token in the input sequence. At each
        step, it expands each hypothesis in the current beam by considering all
        valid next break types ('O', 'LB', 'SB'). Each new potential path is
        scored, and the beam is pruned to keep only the top N hypotheses, where
        N is the beam width.
        Returns:
            A list of `BreakType` enums representing the best-scoring
            segmentation path found.
        """
        if not self.tokens:
            return []

        self._token_rows = _build_token_rows(self.tokens, self.cfg)

        initial_state = PathState(score=0.0, line_num=1, line_len=len(self.tokens[0].w), block_start_idx=0, breaks=())
        self.beam = [initial_state]

        for i, token in tqdm(enumerate(self.tokens), total=len(self.tokens), desc="Segmenting", unit="token"):
            candidates: List[PathState] = []
            is_last_token = (i == len(self.tokens) - 1)
            nxt = self.tokens[i + 1] if not is_last_token else None
            row = self._token_rows[i]

            for state in self.beam:
                context = self._build_transition_context(state, i)
                transition_scores = self._score_transition(row, context)

                # Candidate: 'O' (No Break)
                if nxt and self._is_hard_ok_O(state.line_num, state.line_len, len(nxt.w)):
                    new_line_len = state.line_len + 1 + len(nxt.w)
                    limit_key = f"line{state.line_num}"
                    constraints = self.cfg.line_length_constraints.get(limit_key, {})
                    soft_target = constraints.get("soft_target", 37)
                    soft_min = constraints.get("soft_min", self.cfg.line_length_soft_min)
                    over_scale = constraints.get(
                        "soft_over_penalty_scale", self.cfg.line_length_overflow_scale
                    )
                    under_scale = constraints.get(
                        "soft_under_penalty_scale", self.cfg.line_length_underflow_scale
                    )
                    leniency = max(1e-6, self.line_len_leniency)
                    line_len_penalty = 0.0
                    if new_line_len > soft_target:
                        overage = new_line_len - soft_target
                        line_len_penalty += ((overage ** 2) * over_scale) / leniency
                    if soft_min and new_line_len < soft_min:
                        shortfall = soft_min - new_line_len
                        line_len_penalty += ((shortfall ** 2) * under_scale) / leniency
                    score = state.score + transition_scores["O"] - line_len_penalty
                    candidates.append(
                        PathState(
                            score=score,
                            line_num=state.line_num,
                            line_len=new_line_len,
                            block_start_idx=state.block_start_idx,
                            breaks=state.breaks + ("O",),
                        )
                    )

                # Candidate: 'LB' (Line Break)
                if nxt and self._is_hard_ok_LB(state, i):
                    orphan_penalty = 0.0
                    if i + 2 < len(self.tokens) and self.tokens[i + 2].is_sentence_final:
                        orphan_penalty = 2.5
                    elif i + 1 < len(self.tokens) and self.tokens[i + 1].is_sentence_final:
                        orphan_penalty = 5.0
                    score = state.score + transition_scores["LB"] - (orphan_penalty * self.orphan_leniency)
                    candidates.append(
                        PathState(
                            score=score,
                            line_num=2,
                            line_len=len(nxt.w),
                            block_start_idx=state.block_start_idx,
                            breaks=state.breaks + ("LB",),
                        )
                    )

                # Candidate: 'SB' (Block Break)
                if self._is_hard_ok_SB(state, i):
                    block_tokens, block_breaks, _ = self._block_profiles(state, state.block_start_idx, i)
                    block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                    block_score = self.scorer.score_block(block_token_dicts, block_breaks)
                    score = state.score + transition_scores["SB"] + block_score
                    next_word_len = len(nxt.w) if nxt else 0
                    candidates.append(
                        PathState(
                            score=score,
                            line_num=1,
                            line_len=next_word_len,
                            block_start_idx=i + 1,
                            breaks=state.breaks + ("SB",),
                        )
                    )

            if not candidates and self.beam:
                fallback_state = self.beam[0]
                fallback_context = self._build_transition_context(fallback_state, i)
                fallback_scores = self._score_transition(row, fallback_context)
                block_tokens, block_breaks, lines = self._block_profiles(
                    fallback_state, fallback_state.block_start_idx, i
                )
                block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                block_score = self.scorer.score_block(block_token_dicts, block_breaks) if block_token_dicts else 0.0
                next_word_len = len(nxt.w) if nxt else 0
                violations = self._line_violations(lines)
                if violations:
                    per_violation_penalty = self.short_line_penalty or self.fallback_sb_penalty
                    block_score -= per_violation_penalty * len(violations)
                fallback_candidate = PathState(
                    score=fallback_state.score + fallback_scores.get("SB", 0.0) + block_score - self.fallback_sb_penalty,
                    line_num=1,
                    line_len=next_word_len,
                    block_start_idx=i + 1,
                    breaks=fallback_state.breaks + ("SB",),
                )
                candidates.append(fallback_candidate)

            if not candidates:
                break

            self.beam = nlargest(self.cfg.beam_width, candidates, key=lambda s: s.score)

        best_path = self.beam[0] if self.beam else initial_state
        self.last_path_score = best_path.score
        final_breaks = list(best_path.breaks)
        
        while len(final_breaks) < len(self.tokens):
            final_breaks.append("O")
        if final_breaks:
            final_breaks[-1] = "SB"
        
        return final_breaks

def _refine_blocks(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Re-run the search around cues that were scored as low quality.

    Each block is evaluated with :meth:`Scorer.score_block`.  When the score
    drops below the empirical ``-5.0`` threshold we re-segment a window around
    the problematic cue using a temporary segmenter configured with a wider
    beam.  The function reuses :func:`isce.post_process._block_ranges` so the
    refinement pass stays aligned with the post-processing utilities.  When a
    window extends beyond the block being refined we keep the trailing token's
    original break decision so the localized search does not introduce spurious
    subtitle breaks into the surrounding context.  Candidate refinements are
    accepted only when they improve the holistic segmentation score by a
    meaningful margin to avoid thrashing stable cues.
    """

    refined_tokens = list(tokens)
    if not refined_tokens:
        return refined_tokens

    block_ranges = list(_block_ranges(refined_tokens))
    idx = 0

    while idx < len(block_ranges):
        start, end = block_ranges[idx]
        block = refined_tokens[start : end + 1]
        breaks = [t.break_type for t in block]
        score = scorer.score_block([t.__dict__ for t in block], breaks)

        if score >= -5.0:
            idx += 1
            continue

        window_start = max(0, start - 5)
        window_end = min(len(refined_tokens), end + 6)
        window_tokens = refined_tokens[window_start:window_end]

        if not window_tokens:
            idx += 1
            continue

        baseline_breaks: List[BreakType] = []
        for offset, token in enumerate(window_tokens):
            br = token.break_type
            if br is None:
                br = "SB" if offset == len(window_tokens) - 1 else "O"
            baseline_breaks.append(br)

        baseline_score = _score_segmentation(window_tokens, baseline_breaks, scorer, cfg)

        refined_beam = max(cfg.beam_width * 2, LOCAL_REFINEMENT_MIN_BEAM)
        refined_cfg = replace(cfg, beam_width=refined_beam)
        window_segmenter = Segmenter(list(window_tokens), scorer, refined_cfg)
        candidate_breaks = window_segmenter.run()

        adjusted_breaks = list(candidate_breaks)
        if window_end < len(refined_tokens) and adjusted_breaks:
            preserved = refined_tokens[window_end - 1].break_type
            if preserved is not None:
                adjusted_breaks[-1] = preserved
            else:
                adjusted_breaks[-1] = "SB" if (window_end - 1) == len(refined_tokens) - 1 else "O"

        candidate_score = _score_segmentation(window_tokens, adjusted_breaks, scorer, cfg)

        if candidate_score < (baseline_score + LOCAL_REFINEMENT_IMPROVEMENT):
            idx += 1
            continue

        for offset, new_break in enumerate(adjusted_breaks):
            absolute_idx = window_start + offset
            if absolute_idx >= len(refined_tokens):
                break
            refined_tokens[absolute_idx] = replace(refined_tokens[absolute_idx], break_type=new_break)

        block_ranges = list(_block_ranges(refined_tokens))
        idx = 0

    return refined_tokens

def _reverse_tokens_for_bidirectional(tokens: List[Token]) -> List[Token]:
    """Prepare a reversed copy of the tokens for the backward beam search."""

    reversed_tokens: List[Token] = []
    for token in reversed(tokens):
        original_relative = getattr(token, "relative_position", None)
        mirrored_position = None
        if original_relative is not None:
            mirrored_position = max(0.0, min(1.0, 1.0 - original_relative))

        reversed_tokens.append(
            replace(
                token,
                start=-token.end,
                end=-token.start,
                pause_after_ms=token.pause_before_ms,
                pause_before_ms=token.pause_after_ms,
                is_sentence_initial=token.is_sentence_final,
                is_sentence_final=token.is_sentence_initial,
                relative_position=mirrored_position if mirrored_position is not None else original_relative,
                break_type=None,
            )
        )

    return reversed_tokens


def _map_reversed_breaks(reversed_breaks: List[BreakType]) -> List[BreakType]:
    """Translate reverse-pass break decisions back into forward order.

    The helper keeps subtitle-boundary markers discovered by the backward pass
    so reconciliation can consider them alongside the forward results.  The
    only mutation performed is ensuring the final break is an ``SB`` in forward
    order, matching the invariants used elsewhere in the segmentation code.
    """
    n = len(reversed_breaks)
    if n == 0:
        return []

    mapped = list(reversed(reversed_breaks))
    final_breaks: List[BreakType] = [b if b != "SB" else "O" for b in mapped]
    final_breaks[-1] = "SB"

    # Preserve internal subtitle-block boundaries proposed by the backward
    # search. A reversed boundary after token ``j`` corresponds to the forward
    # boundary after token ``i = n - 2 - j``.
    for backward_idx, br in enumerate(reversed_breaks[:-1]):
        if br == "SB":
            forward_idx = (n - 2) - backward_idx
            final_breaks[forward_idx] = "SB"

    return final_breaks


def _score_segmentation(tokens: List[Token], breaks: List[BreakType], scorer: Scorer, cfg: Config) -> float:
    """Calculate holistic scores for an entire segmentation sequence.

    Used primarily by :func:`_reconcile_bidirectional_breaks`, this helper
    mirrors the scoring performed during beam search by summing transition
    scores for every decision and block scores whenever an ``SB`` is emitted.
    """
    if not tokens:
        return 0.0

    total = 0.0
    segmenter = Segmenter(tokens, scorer, cfg)
    current_breaks: List[BreakType] = []
    line_num = 1
    line_len = len(tokens[0].w)
    block_start_idx = 0

    for idx, br in enumerate(breaks):
        row = segmenter._token_rows[idx]
        state = PathState(
            score=0.0,
            line_num=line_num,
            line_len=line_len,
            block_start_idx=block_start_idx,
            breaks=tuple(current_breaks),
        )
        context = segmenter._build_transition_context(state, idx)
        transition_scores = segmenter._score_transition(row, context)
        total += transition_scores.get(br, 0.0)

        if br == "SB":
            block_tokens, block_breaks, _ = segmenter._block_profiles(state, block_start_idx, idx)
            block_token_dicts = [dict(t.__dict__) for t in block_tokens]
            total += scorer.score_block(block_token_dicts, block_breaks)

        current_breaks.append(br)

        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if br == "O":
            if nxt:
                line_len = line_len + 1 + len(nxt.w)
        elif br == "LB":
            line_num = 2
            line_len = len(nxt.w) if nxt else 0
        elif br == "SB":
            line_num = 1
            line_len = len(nxt.w) if nxt else 0
            block_start_idx = idx + 1

    return total


def _reconcile_bidirectional_breaks(
    forward_breaks: List[BreakType],
    backward_breaks: List[BreakType],
    scorer: Scorer,
    tokens: List[Token],
    cfg: Config,
) -> List[BreakType]:
    """Blend forward and backward break choices into a single sequence.

    We accept the backward decision when it strictly improves the holistic
    score or, in the event of a tie, when its break type has higher priority
    (``LB`` → ``SB`` → ``O``).  When the scores match and both passes would
    otherwise emit consecutive subtitle breaks, we keep the backward
    continuation so the reconciliation does not reintroduce short trailing
    blocks that the reverse search intentionally removed.
    """
    reconciled = list(forward_breaks)
    current_score = _score_segmentation(tokens, reconciled, scorer, cfg)
    tie_priority = {"LB": 3, "SB": 2, "O": 1}

    for i in range(len(tokens)):
        candidate_break = backward_breaks[i]
        if reconciled[i] == candidate_break:
            continue

        candidate_breaks = list(reconciled)
        candidate_breaks[i] = candidate_break
        candidate_score = _score_segmentation(tokens, candidate_breaks, scorer, cfg)

        prefer_candidate = False
        if candidate_score > current_score + 1e-6:
            prefer_candidate = True
        elif abs(candidate_score - current_score) <= 1e-6:
            tie_delta = tie_priority.get(candidate_break, 0) - tie_priority.get(reconciled[i], 0)
            if tie_delta > 0:
                prefer_candidate = True
            elif (
                reconciled[i] == "SB"
                and candidate_break != "SB"
                and i + 1 < len(tokens)
                and reconciled[i + 1] == "SB"
            ):
                # Avoid creating consecutive subtitle breaks when the backward
                # pass suggests keeping the block open. This prevents the
                # reconciliation stage from reintroducing short trailing blocks
                # that the reverse search purposefully removed.
                prefer_candidate = True

        if prefer_candidate:
            reconciled = candidate_breaks
            current_score = candidate_score

    return reconciled

def segment(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Entry point that runs all configured segmentation passes.

    The wrapper first runs the forward beam search, optionally mirrors the
    process in reverse so :func:`_reconcile_bidirectional_breaks` can combine
    the two perspectives, and performs a localized refinement pass when
    ``cfg.enable_refinement_pass`` is enabled.  A final optional reflow stage
    invokes :func:`isce.post_process.reflow_tokens` to tidy short or imbalanced
    cues before returning the updated token list.
    """
    if not tokens:
        return []

    # Initial forward pass
    forward_segmenter = Segmenter(tokens, scorer, cfg)
    final_breaks = forward_segmenter.run()

    # Optional bidirectional pass
    if cfg.enable_bidirectional_pass:
        reversed_tokens = _reverse_tokens_for_bidirectional(tokens)
        backward_segmenter = Segmenter(reversed_tokens, scorer, cfg)
        backward_raw_breaks = backward_segmenter.run()
        backward_breaks = _map_reversed_breaks(backward_raw_breaks)
        final_breaks = _reconcile_bidirectional_breaks(final_breaks, backward_breaks, scorer, tokens, cfg)

    segmented_tokens = [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]

    # Optional refinement pass
    if cfg.enable_refinement_pass:
        segmented_tokens = _refine_blocks(segmented_tokens, scorer, cfg)

    if cfg.enable_reflow:
        segmented_tokens = reflow_tokens(segmented_tokens, scorer)

    return segmented_tokens
