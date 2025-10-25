# C:\dev\Captions_Formatter\Formatter_machine\isce\beam_search.py
"""Implements the core beam search algorithm for text segmentation.

This module contains the `Segmenter` class, which performs the beam search
to find the optimal sequence of break decisions (`O`, `LB`, `SB`) for a given
list of tokens. It maintains a beam of the most promising hypotheses (paths)
at each step, using a `Scorer` to evaluate the quality of each potential
decision based on a statistical model and heuristic rules.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType, TokenRow
from .scorer import Scorer
from .config import Config

FALLBACK_SB_PENALTY = 25.0
LOCAL_REFINEMENT_MIN_BEAM = 5
BALANCE_RATIO_THRESHOLD = 2.5

@dataclass(frozen=True)
class PathState:
    """Represents one hypothesis (a path) in the beam search."""
    score: float
    line_num: int
    line_len: int
    block_start_idx: int
    breaks: tuple[BreakType, ...]

class Segmenter:
    """Stateful orchestrator for the beam search segmentation process.

    The :class:`Segmenter` encapsulates the outer loop of the captioning beam
    search. It iterates through tokens, expands each hypothesis with the three
    allowed break types (``O``, ``LB``, ``SB``), and relies on
    :class:`~isce.scorer.Scorer` for the actual scoring mechanics. The class
    also carries a handful of pre-computed settings derived from the active
    configuration so that the tight inner loops stay lean.

    Attributes
    ----------
    tokens:
        Ordered list of :class:`~isce.types.Token` objects to be segmented.
    scorer:
        Instance responsible for scoring transition and block level decisions.
    cfg:
        The loaded :class:`~isce.config.Config` describing constraints.
    beam:
        Mutable collection of the best :class:`PathState` hypotheses explored
        so far.
    line_len_leniency:
        Multiplier used to scale soft penalties once a line exceeds its target
        character count.
    orphan_leniency:
        Multiplier for discouraging line breaks that would leave sentence-final
        orphans (``LB`` followed by punctuation).
    fallback_sb_penalty:
        Penalty applied when we are forced to emit an ``SB`` because every
        other decision violates a hard constraint.
    lookahead_width:
        Number of future tokens the scorer may inspect when evaluating
        transitions.
    short_line_penalty:
        Optional override letting the scorer fine-tune penalties for sub-minimal
        or single-word lines.
    allowed_proper_nouns:
        Normalised set of proper nouns that may appear as single-word captions
        without being considered violations.
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
        self.allowed_proper_nouns = {
            noun.strip().lower()
            for noun in getattr(self.cfg, "allowed_single_word_proper_nouns", tuple())
        }
        # Cache the best path score from the most recent run so downstream
        # refinement helpers can compare candidate segmentations using the same
        # scoring logic applied by the beam search.
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
        """Return the visual character count for ``line_tokens``.

        Spaces between words are counted so that the total mirrors the value the
        human captioner would perceive when judging line length.
        """
        if not line_tokens:
            return 0
        return sum(len(token.w) for token in line_tokens) + (len(line_tokens) - 1)

    def _is_allowed_single_word(self, token: Token) -> bool:
        """Check whether ``token`` is whitelisted for single-word captions."""
        if token.pos != "PROPN":
            return False
        stripped = token.w.rstrip(".,!?;:\"")
        return stripped.lower() in self.allowed_proper_nouns

    def _block_profiles(self, state: PathState, block_start_idx: int, end_idx: int) -> tuple[List[Token], List[BreakType], List[List[Token]]]:
        """Slice the active block and pre-compute its structural metadata.

        The helpers downstream frequently need access to the tokens in the
        active block, their break decisions, and a line-by-line view.  Building
        the representation once keeps the scoring path tidy.
        """
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
        """Identify hard violations produced by a candidate block.

        The fallback path shares logic with :meth:`_is_hard_ok_SB` and needs to
        understand whether a block produced only a single word, or a line that
        falls below the minimum character count.  Returning the violation type
        strings makes it easy to translate the findings into penalties.
        """
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
        duration = max(1e-6, end_token.end - start_token.start)
        if duration < self.cfg.min_block_duration_s:
            return False
        _, _, lines = self._block_profiles(state, block_start_idx, current_idx)
        if self._line_violations(lines):
            return False
        return True

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

        initial_state = PathState(score=0.0, line_num=1, line_len=len(self.tokens[0].w), block_start_idx=0, breaks=())
        self.beam = [initial_state]

        for i, token in tqdm(enumerate(self.tokens), total=len(self.tokens), desc="Segmenting", unit="token"):
            candidates: List[PathState] = []
            is_last_token = (i == len(self.tokens) - 1)
            nxt = self.tokens[i + 1] if not is_last_token else None

            token_dict = dict(token.__dict__)
            nxt_dict = dict(nxt.__dict__) if nxt else None
            lookahead_tokens = None
            if self.lookahead_width > 0:
                # Expose a shallow copy of the upcoming tokens so the scorer can
                # apply lookahead heuristics without mutating the canonical list.
                future_slice = self.tokens[i + 1 : i + 1 + self.lookahead_width]
                if future_slice:
                    lookahead_tokens = tuple(dict(t.__dict__) for t in future_slice)

            # Create the dictionary-based TokenRow required by the refactored scorer
            scorer_row = TokenRow(
                token=token_dict,
                nxt=nxt_dict,
                feats=None,  # feats object is no longer used by the scorer
                lookahead=lookahead_tokens,
            )
            transition_scores = self.scorer.score_transition(scorer_row)

            for state in self.beam:
                # Candidate: 'O' (No Break)
                if nxt:
                    if self._is_hard_ok_O(state.line_num, state.line_len, len(nxt.w)):
                        new_line_len = state.line_len + 1 + len(nxt.w)
                        limit_key = f"line{state.line_num}"
                        soft_target = self.cfg.line_length_constraints.get(limit_key, {}).get("soft_target", 37)
                        line_len_penalty = 0.0
                        if new_line_len > soft_target:
                            overage = new_line_len - soft_target
                            line_len_penalty = ((overage ** 2) * 0.1) / self.line_len_leniency
                        score = state.score + transition_scores["O"] - line_len_penalty
                        candidates.append(PathState(score=score, line_num=state.line_num, line_len=new_line_len, block_start_idx=state.block_start_idx, breaks=state.breaks + ("O",)))

                # Candidate: 'LB' (Line Break)
                if nxt and self._is_hard_ok_LB(state, i):
                    orphan_penalty = 0.0
                    if i + 2 < len(self.tokens) and self.tokens[i + 2].is_sentence_final:
                        orphan_penalty = 2.5
                    elif i + 1 < len(self.tokens) and self.tokens[i + 1].is_sentence_final:
                        orphan_penalty = 5.0
                    score = state.score + transition_scores["LB"] - (orphan_penalty * self.orphan_leniency)
                    candidates.append(PathState(score=score, line_num=2, line_len=len(nxt.w), block_start_idx=state.block_start_idx, breaks=state.breaks + ("LB",)))

                # Candidate: 'SB' (Block Break)
                if self._is_hard_ok_SB(state, i):
                    block_tokens, block_breaks, _ = self._block_profiles(state, state.block_start_idx, i)
                    block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                    block_score = self.scorer.score_block(block_token_dicts, block_breaks)
                    score = state.score + transition_scores["SB"] + block_score
                    next_word_len = len(nxt.w) if nxt else 0
                    candidates.append(PathState(score=score, line_num=1, line_len=next_word_len, block_start_idx=i + 1, breaks=state.breaks + ("SB",)))

            if not candidates and self.beam:
                # All futures violate hard constraints.  Fall back to forcing an
                # ``SB`` on the best active path while recording the reason as a
                # score penalty so that the search prefers cleaner options later
                # in the sequence.
                fallback_state = self.beam[0]
                block_tokens, block_breaks, lines = self._block_profiles(fallback_state, fallback_state.block_start_idx, i)
                block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                block_score = self.scorer.score_block(block_token_dicts, block_breaks) if block_token_dicts else 0.0
                next_word_len = len(nxt.w) if nxt else 0
                violations = self._line_violations(lines)
                if violations:
                    per_violation_penalty = self.short_line_penalty if self.short_line_penalty > 0 else self.fallback_sb_penalty
                    block_score -= per_violation_penalty * len(violations)
                fallback_candidate = PathState(
                    score=fallback_state.score + transition_scores.get("SB", 0.0) + block_score - self.fallback_sb_penalty,
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
        final_breaks = list(best_path.breaks)

        while len(final_breaks) < len(self.tokens):
            final_breaks.append("O")
        if final_breaks:
            final_breaks[-1] = "SB"

        self.last_path_score = best_path.score
        return final_breaks


def _reverse_tokens_for_bidirectional(tokens: List[Token]) -> List[Token]:
    """Create a reversed copy of the tokens suited for the backward beam."""
    reversed_tokens: List[Token] = []
    for token in reversed(tokens):
        relative_position = 1.0 - token.relative_position if token.relative_position is not None else None
        if relative_position is None:
            relative_position = 0.0
        else:
            relative_position = max(0.0, min(1.0, relative_position))
        reversed_tokens.append(
            replace(
                token,
                start=-token.end,
                end=-token.start,
                pause_after_ms=token.pause_before_ms,
                pause_before_ms=token.pause_after_ms,
                is_sentence_initial=token.is_sentence_final,
                is_sentence_final=token.is_sentence_initial,
                relative_position=relative_position,
                break_type=None,
            )
        )
    return reversed_tokens


def _map_reversed_breaks(reversed_breaks: List[BreakType]) -> List[BreakType]:
    """Translate reversed-order break decisions back to the forward timeline."""
    if not reversed_breaks:
        return []

    mirrored = list(reversed(reversed_breaks))

    if mirrored and mirrored[0] == "SB":
        mirrored = mirrored[1:]

    if mirrored:
        mirrored.append("SB")
    else:
        mirrored = ["SB"]

    return mirrored


def _block_span(breaks: List[BreakType], idx: int) -> tuple[int, int]:
    start = idx
    while start > 0 and breaks[start - 1] != "SB":
        start -= 1
    end = idx
    while end < len(breaks) - 1 and breaks[end] != "SB":
        end += 1
    return start, end


def _score_block(tokens: List[Token], breaks: List[BreakType], start: int, end: int, scorer: Scorer) -> float:
    block_tokens = [dict(t.__dict__) for t in tokens[start : end + 1]]
    block_breaks = list(breaks[start : end + 1])
    if block_breaks and block_breaks[-1] != "SB":
        block_breaks[-1] = "SB"
    return scorer.score_block(block_tokens, block_breaks)


def _reconcile_bidirectional_breaks(
    tokens: List[Token],
    scorer: Scorer,
    forward_breaks: List[BreakType],
    backward_breaks: List[BreakType],
) -> List[BreakType]:
    final_breaks = list(forward_breaks)
    locked = [False] * len(tokens)

    for i in range(len(tokens) - 1):
        if locked[i] or forward_breaks[i] == backward_breaks[i]:
            continue
        if "SB" not in {forward_breaks[i], backward_breaks[i]}:
            continue

        f_start, f_end = _block_span(forward_breaks, i)
        b_start, b_end = _block_span(backward_breaks, i)
        f_score = _score_block(tokens, forward_breaks, f_start, f_end, scorer)
        b_score = _score_block(tokens, backward_breaks, b_start, b_end, scorer)

        if b_score > f_score:
            final_breaks[b_start : b_end + 1] = backward_breaks[b_start : b_end + 1]
            for j in range(b_start, b_end + 1):
                locked[j] = True
        else:
            final_breaks[f_start : f_end + 1] = forward_breaks[f_start : f_end + 1]
            for j in range(f_start, f_end + 1):
                locked[j] = True

    if final_breaks:
        final_breaks[-1] = "SB"
    return final_breaks


def _run_forward_breaks(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[BreakType]:
    return Segmenter(tokens, scorer, cfg).run()


def _run_bidirectional_breaks(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[BreakType]:
    forward_breaks = _run_forward_breaks(tokens, scorer, cfg)
    reversed_tokens = _reverse_tokens_for_bidirectional(tokens)
    backward_reversed_breaks = Segmenter(reversed_tokens, scorer, cfg).run()
    backward_breaks = _map_reversed_breaks(backward_reversed_breaks)
    return _reconcile_bidirectional_breaks(tokens, scorer, forward_breaks, backward_breaks)


def _count_chars(token_slice: List[Token]) -> int:
    """Return the number of printable characters in a slice of tokens.

    The helper mirrors the formatter's line length accounting by including the
    number of inter-word spaces in addition to the raw token lengths so that the
    refinement heuristics evaluate potential cues using the same metric.
    """

    if not token_slice:
        return 0
    return sum(len(t.w) for t in token_slice) + max(0, len(token_slice) - 1)


def _block_balance(block_tokens: List[Token], block_breaks: List[BreakType]) -> float:
    """Compute the ratio between the longer and shorter lines in a cue.

    A perfectly balanced two-line cue will have a ratio close to 1.0, while an
    imbalanced cue (for example a single-word second line) yields a much larger
    ratio.  We rely on this signal to decide whether the refinement pass should
    revisit the break placement.
    """

    try:
        lb_idx = block_breaks.index("LB")
    except ValueError:
        return 1.0
    first_line = block_tokens[: lb_idx + 1]
    second_line = block_tokens[lb_idx + 1 :]
    len1 = _count_chars(first_line)
    len2 = _count_chars(second_line)
    if not len1 or not len2:
        return float("inf")
    longer = max(len1, len2)
    shorter = min(len1, len2)
    return longer / max(1, shorter)


def _score_path(tokens: List[Token], breaks: List[BreakType], scorer: Scorer, cfg: Config) -> float:
    """Re-score a fixed segmentation path using the canonical scorer.

    The refinement helpers occasionally explore localized windows using a wider
    beam.  In order to determine whether those alternates actually improve the
    subtitle quality, we need a deterministic way to score the original
    segmentation with the same heuristics the beam search relies on.  This
    function rebuilds that computation without mutating global state.
    """

    if not tokens:
        return 0.0

    total = 0.0
    line_num = 1
    line_len = len(tokens[0].w)
    block_start_idx = 0
    line_len_leniency = scorer.sl.get("line_length_leniency", 1.0)
    orphan_leniency = scorer.sl.get("orphan_leniency", 1.0)

    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        lookahead_tokens = None
        if cfg.lookahead_width > 0:
            future_slice = tokens[i + 1 : i + 1 + cfg.lookahead_width]
            if future_slice:
                lookahead_tokens = tuple(dict(t.__dict__) for t in future_slice)
        row = TokenRow(
            token=dict(token.__dict__),
            nxt=dict(nxt.__dict__) if nxt else None,
            feats=None,
            lookahead=lookahead_tokens,
        )
        transition_scores = scorer.score_transition(row)
        decision = breaks[i]

        if decision == "O":
            if nxt:
                new_line_len = line_len + 1 + len(nxt.w)
                limit_key = f"line{line_num}"
                soft_target = cfg.line_length_constraints.get(limit_key, {}).get("soft_target", 37)
                line_len_penalty = 0.0
                if new_line_len > soft_target:
                    overage = new_line_len - soft_target
                    line_len_penalty = ((overage ** 2) * 0.1) / line_len_leniency
                total += transition_scores.get("O", 0.0) - line_len_penalty
                line_len = new_line_len
        elif decision == "LB":
            orphan_penalty = 0.0
            if i + 2 < len(tokens) and tokens[i + 2].is_sentence_final:
                orphan_penalty = 2.5
            elif i + 1 < len(tokens) and tokens[i + 1].is_sentence_final:
                orphan_penalty = 5.0
            total += transition_scores.get("LB", 0.0) - (orphan_penalty * orphan_leniency)
            line_num = 2
            line_len = len(nxt.w) if nxt else 0
        elif decision == "SB":
            block_token_dicts = [dict(t.__dict__) for t in tokens[block_start_idx : i + 1]]
            block_breaks = list(breaks[block_start_idx:i]) + ["SB"]
            block_score = scorer.score_block(block_token_dicts, block_breaks)
            total += transition_scores.get("SB", 0.0) + block_score
            line_num = 1
            line_len = len(nxt.w) if nxt else 0
            block_start_idx = i + 1
        else:
            raise ValueError(f"Unknown break type: {decision}")

    return total


def _should_refine(block_tokens: List[Token], block_breaks: List[BreakType], block_score: float) -> bool:
    """Decide whether a cue is low quality enough to warrant refinement.

    The heuristic favors re-scoring blocks that:

    * Collapse to a single word (classic orphan cue).
    * Receive a negative structural score from the model.
    * Display highly imbalanced line lengths.

    Returning ``True`` signals to ``refine_blocks`` that it should run a local
    beam search window to hunt for a better segmentation.
    """

    if not block_tokens:
        return False
    if len(block_tokens) == 1:
        return True
    if block_score < 0.0:
        return True
    balance_ratio = _block_balance(block_tokens, block_breaks)
    if balance_ratio > BALANCE_RATIO_THRESHOLD:
        return True
    return False


def refine_blocks(tokens: List[Token], breaks: List[BreakType], scorer: Scorer, cfg: Config) -> List[BreakType]:
    """Run a localized refinement pass over low quality cues.

    The initial beam search sometimes emits harsh cues—usually a one-word block
    or a severely unbalanced two-line cue—because the constrained search space
    cannot justify a better alternative given the global beam width.  When the
    configuration enables it, this helper scans each block, identifies those
    that are suspect via ``_should_refine``, and then re-runs the beam search on
    a limited token window with a wider beam.  If the refined segmentation
    scores higher than the original by a tiny margin, we splice the alternate
    decisions back into the final break sequence.
    """

    if not tokens or not breaks:
        return breaks

    refined_breaks = list(breaks)

    block_boundaries: List[tuple[int, int]] = []
    # Identify the (start, end) index for every cue in the current segmentation
    # so that we can reason about the surrounding context when re-scoring.
    start_idx = 0
    for i, br in enumerate(refined_breaks):
        if br == "SB":
            block_boundaries.append((start_idx, i))
            start_idx = i + 1

    idx = 0
    while idx < len(block_boundaries):
        block_start, block_end = block_boundaries[idx]
        block_tokens = tokens[block_start : block_end + 1]
        block_breaks = list(refined_breaks[block_start:block_end]) + ["SB"]
        block_token_dicts = [dict(t.__dict__) for t in block_tokens]
        block_score = scorer.score_block(block_token_dicts, block_breaks)

        if _should_refine(block_tokens, block_breaks, block_score):
            window_end_block_idx = idx
            if len(block_tokens) == 1 and idx + 1 < len(block_boundaries):
                window_end_block_idx += 1

            window_start = block_boundaries[idx][0]
            window_end = block_boundaries[window_end_block_idx][1]
            window_tokens = tokens[window_start : window_end + 1]
            window_breaks = refined_breaks[window_start : window_end + 1]

            original_score = _score_path(window_tokens, window_breaks, scorer, cfg)
            local_cfg = replace(cfg, beam_width=max(cfg.beam_width, LOCAL_REFINEMENT_MIN_BEAM))
            local_segmenter = Segmenter(window_tokens, scorer, local_cfg)
            local_breaks = local_segmenter.run()
            local_score = (
                local_segmenter.last_path_score
                if local_segmenter.last_path_score is not None
                else _score_path(window_tokens, local_breaks, scorer, local_cfg)
            )

            if local_score > original_score + 1e-6:
                refined_breaks[window_start : window_end + 1] = local_breaks

            idx = window_end_block_idx + 1
        else:
            idx += 1

    return refined_breaks

def segment(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """
    High-level wrapper to perform beam search segmentation.

    This function instantiates the `Segmenter` class, runs the beam search
    algorithm, and applies the resulting break types to the input tokens.

    Args:
        tokens: The list of `Token` objects to segment.
        scorer: The `Scorer` instance to use for evaluating breaks.
        cfg: The main configuration object.

    Returns:
        A new list of `Token` objects with the `break_type` attribute set
        according to the segmentation result.
    """
    if not tokens:
        return []
    
    if cfg.enable_bidirectional_pass:
        final_breaks = _run_bidirectional_breaks(tokens, scorer, cfg)
    else:
        final_breaks = _run_forward_breaks(tokens, scorer, cfg)

    if cfg.enable_refinement_pass:
        final_breaks = refine_blocks(tokens, final_breaks, scorer, cfg)

    return [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]
