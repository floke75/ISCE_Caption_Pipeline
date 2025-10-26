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
from dataclasses import dataclass, replace
from typing import List
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType, TokenRow
from .scorer import Scorer
from .config import Config
from .utils import _token_to_row_dict, _compute_transition_scores, _get_lookahead_slice
from .post_process import _block_ranges

FALLBACK_SB_PENALTY = 25.0

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
        self.transition_scores_cache = {}

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

    def _is_hard_ok_SB(self, block_start_idx: int, current_idx: int) -> bool:
        """Checks if a block break (`SB`) violates hard constraints."""
        start_token = self.tokens[block_start_idx]
        end_token = self.tokens[current_idx]
        duration = max(1e-6, end_token.end - start_token.start)
        if duration < self.cfg.min_block_duration_s:
            return False
        num_words_in_block = (current_idx - block_start_idx) + 1
        if num_words_in_block == 1:
            word = start_token.w.rstrip('.,?!')
            if len(word) < self.cfg.min_chars_for_single_word_block and start_token.pos != "PROPN":
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

        self.transition_scores_cache = _compute_transition_scores(self.tokens, self.scorer, self.cfg)

        initial_state = PathState(score=0.0, line_num=1, line_len=len(self.tokens[0].w), block_start_idx=0, breaks=())
        self.beam = [initial_state]

        for i, token in tqdm(enumerate(self.tokens), total=len(self.tokens), desc="Segmenting", unit="token"):
            candidates: List[PathState] = []
            is_last_token = (i == len(self.tokens) - 1)
            nxt = self.tokens[i + 1] if not is_last_token else None
            transition_scores = self.transition_scores_cache[i]

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
                if self._is_hard_ok_SB(state.block_start_idx, i):
                    block_token_dicts = [dict(t.__dict__) for t in self.tokens[state.block_start_idx : i + 1]]
                    block_breaks = list(state.breaks[state.block_start_idx:]) + ["SB"]
                    block_score = self.scorer.score_block(block_token_dicts, block_breaks)
                    score = state.score + transition_scores["SB"] + block_score
                    next_word_len = len(nxt.w) if nxt else 0
                    candidates.append(PathState(score=score, line_num=1, line_len=next_word_len, block_start_idx=i + 1, breaks=state.breaks + ("SB",)))

            if not candidates and self.beam:
                fallback_state = self.beam[0]
                block_tokens = [dict(t.__dict__) for t in self.tokens[fallback_state.block_start_idx : i + 1]]
                block_breaks = list(fallback_state.breaks[fallback_state.block_start_idx:]) + ["SB"]
                block_score = self.scorer.score_block(block_tokens, block_breaks) if block_tokens else 0.0
                next_word_len = len(nxt.w) if nxt else 0
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
        
        return final_breaks

def _refine_blocks(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Re-run the search around cues that were scored as low quality.

    Each block is evaluated with :meth:`Scorer.score_block`.  When the score
    drops below the empirical ``-5.0`` threshold we re-segment a window around
    the problematic cue using a temporary segmenter configured with a wider
    beam.  The function reuses :func:`isce.post_process._block_ranges` so the
    refinement pass stays aligned with the post-processing utilities.
    """
    refined_tokens = list(tokens)
    block_ranges = list(_block_ranges(refined_tokens))

    for i, (start, end) in enumerate(block_ranges):
        block = refined_tokens[start : end + 1]
        breaks = [t.break_type for t in block]
        score = scorer.score_block([t.__dict__ for t in block], breaks)

        if score < -5.0:  # Threshold for a "low-scoring" block
            window_start = max(0, start - 5)
            window_end = min(len(tokens), end + 5)
            window_tokens = tokens[window_start:window_end]

            refined_cfg = replace(cfg, beam_width=cfg.beam_width * 2)
            segmenter = Segmenter(window_tokens, scorer, refined_cfg)
            refined_breaks = segmenter.run()

            for j, br in enumerate(refined_breaks):
                if window_start + j < len(refined_tokens):
                    refined_tokens[window_start + j] = replace(refined_tokens[window_start + j], break_type=br)

    return refined_tokens

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
    total = 0.0
    block_tokens: List[Token] = []
    block_breaks: List[BreakType] = []

    for idx, (token, br) in enumerate(zip(tokens, breaks)):
        block_tokens.append(replace(token, break_type=br))
        block_breaks.append(br)

        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        row = TokenRow(
            token=_token_to_row_dict(token),
            nxt=_token_to_row_dict(nxt) if nxt else None,
            lookahead=_get_lookahead_slice(tokens, idx + 1, cfg.lookahead_width),
        )
        transition_scores = scorer.score_transition(row)
        total += transition_scores.get(br, 0.0)

        if br == "SB":
            total += scorer.score_block([t.__dict__ for t in block_tokens], block_breaks)
            block_tokens = []
            block_breaks = []

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
    (``LB`` → ``SB`` → ``O``).  This preserves gains uncovered during the
    reverse pass while continuing to respect preferences for line and subtitle
    boundaries when both segmentations are equivalent.
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

        if candidate_score > current_score + 1e-6 or (
            abs(candidate_score - current_score) <= 1e-6
            and tie_priority.get(candidate_break, 0) > tie_priority.get(reconciled[i], 0)
        ):
            reconciled = candidate_breaks
            current_score = candidate_score

    return reconciled

def segment(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Entry point that runs all configured segmentation passes.

    The wrapper first runs the forward beam search, optionally mirrors the
    process in reverse so :func:`_reconcile_bidirectional_breaks` can combine
    the two perspectives, and finally performs the localized refinement pass
    when ``cfg.enable_refinement_pass`` is enabled.  The resulting break
    decisions are applied to the original token objects to produce an updated
    token list suitable for downstream rendering and post-processing.
    """
    if not tokens:
        return []

    # Initial forward pass
    forward_segmenter = Segmenter(tokens, scorer, cfg)
    final_breaks = forward_segmenter.run()

    # Optional bidirectional pass
    if cfg.enable_bidirectional_pass:
        reversed_tokens = tokens[::-1]
        backward_segmenter = Segmenter(reversed_tokens, scorer, cfg)
        backward_raw_breaks = backward_segmenter.run()
        backward_breaks = _map_reversed_breaks(backward_raw_breaks)
        final_breaks = _reconcile_bidirectional_breaks(final_breaks, backward_breaks, scorer, tokens, cfg)

    segmented_tokens = [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]

    # Optional refinement pass
    if cfg.enable_refinement_pass:
        segmented_tokens = _refine_blocks(segmented_tokens, scorer, cfg)

    return segmented_tokens
