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
from .utils import _token_to_row_dict, _compute_transition_scores, _get_lookahead_slice

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
    """
    Manages the beam search segmentation process.

    This stateful class encapsulates the logic for the beam search algorithm,
    iterating through tokens and maintaining a beam of the most likely
    segmentation hypotheses (`PathState` objects). It uses a `Scorer` to
    evaluate the quality of different break decisions at each step.

    Attributes:
        tokens: The list of `Token` objects to be segmented.
        scorer: The `Scorer` instance used to score potential breaks.
        cfg: The main configuration object.
        beam: The list of current best `PathState` hypotheses.
        line_len_leniency: A factor to adjust penalties for long lines.
        orphan_leniency: A factor to adjust penalties for single-word lines.
        transition_scores_cache: A cache for transition scores.
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
    """
    Improves low-scoring blocks by re-running a localized beam search with a wider beam.
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
    """Translate reversed-order break decisions back to the forward timeline."""
    n = len(reversed_breaks)
    if n == 0:
        return []

    # Reverse the breaks and handle the sentinel "SB"
    mapped = reversed(reversed_breaks)
    final_breaks = [b if b != "SB" else "O" for b in mapped]
    if final_breaks:
        final_breaks[-1] = "SB"
    return final_breaks


def _reconcile_bidirectional_breaks(forward_breaks: List[BreakType], backward_breaks: List[BreakType], scorer: Scorer, tokens: List[Token]) -> List[BreakType]:
    """
    Reconciles conflicting break decisions from forward and backward passes.
    """
    reconciled = list(forward_breaks)
    for i in range(len(tokens)):
        if forward_breaks[i] != backward_breaks[i]:
            # Simple reconciliation: prefer SB over LB, and LB over O
            if backward_breaks[i] == "SB":
                reconciled[i] = "SB"
            elif backward_breaks[i] == "LB" and forward_breaks[i] == "O":
                reconciled[i] = "LB"
    return reconciled

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

    # Initial forward pass
    forward_segmenter = Segmenter(tokens, scorer, cfg)
    final_breaks = forward_segmenter.run()

    # Optional bidirectional pass
    if cfg.enable_bidirectional_pass:
        reversed_tokens = tokens[::-1]
        backward_segmenter = Segmenter(reversed_tokens, scorer, cfg)
        backward_raw_breaks = backward_segmenter.run()
        backward_breaks = _map_reversed_breaks(backward_raw_breaks)
        final_breaks = _reconcile_bidirectional_breaks(final_breaks, backward_breaks, scorer, tokens)

    segmented_tokens = [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]

    # Optional refinement pass
    if cfg.enable_refinement_pass:
        segmented_tokens = _refine_blocks(segmented_tokens, scorer, cfg)

    return segmented_tokens
