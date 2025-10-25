# C:\dev\Captions_Formatter\Formatter_machine\isce\beam_search.py

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType, TokenRow
from .scorer import Scorer
from .config import Config

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
    """
    def __init__(self, tokens: List[Token], scorer: Scorer, cfg: Config):
        self.tokens = tokens
        self.scorer = scorer
        self.cfg = cfg
        self.beam: List[PathState] = []
        self.line_len_leniency = self.scorer.sl.get("line_length_leniency", 1.0)
        self.orphan_leniency = self.scorer.sl.get("orphan_leniency", 1.0)
        self.fallback_sb_penalty = float(self.scorer.sl.get("fallback_sb_penalty", FALLBACK_SB_PENALTY))

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

        initial_state = PathState(score=0.0, line_num=1, line_len=len(self.tokens[0].w), block_start_idx=0, breaks=())
        self.beam = [initial_state]

        for i, token in tqdm(enumerate(self.tokens), total=len(self.tokens), desc="Segmenting", unit="token"):
            candidates: List[PathState] = []
            is_last_token = (i == len(self.tokens) - 1)
            nxt = self.tokens[i + 1] if not is_last_token else None

            token_dict = dict(token.__dict__)
            nxt_dict = dict(nxt.__dict__) if nxt else None

            # Create the dictionary-based TokenRow required by the refactored scorer
            scorer_row = TokenRow(
                token=token_dict,
                nxt=nxt_dict,
                feats=None # feats object is no longer used by the scorer
            )
            transition_scores = self.scorer.score_transition(scorer_row)

            for state in self.beam:
                # Candidate: 'O' (No Break)
                if nxt:
                    if self._is_hard_ok_O(state.line_num, state.line_len, len(nxt.w)):
                        new_line_len = state.line_len + 1 + len(nxt.w)
                        limit_key = f"line{state.line_num}"
                        constraints = self.cfg.line_length_constraints.get(limit_key, {})
                        soft_target = constraints.get("soft_target", 37)
                        soft_min = constraints.get("soft_min", 0)
                        over_scale = constraints.get("soft_over_penalty_scale", 0.1)
                        under_scale = constraints.get("soft_under_penalty_scale", 0.05)
                        line_len_penalty = 0.0
                        if new_line_len > soft_target:
                            overage = new_line_len - soft_target
                            line_len_penalty += ((overage ** 2) * over_scale) / self.line_len_leniency
                        if soft_min and new_line_len < soft_min:
                            shortfall = soft_min - new_line_len
                            line_len_penalty += ((shortfall ** 2) * under_scale) / self.line_len_leniency
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
    
    segmenter = Segmenter(tokens, scorer, cfg)
    final_breaks = segmenter.run()

    return [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]
