# C:\dev\Captions_Formatter\Formatter_machine\isce\beam_search.py

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Optional
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType
from .scorer import Scorer
from scripts.train_model import TokenRow as ScorerTokenRow
from .config import Config

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
    A stateful class to manage the beam search segmentation process.
    """
    def __init__(self, tokens: List[Token], scorer: Scorer, cfg: Config):
        self.tokens = tokens
        self.scorer = scorer
        self.cfg = cfg
        self.beam: List[PathState] = []
        self.line_len_leniency = self.scorer.sl.get("line_length_leniency", 1.0)
        self.orphan_leniency = self.scorer.sl.get("orphan_leniency", 1.0)

    def _is_hard_ok_O(self, line_num: int, line_len: int, next_word_len: int) -> bool:
        limit_key = f"line{line_num}"
        hard_limit = self.cfg.line_length_constraints.get(limit_key, {}).get("hard_limit", 42)
        return (line_len + 1 + next_word_len) <= hard_limit

    def _is_hard_ok_LB(self, line_num: int) -> bool:
        return line_num == 1

    def _is_hard_ok_SB(self, block_start_idx: int, current_idx: int) -> bool:
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
        if not self.tokens:
            return []

        initial_state = PathState(score=0.0, line_num=1, line_len=len(self.tokens[0].w), block_start_idx=0, breaks=())
        self.beam = [initial_state]

        for i, token in tqdm(enumerate(self.tokens), total=len(self.tokens), desc="Segmenting", unit="token"):
            candidates: List[PathState] = []
            is_last_token = (i == len(self.tokens) - 1)
            nxt = self.tokens[i + 1] if not is_last_token else None
            
            # Create the dictionary-based TokenRow required by the refactored scorer
            scorer_row = ScorerTokenRow(
                token=token.__dict__, 
                nxt=nxt.__dict__ if nxt else None,
                feats=None # feats object is no longer used by the scorer
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
                if nxt and self._is_hard_ok_LB(state.line_num):
                    orphan_penalty = 0.0
                    if i + 2 < len(self.tokens) and self.tokens[i + 2].is_sentence_final:
                        orphan_penalty = 2.5
                    elif i + 1 < len(self.tokens) and self.tokens[i + 1].is_sentence_final:
                        orphan_penalty = 5.0
                    score = state.score + transition_scores["LB"] - (orphan_penalty * self.orphan_leniency)
                    candidates.append(PathState(score=score, line_num=2, line_len=len(nxt.w), block_start_idx=state.block_start_idx, breaks=state.breaks + ("LB",)))

                # Candidate: 'SB' (Block Break)
                if self._is_hard_ok_SB(state.block_start_idx, i):
                    block_token_dicts = [t.__dict__ for t in self.tokens[state.block_start_idx : i + 1]]
                    block_breaks = list(state.breaks[state.block_start_idx:]) + ["SB"]
                    block_score = self.scorer.score_block(block_token_dicts, block_breaks)
                    score = state.score + transition_scores["SB"] + block_score
                    next_word_len = len(nxt.w) if nxt else 0
                    candidates.append(PathState(score=score, line_num=1, line_len=next_word_len, block_start_idx=i + 1, breaks=state.breaks + ("SB",)))

            if not candidates and self.beam:
                fallback_state = self.beam[0]
                if self._is_hard_ok_SB(fallback_state.block_start_idx, i):
                    self.beam[0] = replace(fallback_state, breaks=fallback_state.breaks + ("SB",))
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
    """High-level wrapper for the stateful Segmenter class."""
    if not tokens:
        return []
    
    segmenter = Segmenter(tokens, scorer, cfg)
    final_breaks = segmenter.run()

    output_tokens = []
    for i, token in enumerate(tokens):
        token_dict = token.__dict__
        token_dict["break_type"] = final_breaks[i]
        # Create a new Token instance with the final break_type
        output_tokens.append(Token(**token_dict))
        
    return output_tokens
