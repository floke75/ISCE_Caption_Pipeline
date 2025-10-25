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
    n = len(reversed_breaks)
    if n == 0:
        return []
    mapped: List[BreakType] = ["O"] * n
    for i in range(n - 1):
        mapped[i] = reversed_breaks[n - 2 - i]
    mapped[-1] = "SB"
    return mapped


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

    return [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]
