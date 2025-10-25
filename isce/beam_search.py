"""Beam search segmentation with lookahead-aware heuristics.

This module hosts the core implementation of ISCE's beam search.  The logic
has accumulated a number of heuristics over the years, so we keep the code
heavily documented to make the data flow and motivations explicit.  In
particular, the scorer now receives a transition context describing the
partially written block in order to penalize prospective line breaks that
would leave unreasonably short orphan lines.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType, TokenRow, TransitionContext
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
    """Stateful helper that manages the beam search segmentation process.

    The segmenter walks token-by-token through the transcript.  At each
    position it expands the active hypotheses (represented by :class:`PathState`
    instances), calls into :class:`~isce.scorer.Scorer` to evaluate potential
    break transitions, and then prunes the hypotheses back down to the
    configured beam width.  The docstring deliberately calls out the tuning
    knobs that shape the heuristics so that future changes remain grounded in
    the original intent.

    Attributes:
        tokens: Sequence of :class:`~isce.types.Token` objects awaiting
            segmentation.
        scorer: The :class:`~isce.scorer.Scorer` instance providing transition
            and block level scores.
        cfg: Parsed :class:`~isce.config.Config` containing algorithm settings.
        beam: Current top-N :class:`PathState` hypotheses under consideration.
        line_len_leniency: Scalar that softens penalties for exceeding the
            preferred line length.
        orphan_leniency: Scalar that scales the penalty for producing a short
            second line after a line break.
        fallback_sb_penalty: Manual penalty applied when the search must emit a
            forced block break to satisfy hard constraints.
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

    def _estimate_second_line(self, current_idx: int) -> tuple[int, int]:
        """Estimate room available for a hypothetical second line.

        When the current path is still on the first line of a block we ask the
        scorer to evaluate what a line break would look like.  That requires a
        rough idea of how many characters and words could fit on the second
        line before encountering a natural stopping point such as punctuation,
        a speaker change, or the soft length target.  The heuristic is simple
        and intentionally deterministic so that unit tests can reason about its
        behaviour.

        Args:
            current_idx: Index of the token currently being evaluated.

        Returns:
            A ``(characters, words)`` tuple estimating the second line content.
            ``(0, 0)`` indicates that no additional words are available.
        """
        if current_idx + 1 >= len(self.tokens):
            return 0, 0

        length = 0
        words = 0
        soft_target = self.cfg.line_length_constraints.get("line2", {}).get("soft_target", 37)

        for j in range(current_idx + 1, len(self.tokens)):
            token = self.tokens[j]
            if words > 0:
                length += 1  # account for the space preceding the token
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
        """Construct the lookahead context passed into the scorer.

        The :class:`~isce.scorer.Scorer` only works with serialisable
        dictionaries, so we project the dataclass tokens into dictionaries and
        package the relevant line metrics.  ``projected_second_line_*`` is
        provided only when the current path is still mid-first-line; otherwise
        the scorer can infer that the transition stays within the same line.
        """

        pending_tokens = tuple(dict(t.__dict__) for t in self.tokens[state.block_start_idx : current_idx + 1])
        projected_chars: int | None = None
        projected_words: int | None = None
        if state.line_num == 1 and current_idx + 1 < len(self.tokens):
            projected_chars, projected_words = self._estimate_second_line(current_idx)

        return TransitionContext(
            pending_tokens=pending_tokens,
            current_line_num=state.line_num,
            current_line_len=state.line_len,
            projected_second_line_chars=projected_chars,
            projected_second_line_words=projected_words,
        )

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
                feats=None,  # feats object is no longer used by the scorer
            )
            for state in self.beam:
                context = self._build_transition_context(state, i)
                transition_scores = self.scorer.score_transition(scorer_row, context)

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
                # The hard constraints may occasionally paint the search into a
                # corner (for example when the minimum block duration has not
                # yet been satisfied).  Rather than crashing we emit a forced
                # block break from the best surviving hypothesis and assess a
                # manual penalty so that the search only uses this escape hatch
                # when no feasible alternative exists.
                fallback_state = self.beam[0]
                fallback_context = self._build_transition_context(fallback_state, i)
                fallback_scores = self.scorer.score_transition(scorer_row, fallback_context)
                block_tokens = [dict(t.__dict__) for t in self.tokens[fallback_state.block_start_idx : i + 1]]
                block_breaks = list(fallback_state.breaks[fallback_state.block_start_idx:]) + ["SB"]
                block_score = self.scorer.score_block(block_tokens, block_breaks) if block_tokens else 0.0
                next_word_len = len(nxt.w) if nxt else 0
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
