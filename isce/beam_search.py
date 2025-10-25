"""Beam search segmentation with lookahead-aware heuristics.

This module hosts the core implementation of ISCE's beam search. The logic has
accumulated a number of heuristics over the years, so we keep the code heavily
documented to make the data flow and motivations explicit. In particular, the
scorer now receives a transition context describing the partially written block
in order to penalize prospective line breaks that would leave unreasonably short
orphan lines.
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
    """Stateful orchestrator for the beam search segmentation process.

    The :class:`Segmenter` encapsulates the outer loop of the captioning beam
    search. It iterates through tokens, expands each hypothesis with the three
    allowed break types (``O``, ``LB``, ``SB``), and relies on
    :class:`~isce.scorer.Scorer` for the actual scoring mechanics. The class also
    carries a handful of pre-computed settings derived from the active
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
    short_line_penalty:
        Optional override letting the scorer fine-tune penalties for sub-minimal
        or single-word lines.
    allowed_proper_nouns:
        Normalised set of proper nouns that may appear as single-word captions
        without being considered violations.
    lookahead_width:
        Number of future tokens to pass to the scorer when computing transition
        scores.
    """
    def __init__(self, tokens: List[Token], scorer: Scorer, cfg: Config):
        self.tokens = tokens
        self.scorer = scorer
        self.cfg = cfg
        self.beam: List[PathState] = []
        self.line_len_leniency = self.scorer.sl.get("line_length_leniency", 1.0)
        self.orphan_leniency = self.scorer.sl.get("orphan_leniency", 1.0)
        self.fallback_sb_penalty = float(self.scorer.sl.get("fallback_sb_penalty", FALLBACK_SB_PENALTY))
        self.lookahead_width = getattr(cfg, "lookahead_width", 0)
        self.short_line_penalty = float(self.scorer.sl.get("single_word_line_penalty", 0.0))
        allowed_proper_nouns = getattr(self.cfg, "allowed_single_word_proper_nouns", tuple())
        self.allowed_proper_nouns = {noun.strip().lower() for noun in allowed_proper_nouns}

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
        return sum(len(token.w) for token in line_tokens) + (len(line_tokens) - 1)

    def _is_allowed_single_word(self, token: Token) -> bool:
        if token.pos != "PROPN":
            return False
        stripped = token.w.rstrip(".,!?;:\"")
        return stripped.lower() in self.allowed_proper_nouns

    def _block_profiles(
        self, state: PathState, block_start_idx: int, end_idx: int
    ) -> tuple[List[Token], List[BreakType], List[List[Token]]]:
        """Construct token, break, and line views for a completed block.

        The scorer expects both the raw token payloads and the per-line
        segmentation decisions when evaluating a block break. This helper
        recreates those structures from the running path state so that callers
        can hand the scorer a self-contained snapshot of the block we are about
        to close.
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
        """Report soft violations found within the prospective block lines.

        The current fallback strategy applies manual penalties when we are
        forced to emit a block despite failing soft constraints. Returning the
        violation labels lets the caller scale those penalties proportionally
        while keeping the core heuristics encapsulated.
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

    def _estimate_second_line(self, current_idx: int) -> tuple[int, int]:
        """Estimate room available for a hypothetical second line.

        When the current path is still on the first line of a block we ask the
        scorer to evaluate what a line break would look like. That requires a
        rough idea of how many characters and words could fit on the second
        line before encountering a natural stopping point such as punctuation,
        a speaker change, or the soft length target. The heuristic is simple and
        intentionally deterministic so that unit tests can reason about its
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
        """Construct the lookahead context passed into the scorer.

        The :class:`~isce.scorer.Scorer` only works with serialisable
        dictionaries, so we project the dataclass tokens into dictionaries and
        package the relevant line metrics. ``projected_second_line_*`` is
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
            lookahead_tokens = None
            if self.lookahead_width > 0:
                # Expose a shallow copy of the upcoming tokens so the scorer can
                # apply lookahead heuristics without mutating the canonical list.
                future_slice = self.tokens[i + 1 : i + 1 + self.lookahead_width]
                if future_slice:
                    lookahead_tokens = tuple(dict(t.__dict__) for t in future_slice)

            scorer_row = TokenRow(
                token=token_dict,
                nxt=nxt_dict,
                feats=None,  # feats object is no longer used by the scorer
                lookahead=lookahead_tokens,
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
                if self._is_hard_ok_SB(state, i):
                    block_tokens, block_breaks, _ = self._block_profiles(state, state.block_start_idx, i)
                    block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                    block_score = self.scorer.score_block(block_token_dicts, block_breaks)
                    score = state.score + transition_scores["SB"] + block_score
                    next_word_len = len(nxt.w) if nxt else 0
                    candidates.append(PathState(score=score, line_num=1, line_len=next_word_len, block_start_idx=i + 1, breaks=state.breaks + ("SB",)))

            if not candidates and self.beam:
                # The hard constraints occasionally paint the search into a
                # corner—for example when the minimum block duration has not yet
                # been satisfied. Rather than crashing we emit a forced block
                # break from the best surviving hypothesis and assess a manual
                # penalty so that the search only uses this escape hatch when no
                # feasible alternative exists.
                fallback_state = self.beam[0]
                fallback_context = self._build_transition_context(fallback_state, i)
                fallback_scores = self.scorer.score_transition(scorer_row, fallback_context)
                block_tokens, block_breaks, lines = self._block_profiles(
                    fallback_state, fallback_state.block_start_idx, i
                )
                block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                block_score = self.scorer.score_block(block_token_dicts, block_breaks) if block_token_dicts else 0.0
                next_word_len = len(nxt.w) if nxt else 0
                violations = self._line_violations(lines)
                if violations:
                    per_violation_penalty = (
                        self.short_line_penalty if self.short_line_penalty > 0 else self.fallback_sb_penalty
                    )
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
