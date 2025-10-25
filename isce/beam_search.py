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
from typing import Dict, List, Optional, Sequence, Tuple
from heapq import nlargest
from tqdm import tqdm

from .types import Token, BreakType, TokenRow, TransitionContext
from .scorer import Scorer
from .config import Config

FALLBACK_SB_PENALTY = 25.0
DISAGREEMENT_MARGIN = 5.0
LOCAL_REFINEMENT_MIN_BEAM = 5
BALANCE_RATIO_THRESHOLD = 2.5
LOCAL_REFINEMENT_IMPROVEMENT = 0.5


def _token_to_row_dict(token: Optional[Token], *, reverse: bool = False) -> Optional[dict]:
    if token is None:
        return None

    data = dict(token.__dict__)
    if reverse:
        pause_after = data.get("pause_after_ms", 0)
        pause_before = data.get("pause_before_ms", 0)
        data["pause_after_ms"] = pause_before
        data["pause_before_ms"] = pause_after

        if "pause_z" in data:
            data["pause_z"] = -data["pause_z"]

        if "is_sentence_initial" in data or "is_sentence_final" in data:
            data["is_sentence_initial"], data["is_sentence_final"] = (
                data.get("is_sentence_final", False),
                data.get("is_sentence_initial", False),
            )

        if data.get("relative_position") is not None:
            data["relative_position"] = 1.0 - float(data["relative_position"])

    return data


def _compute_transition_scores(
    tokens: Sequence[Token],
    scorer: Scorer,
    cfg: Config,
    *,
    reverse: bool = False,
) -> List[Dict[str, float]]:
    if not tokens:
        return []

    lookahead_width = getattr(cfg, "lookahead_width", 0)

    if not reverse:
        results: List[Dict[str, float]] = []
        for idx, token in enumerate(tokens):
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            lookahead = None
            if lookahead_width > 0:
                future_slice = tokens[idx + 1 : idx + 1 + lookahead_width]
                if future_slice:
                    lookahead = tuple(_token_to_row_dict(t) for t in future_slice)

            row = TokenRow(
                token=_token_to_row_dict(token) or {},
                nxt=_token_to_row_dict(nxt) if nxt else None,
                feats=None,
                lookahead=lookahead,
            )
            results.append(scorer.score_transition(row))
        return results

    reversed_tokens = list(reversed(tokens))
    scores: List[Optional[Dict[str, float]]] = [None] * len(tokens)

    for ridx, token in enumerate(reversed_tokens):
        nxt = reversed_tokens[ridx + 1] if ridx + 1 < len(reversed_tokens) else None
        lookahead = None
        if lookahead_width > 0:
            future_slice = reversed_tokens[ridx + 1 : ridx + 1 + lookahead_width]
            if future_slice:
                lookahead = tuple(_token_to_row_dict(t, reverse=True) for t in future_slice)

        row = TokenRow(
            token=_token_to_row_dict(token, reverse=True) or {},
            nxt=_token_to_row_dict(nxt, reverse=True) if nxt else None,
            feats=None,
            lookahead=lookahead,
        )
        forward_idx = len(tokens) - 1 - ridx
        scores[forward_idx] = scorer.score_transition(row)

    for idx, sc in enumerate(scores):
        if sc is None:
            token = tokens[idx]
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            lookahead = None
            if lookahead_width > 0:
                future_slice = tokens[idx + 1 : idx + 1 + lookahead_width]
                if future_slice:
                    lookahead = tuple(_token_to_row_dict(t) for t in future_slice)
            row = TokenRow(
                token=_token_to_row_dict(token) or {},
                nxt=_token_to_row_dict(nxt) if nxt else None,
                feats=None,
                lookahead=lookahead,
            )
            scores[idx] = scorer.score_transition(row)

    return [dict(sc) for sc in scores if sc is not None]


def _best_choice_and_margin(scores: Dict[str, float]) -> Tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return "O", 0.0
    best_score = ordered[0][1]
    runner_up = ordered[1][1] if len(ordered) > 1 else best_score
    return ordered[0][0], best_score - runner_up


def _blend_transition_scores(
    forward: Sequence[Dict[str, float]],
    backward: Sequence[Dict[str, float]],
) -> List[Optional[Dict[str, float]]]:
    blended: List[Optional[Dict[str, float]]] = []
    for fwd, bwd in zip(forward, backward):
        forward_choice, forward_margin = _best_choice_and_margin(fwd)
        backward_choice, backward_margin = _best_choice_and_margin(bwd)

        if backward_choice == "SB" and backward_margin >= DISAGREEMENT_MARGIN:
            blended.append(dict(bwd))
            continue
        if forward_choice == "SB" and forward_margin >= DISAGREEMENT_MARGIN:
            blended.append(dict(fwd))
            continue

        if (
            forward_choice != backward_choice
            and forward_margin >= DISAGREEMENT_MARGIN
            and backward_margin >= DISAGREEMENT_MARGIN
        ):
            blended.append(dict(fwd))
            continue

        merged = {outcome: (fwd[outcome] + bwd[outcome]) / 2.0 for outcome in fwd.keys()}
        blended.append(merged)

    if len(forward) > len(backward):
        for idx in range(len(backward), len(forward)):
            blended.append(dict(forward[idx]))
    elif len(backward) > len(forward):
        for idx in range(len(forward), len(backward)):
            blended.append(dict(backward[idx]))

    return blended

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

    The :class:`Segmenter` walks token-by-token through the transcript, expands
    the active hypotheses (represented by :class:`PathState` instances), and
    delegates all scoring to :class:`~isce.scorer.Scorer`.  The class also keeps
    a handful of derived configuration knobs so that the inner loops remain
    lean and the heuristics stay discoverable.

    Attributes
    ----------
    tokens:
        Ordered sequence of :class:`~isce.types.Token` objects awaiting
        segmentation.
    scorer:
        The :class:`~isce.scorer.Scorer` instance providing transition and block
        scores.
    cfg:
        Parsed :class:`~isce.config.Config` describing the active constraints.
    beam:
        Mutable collection of the best :class:`PathState` hypotheses explored so
        far.
    line_len_leniency:
        Multiplier used to soften penalties once a line exceeds its soft target.
    orphan_leniency:
        Multiplier for discouraging ``LB`` decisions that would strand a short
        second line.
    fallback_sb_penalty:
        Manual penalty applied when we are forced to emit an ``SB`` to satisfy
        a hard constraint.
    lookahead_width:
        Number of future tokens to pass into the scorer when computing
        transition scores.
    short_line_penalty:
        Optional override used when scoring fallback ``SB`` blocks that contain
        short or single-word lines.
    allowed_proper_nouns:
        Normalised allowlist of proper nouns that may appear as single-word
        captions without triggering penalties.
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
        allowed = getattr(self.cfg, "allowed_single_word_proper_nouns", tuple())
        self.allowed_proper_nouns = {noun.strip().lower() for noun in allowed}
        self.last_path_score: Optional[float] = None

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
        """Return the perceived character count for ``line_tokens``.

        Spaces between words are included so that the total mirrors what a human
        captioner would judge on screen.
        """

        if not line_tokens:
            return 0
        return sum(len(token.w) for token in line_tokens) + max(0, len(line_tokens) - 1)

    def _is_allowed_single_word(self, token: Token) -> bool:
        """Check whether ``token`` is whitelisted for single-word captions."""

        if token.pos != "PROPN":
            return False
        stripped = token.w.rstrip(".,!?;:\"")
        return stripped.lower() in self.allowed_proper_nouns

    def _block_profiles(
        self, state: PathState, block_start_idx: int, end_idx: int
    ) -> tuple[List[Token], List[BreakType], List[List[Token]]]:
        """Return block-level metadata shared by the scoring helpers."""

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
        """Identify hard violations yielded by a candidate block."""

        violations: List[str] = []
        min_chars = self.cfg.min_chars_for_single_word_block
        for line_tokens in lines:
            if not line_tokens:
                continue
            is_single_word = len(line_tokens) == 1
            allowed_single = is_single_word and self._is_allowed_single_word(line_tokens[0])
            char_count = self._count_chars(line_tokens)
            if is_single_word and not allowed_single and char_count < min_chars:
                violations.append("single_word")
                continue
            if char_count < min_chars and not allowed_single:
                violations.append("short_line")
        return violations

    def _is_hard_ok_SB(self, state: PathState, current_idx: int) -> bool:
        """Checks if a block break (`SB`) violates hard constraints."""

        block_start_idx = state.block_start_idx
        block_tokens, _, lines = self._block_profiles(state, block_start_idx, current_idx)
        start_token = block_tokens[0]
        end_token = block_tokens[-1]
        duration = max(1e-6, end_token.end - start_token.start)
        if duration < self.cfg.min_block_duration_s:
            return False
        if self._line_violations(lines):
            return False
        return True

    def _estimate_second_line(self, current_idx: int) -> tuple[int, int]:
        """Estimate room available for a hypothetical second line.

        When the current path is still on the first line of a block we ask the
        scorer to evaluate what a line break would look like. That requires a
        rough idea of how many characters and words could fit on the second line
        before encountering a natural stopping point such as punctuation, a
        speaker change, or the soft length target. The heuristic is simple and
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
                length += 1  # space before the token
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

        pending_tokens = tuple(
            dict(t.__dict__) for t in self.tokens[state.block_start_idx : current_idx + 1]
        )
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

    def run(
        self, transition_overrides: Optional[Sequence[Optional[Dict[str, float]]]] = None
    ) -> List[BreakType]:
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

        if transition_overrides is not None and len(transition_overrides) != len(self.tokens):
            raise ValueError("transition_overrides must match the number of tokens")

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
                future_slice = self.tokens[i + 1 : i + 1 + self.lookahead_width]
                if future_slice:
                    lookahead_tokens = tuple(dict(t.__dict__) for t in future_slice)

            scorer_row = TokenRow(
                token=token_dict,
                nxt=nxt_dict,
                feats=None,  # feats object is no longer used by the scorer
                lookahead=lookahead_tokens,
            )

            override_scores = transition_overrides[i] if transition_overrides is not None else None

            for state in self.beam:
                context = self._build_transition_context(state, i)
                base_scores = self.scorer.score_transition(scorer_row, context)
                if override_scores:
                    transition_scores = {
                        outcome: base_scores[outcome] + override_scores.get(outcome, 0.0)
                        for outcome in base_scores
                    }
                else:
                    transition_scores = base_scores

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
                if self._is_hard_ok_SB(state, i):
                    block_tokens, block_breaks, lines = self._block_profiles(state, state.block_start_idx, i)
                    block_token_dicts = [dict(t.__dict__) for t in block_tokens]
                    block_score = self.scorer.score_block(block_token_dicts, block_breaks)
                    violations = self._line_violations(lines)
                    if violations and self.short_line_penalty > 0:
                        block_score -= self.short_line_penalty * len(violations)
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
                # The hard constraints may occasionally paint the search into a
                # corner (for example when the minimum block duration has not yet
                # been satisfied). Rather than crashing we emit a forced block
                # break from the best surviving hypothesis and assess a manual
                # penalty so that the search only uses this escape hatch when no
                # feasible alternative exists.
                fallback_state = self.beam[0]
                fallback_context = self._build_transition_context(fallback_state, i)
                fallback_scores = self.scorer.score_transition(scorer_row, fallback_context)
                if override_scores:
                    fallback_scores = {
                        outcome: fallback_scores.get(outcome, 0.0) + override_scores.get(outcome, 0.0)
                        for outcome in fallback_scores
                    }
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

        self.last_path_score = best_path.score
        return final_breaks


def _count_chars(tokens: Sequence[Token]) -> int:
    if not tokens:
        return 0
    return sum(len(t.w) for t in tokens) + max(0, len(tokens) - 1)


def _block_balance(block_tokens: Sequence[Token], block_breaks: Sequence[BreakType]) -> float:
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


def _score_path(tokens: Sequence[Token], breaks: Sequence[BreakType], scorer: Scorer, cfg: Config) -> float:
    if not tokens:
        return 0.0

    shadow = Segmenter(list(tokens), scorer, cfg)
    total = 0.0
    line_num = 1
    line_len = len(tokens[0].w)
    block_start_idx = 0
    line_len_leniency = scorer.sl.get("line_length_leniency", 1.0)
    orphan_leniency = scorer.sl.get("orphan_leniency", 1.0)

    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        lookahead = None
        if getattr(cfg, "lookahead_width", 0) > 0:
            future_slice = tokens[i + 1 : i + 1 + cfg.lookahead_width]
            if future_slice:
                lookahead = tuple(dict(t.__dict__) for t in future_slice)

        row = TokenRow(
            token=dict(token.__dict__),
            nxt=dict(nxt.__dict__) if nxt else None,
            feats=None,
            lookahead=lookahead,
        )

        state = PathState(
            score=0.0,
            line_num=line_num,
            line_len=line_len,
            block_start_idx=block_start_idx,
            breaks=tuple(breaks[:i]),
        )
        context = shadow._build_transition_context(state, i)
        transition_scores = scorer.score_transition(row, context)
        decision = breaks[i]

        if decision == "O":
            if nxt:
                new_line_len = line_len + 1 + len(nxt.w)
                constraints = cfg.line_length_constraints.get(f"line{line_num}", {})
                soft_target = constraints.get("soft_target", 37)
                soft_min = constraints.get("soft_min", 0)
                over_scale = constraints.get("soft_over_penalty_scale", 0.1)
                under_scale = constraints.get("soft_under_penalty_scale", 0.05)
                line_penalty = 0.0
                if new_line_len > soft_target:
                    line_penalty += ((new_line_len - soft_target) ** 2) * over_scale / line_len_leniency
                if soft_min and new_line_len < soft_min:
                    line_penalty += ((soft_min - new_line_len) ** 2) * under_scale / line_len_leniency
                total += transition_scores.get("O", 0.0) - line_penalty
                line_len = new_line_len
        elif decision == "LB":
            orphan_penalty = 0.0
            if i + 2 < len(tokens) and tokens[i + 2].is_sentence_final:
                orphan_penalty = 2.5
            elif i + 1 < len(tokens) and tokens[i + 1].is_sentence_final:
                orphan_penalty = 5.0
            total += transition_scores.get("LB", 0.0) - orphan_penalty * orphan_leniency
            line_num = 2
            line_len = len(nxt.w) if nxt else 0
        elif decision == "SB":
            block_tokens = tokens[block_start_idx : i + 1]
            block_breaks = list(breaks[block_start_idx:i]) + ["SB"]
            block_token_dicts = [dict(t.__dict__) for t in block_tokens]
            block_score = scorer.score_block(block_token_dicts, block_breaks)
            total += transition_scores.get("SB", 0.0) + block_score
            line_num = 1
            line_len = len(nxt.w) if nxt else 0
            block_start_idx = i + 1
        else:
            raise ValueError(f"Unknown break type: {decision}")

    return total


def _should_refine_block(
    block_tokens: Sequence[Token], block_breaks: Sequence[BreakType], block_score: float
) -> bool:
    if not block_tokens:
        return False
    if len(block_tokens) == 1:
        return True
    if block_score < 0.0:
        return True
    balance_ratio = _block_balance(block_tokens, block_breaks)
    return balance_ratio > BALANCE_RATIO_THRESHOLD


def _prepare_transition_overrides(
    tokens: Sequence[Token], scorer: Scorer, cfg: Config
) -> Optional[List[Optional[Dict[str, float]]]]:
    if len(tokens) <= 1:
        return None

    forward_scores = _compute_transition_scores(tokens, scorer, cfg)
    backward_scores = _compute_transition_scores(tokens, scorer, cfg, reverse=True)
    return _blend_transition_scores(forward_scores, backward_scores)


def refine_blocks(tokens: List[Token], breaks: Sequence[BreakType], scorer: Scorer, cfg: Config) -> List[BreakType]:
    if not tokens or not breaks:
        return list(breaks)

    refined_breaks = list(breaks)

    def _block_spans() -> List[tuple[int, int]]:
        boundaries: List[tuple[int, int]] = []
        start_idx = 0
        for idx, decision in enumerate(refined_breaks):
            if decision == "SB":
                boundaries.append((start_idx, idx))
                start_idx = idx + 1
        return boundaries

    block_boundaries = _block_spans()
    idx = 0
    while idx < len(block_boundaries):
        block_start, block_end = block_boundaries[idx]
        block_tokens = tokens[block_start : block_end + 1]
        block_breaks = list(refined_breaks[block_start : block_end + 1])
        block_token_dicts = [dict(t.__dict__) for t in block_tokens]
        block_score = scorer.score_block(block_token_dicts, block_breaks)

        if not _should_refine_block(block_tokens, block_breaks, block_score):
            idx += 1
            continue

        window_start = block_start
        window_end_block_idx = idx
        if len(block_tokens) == 1 and idx + 1 < len(block_boundaries):
            window_end_block_idx += 1
        window_end = block_boundaries[window_end_block_idx][1]

        window_tokens = tokens[window_start : window_end + 1]
        window_breaks = refined_breaks[window_start : window_end + 1]
        baseline_score = _score_path(window_tokens, window_breaks, scorer, cfg)

        refine_beam = max(cfg.beam_width, LOCAL_REFINEMENT_MIN_BEAM)
        refine_cfg = replace(cfg, beam_width=refine_beam, enable_refinement_pass=False)
        overrides = None
        if getattr(cfg, "enable_bidirectional_pass", False):
            overrides = _prepare_transition_overrides(window_tokens, scorer, cfg)

        window_segmenter = Segmenter(list(window_tokens), scorer, refine_cfg)
        candidate_breaks = window_segmenter.run(overrides)
        candidate_score = (
            window_segmenter.last_path_score
            if window_segmenter.last_path_score is not None
            else _score_path(window_tokens, candidate_breaks, scorer, cfg)
        )

        if candidate_score >= baseline_score + LOCAL_REFINEMENT_IMPROVEMENT:
            refined_breaks[window_start : window_end + 1] = candidate_breaks
            block_boundaries = _block_spans()
            idx = 0
            continue

        idx += 1

    return refined_breaks


def segment(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Run the beam search (and optional refinement) over the provided tokens."""

    if not tokens:
        return []

    overrides = None
    if getattr(cfg, "enable_bidirectional_pass", False):
        overrides = _prepare_transition_overrides(tokens, scorer, cfg)

    segmenter = Segmenter(tokens, scorer, cfg)
    final_breaks = segmenter.run(overrides)

    if getattr(cfg, "enable_refinement_pass", False):
        final_breaks = refine_blocks(tokens, final_breaks, scorer, cfg)

    return [replace(token, break_type=final_breaks[i]) for i, token in enumerate(tokens)]
