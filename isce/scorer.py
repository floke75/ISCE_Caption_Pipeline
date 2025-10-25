# C:\dev\Captions_Formatter\Formatter_machine\isce\scorer.py
"""Provides the Scorer class for evaluating segmentation decisions.

This module is central to the beam search process. The `Scorer` class takes
the trained model weights, derived corpus constraints, and user-configurable
"sliders" to evaluate the quality of potential break decisions.

The scoring is divided into two main components:
1.  **Transition Scoring**: `score_transition` evaluates the quality of placing
    a break (`O`, `LB`, `SB`) *between* two tokens. This score is based on
    local linguistic and prosodic features.
2.  **Block Scoring**: `score_block` evaluates the quality of a *completed*
    subtitle block. This score is based on holistic properties like
    characters-per-second (CPS), line balance, and total duration, ensuring
    the final output is readable and well-formed.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from .model_builder import bin_pause_z, punct_class, bin_rel_pos
from .config import Config
from .types import BreakType, TokenRow, TransitionContext

class Scorer:
    """
    Calculates scores for captioning decisions based on a statistical model.

    The Scorer is responsible for evaluating the quality of different break
    types (`O`, `LB`, `SB`) at each potential decision point in the token
    sequence. It uses a set of learned weights from a trained model, corpus-
    derived constraints, and user-adjustable sliders to calculate scores.

    The scoring process is divided into two main parts:
    1.  `score_transition`: Scores the decision to place a break *after* a
        given token, based on local features.
    2.  `score_block`: Scores a *completed* subtitle block holistically, based
        on aggregate features like CPS and line balance.

    Attributes:
        w: A dictionary of learned weights for different features.
        c: A dictionary of corpus-derived constraints (e.g., ideal CPS range).
        sl: A dictionary of user-adjustable sliders to tune model behavior.
        structure_boost: A powerful, non-statistical bonus applied to breaks
                         that align with strong structural hints.
    """
    def __init__(self, weights: Dict, constraints: Dict, sliders: Dict, cfg: Config):
        self.w = weights
        self.c = constraints
        self.cfg = cfg
        self.sl = {
            "flow": 1.0,
            "density": 1.0,
            "balance": 1.0,
            "structure": 1.0,
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "fallback_sb_penalty": 25.0,
            "single_word_line_penalty": 0.0,
            "extreme_balance_penalty": 0.0,
            "extreme_balance_threshold": 2.5,
            "short_block_penalty": 0.0,
            "short_line_penalty": 0.0,
        }
        self.sl.update(sliders)
        self.structure_boost = self.sl.get("structure_boost", 15.0)
        self.allowed_single_word_proper_nouns = {
            noun.strip().lower() for noun in cfg.allowed_single_word_proper_nouns
        }

    def _get_weight(self, group: str, key: str, outcome: str) -> float:
        """
        Safely retrieves a specific weight from the nested weights dictionary.

        This utility function prevents errors by handling cases where a
        particular feature or outcome may not have an entry in the weights model,
        returning 0.0 as a default.

        Args:
            group: The top-level feature group (e.g., "prosody", "syntax").
            key: The specific feature key (e.g., "pz:long", "p:final").
            outcome: The break type being scored (e.g., "O", "LB", "SB").

        Returns:
            The learned weight as a float, or 0.0 if not found.
        """
        try:
            return float(self.w[group][key].get(outcome, 0.0))
        except (KeyError, AttributeError):
            return 0.0

    def score_transition(self, row: TokenRow, ctx: Optional[TransitionContext] = None) -> Dict[str, float]:
        """
        Calculates the scores for each possible break type after the current token.

        This is the core function for scoring local decisions. It aggregates
        weights from various feature groups (prosody, syntax, etc.) and applies
        strong boosts based on pre-engineered structural hints like speaker
        changes or LLM-suggested breaks.

        Args:
            row: A `TokenRow` object containing the current token and the next
                 token, structured as dictionaries.

        Returns:
            A dictionary mapping each break type ('O', 'LB', 'SB') to its
            calculated score.
        """
        scores: Dict[str, float] = {"O": 0.0, "LB": 0.0, "SB": 0.0}
        token = row.token
        nxt = row.nxt if row.nxt else {}
        lookahead = list(row.lookahead) if row.lookahead else []

        # --- Feature Discretization from pre-engineered data ---
        pz_bin = bin_pause_z(token.get("pause_z"))
        pc_class = punct_class(token)
        rp_bin = bin_rel_pos(token.get("relative_position"))
        
        pos_bigram = f"{token.get('pos', 'none')}|{nxt.get('pos', 'none')}"
        pb_key = f"pb:{pos_bigram}"
        
        is_cap_mid = token.get('w', '') and token['w'][0].isupper() and not token.get('is_sentence_initial', False)
        splits_cap_pair = bool(is_cap_mid and nxt.get('w', '') and nxt['w'][0].isupper() and not nxt.get('is_sentence_initial', False))
        cap_key = "cap:split" if splits_cap_pair else "cap:ok"
        
        glue_key = str(token.get("num_unit_glue", False))
        dangling_flag = str(token.get("is_dangling_eos", False))
        dangling_key = f"is_dangling_eos:{dangling_flag}"

        interact_pp_key = f"pp:{pc_class}_{pz_bin}"
        interact_ps_key = f"ps:{pc_class}_{pb_key}" if pc_class in ['p:comma', 'p:final'] else "ps:none"

        # --- Step 1: Calculate the score from all learned weights ---
        for outcome in scores.keys():
            learned_score = (
                self._get_weight("prosody", pz_bin, outcome) +
                self._get_weight("punctuation", pc_class, outcome) +
                self._get_weight("position", rp_bin, outcome) +
                self._get_weight("syntax", pb_key, outcome) +
                self._get_weight("capitalization", cap_key, outcome) +
                self._get_weight("cohesion", glue_key, outcome) +
                self._get_weight("structural_heuristics", dangling_key, outcome)
            )
            interaction_score = (
                self._get_weight("interaction_punct_pause", interact_pp_key, outcome) +
                self._get_weight("interaction_punct_syntax", interact_ps_key, outcome)
            )
            scores[outcome] += self.sl.get("flow", 1.0) * (learned_score + interaction_score)

        # --- Step 2: Apply structural scores from pre-engineered features ---
        sc_key = str(token.get("speaker_change", False))
        dash_flag = str(token.get("starts_with_dialogue_dash", False))
        dash_key = f"starts_with_dash:{dash_flag}"

        for outcome in scores.keys():
            scores[outcome] += self.sl.get("structure", 1.0) * (
                self._get_weight("speaker_change_feature", sc_key, outcome) +
                self._get_weight("structural_heuristics", dash_key, outcome)
            )

        # Apply the powerful, non-statistical "nudges"
        if token.get("speaker_change") or token.get("starts_with_dialogue_dash"):
            scores["SB"] += self.structure_boost
            scores["O"] -= self.structure_boost
        
        if token.get("is_llm_structural_break"):
            scores["SB"] += self.structure_boost
            scores["O"] -= self.structure_boost

        # --- Step 3: Optional lookahead heuristics ---
        self._apply_lookahead_heuristics(scores, lookahead)

        if ctx and ctx.current_line_num == 1:
            projected_words = ctx.projected_second_line_words
            projected_chars = ctx.projected_second_line_chars if ctx.projected_second_line_chars is not None else 0
            if projected_words is not None and projected_words <= 1:
                threshold = float(self.sl.get("fragment_char_threshold", 8.0))
                penalty_strength = float(self.sl.get("fragment_penalty", 6.0))
                deficit = max(0.0, threshold - projected_chars)
                severity = 1.0 if projected_words == 0 else deficit / max(threshold, 1e-6)
                scores["LB"] -= penalty_strength * severity

        return scores

    def _apply_lookahead_heuristics(self, scores: Dict[str, float], lookahead: List[dict]) -> None:
        """Apply forward-looking adjustments when future context is available.

        The transition scorer operates primarily on local features, but when the
        segmenter provides a small window of upcoming tokens we can anticipate
        high-impact structural events. This helper nudges the raw transition
        scores toward breaks that align with those events, encouraging earlier
        block or line boundaries when:

        * a speaker change is imminent,
        * strong punctuation (comma or sentence-final mark) is approaching, or
        * a long pause is about to occur.

        These heuristics are intentionally gentle—they scale by the distance to
        the future event so that remote hints do not overwhelm learned weights.

        Args:
            scores: The mutable transition score dictionary produced by
                :meth:`score_transition`.
            lookahead: A list of dictionary-style token payloads describing the
                upcoming words. An empty list leaves ``scores`` unchanged.
        """

        if not lookahead:
            return

        # Speaker changes are a strong cue for starting a new subtitle block.
        speaker_idx = next((idx for idx, future in enumerate(lookahead) if future.get("speaker_change")), None)
        if speaker_idx is not None:
            distance = speaker_idx + 1
            structural_bonus = self.structure_boost / max(1, (distance + 1))
            scores["SB"] += structural_bonus
            scores["LB"] += structural_bonus * 0.5
            scores["O"] -= structural_bonus

        # Commas often mark natural clause boundaries; bias toward a gentle break.
        comma_idx = next((idx for idx, future in enumerate(lookahead) if punct_class(future) == "p:comma"), None)
        if comma_idx is not None:
            closeness = comma_idx + 1
            flow_bonus = self.sl.get("flow", 1.0) * (0.6 / closeness)
            scores["LB"] += flow_bonus
            scores["O"] -= flow_bonus * 0.5

        # Sentence-final punctuation is an even stronger cue for ending the block.
        final_idx = next((idx for idx, future in enumerate(lookahead) if punct_class(future) == "p:final"), None)
        if final_idx is not None:
            closeness = final_idx + 1
            flow_bonus = self.sl.get("flow", 1.0) * (0.8 / closeness)
            scores["SB"] += flow_bonus
            scores["O"] -= flow_bonus * 0.5

        # Extended pauses should not straddle a subtitle boundary; favor a break.
        upcoming_pause = max((future.get("pause_before_ms") or future.get("pause_after_ms") or 0) for future in lookahead)
        if upcoming_pause >= 500:
            pause_bonus = self.sl.get("flow", 1.0) * 0.5
            scores["SB"] += pause_bonus
            scores["LB"] += pause_bonus * 0.5

    def score_block(self, block_tokens: List[dict], block_breaks: List[BreakType]) -> float:
        """
        Calculates a holistic quality score for a completed subtitle block.

        This function evaluates a finished block based on aggregate metrics
        that are crucial for readability, such as:
        -   Characters-Per-Second (CPS): How fast the subtitle appears.
        -   Line Balance: The relative length of the two lines (if a line
            break exists).
        -   Total Duration: Whether the subtitle is on screen for too long
            or too short a time.

        Scores are calculated by comparing these metrics against ideal ranges
        defined in the corpus constraints.

        Args:
            block_tokens: A list of token dictionaries within the completed block.
            block_breaks: The list of break types corresponding to the `block_tokens`.

        Returns:
            A float representing the overall quality score for the block. A higher
            score is better. Penalties are applied for violating constraints.
        """
        if not block_tokens:
            return 0.0
        score = 0.0

        def count_chars(token_slice: List[dict]) -> int:
            if not token_slice:
                return 0
            return sum(len(t.get("w", "")) for t in token_slice) + (len(token_slice) - 1)

        def is_allowed_single_word(token_dict: dict) -> bool:
            if token_dict.get("pos") != "PROPN":
                return False
            stripped = token_dict.get("w", "").rstrip(".,!?;:\"")
            return stripped.lower() in self.allowed_single_word_proper_nouns

        lines: List[List[dict]] = []
        current_line: List[dict] = []
        for idx, token in enumerate(block_tokens):
            current_line.append(token)
            break_type = block_breaks[idx] if idx < len(block_breaks) else "SB"
            if break_type in ("LB", "SB"):
                lines.append(list(current_line))
                current_line = []
        if current_line:
            lines.append(list(current_line))

        line_char_counts = [count_chars(line) for line in lines]
        total_chars = sum(line_char_counts)

        if len(lines) == 2:
            len1, len2 = line_char_counts
            balance = (len1 / max(1, len2)) if len1 and len2 else 1.0
        else:
            balance = 1.0

        gross_duration = max(1e-6, block_tokens[-1].get("end", 0.0) - block_tokens[0].get("start", 0.0))
        LONG_PAUSE_THRESHOLD_MS = 500
        speech_time_duration = 0.0
        for i, token in enumerate(block_tokens):
            speech_time_duration += (token.get("end", 0.0) - token.get("start", 0.0))
            if i < len(block_tokens) - 1 and token.get("pause_after_ms", 0) < LONG_PAUSE_THRESHOLD_MS:
                speech_time_duration += token.get("pause_after_ms", 0) / 1000.0
        speech_time_duration = max(1e-6, speech_time_duration)
        
        cps = total_chars / speech_time_duration
        cps_lo, cps_hi = self.c.get("ideal_cps_iqr", [10.0, 18.0])
        if cps_lo <= cps <= cps_hi:
            score += self.sl.get("density", 1.0) * 1.0
        else:
            median = self.c.get("ideal_cps_median", (cps_lo + cps_hi) / 2)
            distance = abs(cps - median) / max(1e-6, median)
            score -= self.sl.get("density", 1.0) * (1.0 + distance)

        if len(lines) == 2:
            bal_lo, bal_hi = self.c.get("ideal_balance_iqr", [0.7, 1.4])
            if bal_lo <= balance <= bal_hi:
                score += self.sl.get("balance", 1.0) * 0.5
            else:
                score -= self.sl.get("balance", 1.0) * 0.5

        single_word_penalty = float(self.sl.get("single_word_line_penalty", 0.0))
        if single_word_penalty:
            min_chars = self.cfg.min_chars_for_single_word_block
            penalty_lines = 0
            for line_tokens, char_count in zip(lines, line_char_counts):
                if not line_tokens:
                    continue
                is_single = len(line_tokens) == 1
                allowed_single = is_single and is_allowed_single_word(line_tokens[0])
                if is_single and not allowed_single:
                    penalty_lines += 1
                    continue
                if char_count < min_chars and not allowed_single:
                    penalty_lines += 1
            if penalty_lines:
                score -= single_word_penalty * penalty_lines

        extreme_balance_penalty = float(self.sl.get("extreme_balance_penalty", 0.0))
        extreme_threshold = float(self.sl.get("extreme_balance_threshold", 2.5))
        if len(line_char_counts) == 2 and extreme_balance_penalty and extreme_threshold > 0:
            len1, len2 = line_char_counts
            smaller = min(len1, len2)
            larger = max(len1, len2)
            if smaller == 0:
                severity = 1.0
            else:
                ratio = larger / smaller
                severity = None if ratio <= extreme_threshold else (ratio - extreme_threshold) / extreme_threshold
            if severity is not None:
                # Give increasingly large penalties as the ratio drifts farther
                # from the configured balance threshold.
                score -= extreme_balance_penalty * (1.0 + max(0.0, severity))

        block_constraints = self.cfg.line_length_constraints.get("block", {})
        min_total_chars = int(block_constraints.get("min_total_chars", 0))
        min_last_line_chars = int(block_constraints.get("min_last_line_chars", 0))

        short_block_penalty = float(self.sl.get("short_block_penalty", 0.0))
        if short_block_penalty and min_total_chars:
            if total_chars < min_total_chars:
                last_token = block_tokens[-1]
                if not last_token.get("is_sentence_final", False):
                    deficit = min_total_chars - total_chars
                    score -= short_block_penalty * deficit

        short_line_penalty = float(self.sl.get("short_line_penalty", 0.0))
        if short_line_penalty and min_last_line_chars and lines:
            last_line_len = line_char_counts[-1]
            if last_line_len < min_last_line_chars:
                deficit = min_last_line_chars - last_line_len
                score -= short_line_penalty * deficit

        if gross_duration < self.c.get("min_block_duration_s", 1.0):
            score -= 10.0
        if gross_duration > self.c.get("max_block_duration_s", 8.0):
            score -= 2.0

        return score