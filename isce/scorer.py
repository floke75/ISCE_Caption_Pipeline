# C:\dev\Captions_Formatter\Formatter_machine\isce\scorer.py

from __future__ import annotations
from typing import Dict, List

from .model_builder import bin_pause_z, punct_class, bin_rel_pos
from .config import Config
from .types import BreakType, TokenRow

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
        self.sl = {"flow": 1.0, "density": 1.0, "balance": 1.0, "structure": 1.0}
        self.sl.update(sliders)
        self.structure_boost = self.sl.get("structure_boost", 15.0)
        self.cfg = cfg

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

    def score_transition(self, row: TokenRow) -> Dict[str, float]:
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

        return scores

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
        if not block_tokens: return 0.0
        score = 0.0
        
        def count_chars(token_slice: List[dict]) -> int:
            if not token_slice: return 0
            return sum(len(t.get("w", "")) for t in token_slice) + (len(token_slice) - 1)
        
        lb_idx = next((i for i, br in enumerate(block_breaks) if br == "LB"), -1)
        if lb_idx != -1:
            len1 = count_chars(block_tokens[:lb_idx + 1])
            len2 = count_chars(block_tokens[lb_idx + 1:])
            total_chars = len1 + len2
            balance = (len1 / max(1, len2)) if len1 and len2 else 1.0
        else:
            total_chars = count_chars(block_tokens)
            balance = 1.0
            len2 = total_chars

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

        if lb_idx != -1:
            bal_lo, bal_hi = self.c.get("ideal_balance_iqr", [0.7, 1.4])
            if bal_lo <= balance <= bal_hi:
                score += self.sl.get("balance", 1.0) * 0.5
            else:
                score -= self.sl.get("balance", 1.0) * 0.5
        
        block_constraints = self.cfg.line_length_constraints.get("block", {})
        min_total_chars = block_constraints.get("min_total_chars", 0)
        min_last_line_chars = block_constraints.get("min_last_line_chars", 0)
        is_sentence_final = bool(block_tokens[-1].get("is_sentence_final"))
        if not is_sentence_final:
            short_block_penalty = self.sl.get("short_block_penalty", 1.0)
            if min_total_chars and total_chars < min_total_chars:
                deficit = min_total_chars - total_chars
                score -= short_block_penalty * deficit
            last_line_len = len2 if lb_idx != -1 else total_chars
            if min_last_line_chars and last_line_len < min_last_line_chars:
                deficit = min_last_line_chars - last_line_len
                score -= self.sl.get("short_line_penalty", short_block_penalty) * deficit

        if gross_duration < self.c.get("min_block_duration_s", 1.0): score -= 10.0
        if gross_duration > self.c.get("max_block_duration_s", 8.0): score -= 2.0

        return score