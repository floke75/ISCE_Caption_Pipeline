# C:\dev\Captions_Formatter\Formatter_machine\isce\scorer.py

from __future__ import annotations
from typing import Dict, List
# The scorer now needs to know about the TokenRow structure defined in the trainer
from scripts.train_model import TokenRow
from .model_builder import bin_pause_z, punct_class, bin_rel_pos
from .config import Config
from .types import BreakType

class Scorer:
    """
    Calculates scores for captioning decisions.
    Refactored to work with TokenRows of dictionaries and pre-engineered features.
    """
    def __init__(self, weights: Dict, constraints: Dict, sliders: Dict, cfg: Config):
        self.w = weights
        self.c = constraints
        self.sl = {"flow": 1.0, "density": 1.0, "balance": 1.0, "structure": 1.0}
        self.sl.update(sliders)
        self.structure_boost = self.sl.get("structure_boost", 15.0)

    def _get_weight(self, group: str, key: str, outcome: str) -> float:
        """Safely retrieves a weight from the nested weights dictionary."""
        try:
            return float(self.w[group][key].get(outcome, 0.0))
        except (KeyError, AttributeError):
            return 0.0

    def score_transition(self, row: TokenRow) -> Dict[str, float]:
        """Calculates scores for each possible break type (O, LB, SB) after a token."""
        scores: Dict[str, float] = {"O": 0.0, "LB": 0.0, "SB": 0.0}
        token = row.token
        nxt = row.nxt if row.nxt else {}

        # --- Feature Discretization from pre-engineered data ---
        pz_bin = bin_pause_z(token.get("pause_z"))
        pc_class = punct_class(token)
        rp_bin = "rp:none" # Placeholder
        
        pos_bigram = f"{token.get('pos', 'none')}|{nxt.get('pos', 'none')}"
        pb_key = f"pb:{pos_bigram}"
        
        is_cap_mid = token.get('w', '') and token['w'][0].isupper() and not token.get('is_sentence_initial', False)
        splits_cap_pair = bool(is_cap_mid and nxt.get('w', '') and nxt['w'][0].isupper() and not nxt.get('is_sentence_initial', False))
        cap_key = "cap:split" if splits_cap_pair else "cap:ok"
        
        glue_key = str(token.get("num_unit_glue", False))
        dangling_key = "False" # Placeholder

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
        dash_key = str(token.get("starts_with_dialogue_dash", False))

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
            scores["SB"] += self.structure_boost * 1.5 
            scores["O"] -= self.structure_boost * 1.5

        return scores

    def score_block(self, block_tokens: List[dict], block_breaks: List[BreakType]) -> float:
        """Calculates a holistic quality score for a completed block of dicts."""
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
        
        if gross_duration < self.c.get("min_block_duration_s", 1.0): score -= 10.0
        if gross_duration > self.c.get("max_block_duration_s", 8.0): score -= 2.0

        return score