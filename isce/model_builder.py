# C:\dev\Captions_Formatter\Formatter_machine\isce\model_builder.py

from __future__ import annotations
import json
import math
import statistics as stats
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from scripts.train_model import TokenRow # Use the TokenRow from the trainer script

# --- Helper Functions (Updated for Dictionary Input) ---
def bin_pause_z(z: float | None) -> str:
    if z is None: return "pz:none"
    if z < 0: return "pz:<0"
    if z < 0.5: return "pz:0-0.5"
    if z < 1.0: return "pz:0.5-1.0"
    if z < 2.0: return "pz:1.0-2.0"
    return "pz:>2.0"

def punct_class(tok: dict) -> str:
    w = tok.get("w", "")
    if not w: return "p:none"
    char = w.strip()[-1] if w.strip() else ''
    if char in {".", "?", "!"}: return "p:final"
    if char == ",": return "p:comma"
    return "p:none"

def bin_rel_pos(x: float | None) -> str:
    # This feature is not currently being generated in the new pipeline,
    # but we keep the helper for future compatibility.
    if x is None: return "rp:none"
    if x <= 0.25: return "rp:early"
    if x < 0.8: return "rp:mid"
    return "rp:late"

def log_odds(p: float, eps: float = 1e-6) -> float:
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))

# --- derive_constraints function (Updated for Dictionary Input) ---
def derive_constraints(corpus_paths: List[str], fallback_cfg: Config) -> Dict[str, Any]:
    """Analyzes a corpus of pre-enriched JSON files to derive data-driven constraints."""
    block_durs, cps_vals, balances = [], [], []
    line1_lengths, line2_lengths = [], []
    LONG_PAUSE_THRESHOLD_MS = 500

    print(f"Analyzing {len(corpus_paths)} files to derive constraints...")
    for path in tqdm(corpus_paths, desc="Deriving Constraints"):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            tokens = data.get("tokens", [])
            if not tokens: continue
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"\nWarning: Skipping file {path} due to error: {e}")
            continue
        
        start_idx = 0
        for i, token in enumerate(tokens):
            if token.get("break_type") == "SB":
                end_idx = i
                block_tokens = tokens[start_idx : end_idx + 1]
                if not block_tokens: continue

                gross_duration = max(1e-6, block_tokens[-1].get("end", 0.0) - block_tokens[0].get("start", 0.0))
                block_durs.append(gross_duration)
                
                speech_time_duration = 0.0
                for j, tok in enumerate(block_tokens):
                    speech_time_duration += (tok.get("end", 0.0) - tok.get("start", 0.0))
                    if j < len(block_tokens) - 1 and tok.get("pause_after_ms", 0) < LONG_PAUSE_THRESHOLD_MS:
                        speech_time_duration += tok.get("pause_after_ms", 0) / 1000.0
                speech_time_duration = max(1e-6, speech_time_duration)

                def count_chars(token_slice: List[dict]) -> int:
                    if not token_slice: return 0
                    return sum(len(t.get("w", "")) for t in token_slice) + (len(token_slice) - 1)

                line_break_token_idx = next((j for j, t in enumerate(block_tokens) if t.get("break_type") == "LB"), -1)
                if line_break_token_idx != -1:
                    len1 = count_chars(block_tokens[:line_break_token_idx + 1])
                    len2 = count_chars(block_tokens[line_break_token_idx + 1:])
                    line1_lengths.append(len1); line2_lengths.append(len2)
                    total_chars = len1 + len2
                    if len1 > 0 and len2 > 0: balances.append(len1 / len2)
                else:
                    length = count_chars(block_tokens)
                    line1_lengths.append(length)
                    total_chars = length
                
                cps_vals.append(total_chars / speech_time_duration)
                start_idx = i + 1
            
    def percentile(data: List[float], p: float) -> float | None:
        return float(np.percentile(data, p)) if data else None

    constraints = {
        "min_block_duration_s": float(percentile(block_durs, 1.0) or fallback_cfg.min_block_duration_s),
        "max_block_duration_s": float(percentile(block_durs, 99.0) or fallback_cfg.max_block_duration_s),
        "ideal_cps_median": float(stats.median(cps_vals)) if cps_vals else 14.0,
        "ideal_cps_iqr": [percentile(cps_vals, 25), percentile(cps_vals, 75)] if cps_vals else [10.0, 18.0],
        "ideal_balance_iqr": [percentile(balances, 25), percentile(balances, 75)] if balances else [0.7, 1.4],
        "line1": {"soft_target": int(percentile(line1_lengths, 75) or 37), "hard_limit": int(percentile(line1_lengths, 99.5) or 42)},
        "line2": {"soft_target": int(percentile(line2_lengths, 75) or 37), "hard_limit": int(percentile(line2_lengths, 99.5) or 42)}
    }
    return constraints

# --- Centralized Feature Extraction Logic (Updated for Dictionary Input) ---
def create_feature_row(row: TokenRow, cfg: Config) -> dict:
    """Creates a dictionary of features for a single breakpoint (TokenRow of dicts)."""
    token = row.token
    nxt = row.nxt if row.nxt else {}

    pos_bigram = f"{token.get('pos', 'none')}|{nxt.get('pos', 'none')}"
    is_cap_mid = token.get('w', '') and token['w'][0].isupper() and not token.get('is_sentence_initial', False)
    splits_cap_pair = bool(is_cap_mid and nxt.get('w', '') and nxt['w'][0].isupper() and not nxt.get('is_sentence_initial', False))

    base_features = {
        "pause_z_bin": bin_pause_z(token.get("pause_z")),
        "punct_class": punct_class(token),
        "pos_bigram": f"pb:{pos_bigram}",
        "splits_capital": "cap:split" if splits_cap_pair else "cap:ok",
        "rel_pos_bin": "rp:none", # Placeholder
        "break_before_glue": str(token.get("num_unit_glue", False)),
        "is_dangling_eos": "False", # Placeholder
        "outcome": token.get("break_type"),
        "speaker_change": str(token.get("speaker_change", False)),
        "starts_with_dash": str(token.get("starts_with_dialogue_dash", False)),
    }
    
    base_features['interact_punct_pause'] = f"pp:{base_features['punct_class']}_{base_features['pause_z_bin']}"
    if base_features['punct_class'] in ['p:comma', 'p:final']:
        base_features['interact_punct_syntax'] = f"ps:{base_features['punct_class']}_{base_features['pos_bigram']}"
    else:
        base_features['interact_punct_syntax'] = "ps:none"
        
    return base_features

def build_weights(df: pd.DataFrame, cfg: Config, alpha: float = 0.1, sample_weights: pd.Series = None) -> Dict[str, Any]:
    """Builds a statistical model from a pre-built feature DataFrame."""
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot build weights.")

    if sample_weights is not None:
        df['sample_weight'] = sample_weights.reindex(df.index).fillna(1.0)
    else:
        df['sample_weight'] = 1.0

    outcomes = ["O", "LB", "SB"]
    weights = {}

    feature_groups = {
        "prosody": ["pause_z_bin"],
        "punctuation": ["punct_class"],
        "syntax": ["pos_bigram"],
        "capitalization": ["splits_capital"],
        "cohesion": ["break_before_glue"],
        "position": ["rel_pos_bin"],
        "structural_heuristics": ["is_dangling_eos", "starts_with_dash"],
        "speaker_change_feature": ["speaker_change"],
        "interaction_punct_pause": ["interact_punct_pause"],
        "interaction_punct_syntax": ["interact_punct_syntax"]
    }

    for group, features in feature_groups.items():
        weights[group] = {}
        for feature in features:
            if feature not in df.columns:
                print(f"Warning: Feature '{feature}' not found in DataFrame. Skipping.")
                continue
            counts = df.groupby([feature, 'outcome'])['sample_weight'].sum().unstack(fill_value=0) + alpha
            for out in outcomes:
                if out not in counts.columns:
                    counts[out] = alpha
            row_totals = counts.sum(axis=1)
            probs = counts.div(row_totals, axis=0)
            for feature_value, row in probs.iterrows():
                weights[group][feature_value] = {
                    outcome: log_odds(row.get(outcome, 0.0)) for outcome in outcomes
                }
                
    return weights