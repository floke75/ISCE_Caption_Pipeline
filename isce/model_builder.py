"""Core logic for training the statistical segmentation model.

This module provides the functions necessary to train the ISCE model from a
corpus of labeled data. The process involves several key steps:

1.  **Constraint Derivation**: The `derive_constraints` function analyzes the
    training corpus to learn statistical properties about human-made subtitles,
    such as typical block durations and line lengths. These become the guiding
    constraints for the segmentation algorithm.
2.  **Feature Extraction**: The `create_feature_row` function transforms the
    rich, continuous data from the enriched tokens into a set of discrete,
    categorical features suitable for a statistical model. This includes
    binning continuous values and creating interaction features.
3.  **Weight Building**: The `build_weights` function takes the featurized data
    and calculates the conditional probability of each break type (`O`, `LB`,
    `SB`) given each feature. These probabilities are converted to log-odds to
    form the final `model_weights.json` file.
"""
from __future__ import annotations
import json
import math
import statistics as stats
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import Config
from .types import TokenRow

# --- Helper Functions (Updated for Dictionary Input) ---
def bin_pause_z(z: float | None) -> str:
    """Categorizes a pause z-score into discrete bins."""
    if z is None: return "pz:none"
    if z < 0: return "pz:<0"
    if z < 0.5: return "pz:0-0.5"
    if z < 1.0: return "pz:0.5-1.0"
    if z < 2.0: return "pz:1.0-2.0"
    return "pz:>2.0"

def punct_class(tok: dict) -> str:
    """Classifies the final punctuation of a token."""
    w = tok.get("w", "")
    if not w: return "p:none"
    char = w.strip()[-1] if w.strip() else ''
    if char in {".", "?", "!"}: return "p:final"
    if char == ",": return "p:comma"
    return "p:none"

def bin_rel_pos(x: float | None) -> str:
    """Categorizes a relative position float (0.0 to 1.0) into discrete bins."""
    # This feature is not currently being generated in the new pipeline,
    # but we keep the helper for future compatibility.
    if x is None: return "rp:none"
    if x <= 0.25: return "rp:early"
    if x < 0.8: return "rp:mid"
    return "rp:late"

def log_odds(p: float, eps: float = 1e-6) -> float:
    """
    Converts a probability to log-odds.

    Args:
        p: The probability (0.0 to 1.0).
        eps: A small epsilon value to prevent division by zero or log(0).

    Returns:
        The log-odds representation of the probability.
    """
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))

# --- Feature helpers shared with the scorer ---
def _sanitize_value(value: Optional[str]) -> str:
    """Normalise spaCy feature strings for stable key generation."""
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    return text.lower().replace("|", "+").replace(" ", "_")


def _normalize_morph(value: Optional[str]) -> str:
    """Canonicalise morphological feature strings."""
    if value is None:
        return "none"
    text = str(value).strip()
    if not text:
        return "none"
    parts = [part.strip().lower().replace(" ", "_") for part in text.split("|") if part.strip()]
    if not parts:
        return "none"
    parts.sort()
    return "+".join(part.replace("|", "+") for part in parts)


def lemma_bigram_key(token: dict, nxt: Optional[dict]) -> str:
    return f"lb:{_sanitize_value(token.get('lemma'))}|{_sanitize_value((nxt or {}).get('lemma'))}"


def tag_bigram_key(token: dict, nxt: Optional[dict]) -> str:
    return f"tb:{_sanitize_value(token.get('tag'))}|{_sanitize_value((nxt or {}).get('tag'))}"


def morph_bigram_key(token: dict, nxt: Optional[dict]) -> str:
    return f"mb:{_normalize_morph(token.get('morph'))}|{_normalize_morph((nxt or {}).get('morph'))}"


def dep_bigram_key(token: dict, nxt: Optional[dict]) -> str:
    return f"db:{_sanitize_value(token.get('dep'))}|{_sanitize_value((nxt or {}).get('dep'))}"


def head_position_key(token: dict) -> str:
    token_idx = token.get("token_index")
    head_idx = token.get("head_idx")
    try:
        token_idx = int(token_idx) if token_idx is not None else None
        head_idx = int(head_idx) if head_idx is not None else None
    except (TypeError, ValueError):
        return "head_pos:unknown"

    if token_idx is None or head_idx is None:
        return "head_pos:unknown"

    diff = head_idx - token_idx
    if diff == 0:
        bucket = "self"
    elif diff == 1:
        bucket = "next"
    elif diff == -1:
        bucket = "prev"
    elif diff > 1:
        bucket = "ahead"
    else:
        bucket = "back"
    return f"head_pos:{bucket}"


def dependency_link_key(token: dict, nxt: Optional[dict]) -> str:
    token_idx = token.get("token_index")
    nxt = nxt or {}
    nxt_idx = nxt.get("token_index")
    head_idx = token.get("head_idx")
    nxt_head_idx = nxt.get("head_idx")

    try:
        token_idx = int(token_idx) if token_idx is not None else None
        nxt_idx = int(nxt_idx) if nxt_idx is not None else None
        head_idx = int(head_idx) if head_idx is not None else None
        nxt_head_idx = int(nxt_head_idx) if nxt_head_idx is not None else None
    except (TypeError, ValueError):
        return "dep_link:none"

    if token_idx is None or nxt_idx is None:
        return "dep_link:none"

    link = "none"
    if head_idx == nxt_idx:
        link = "token->next"
    elif nxt_head_idx == token_idx:
        link = "next->token"

    return f"dep_link:{link}"


# --- derive_constraints function (Updated for Dictionary Input) ---
def derive_constraints(corpus_paths: List[str], fallback_cfg: Config) -> Dict[str, Any]:
    """
    Analyzes a training corpus to derive data-driven constraints for the model.

    This function iterates through a collection of labeled training files and
    calculates statistics about the human-generated subtitles. These statistics
    are then used to define the "ideal" characteristics of a subtitle block,
    such as its duration, characters-per-second (CPS), and line length. Using
    percentiles helps create robust constraints that are representative of the
    corpus.

    Args:
        corpus_paths: A list of file paths to the enriched training JSON files.
        fallback_cfg: A `Config` object to provide default values if the
                      corpus is empty or analysis fails.

    Returns:
        A dictionary containing the derived constraints, which will be saved
        as `constraints.json`.
    """
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

                edited_flags = [tok.get("is_edited_transcript") for tok in block_tokens]
                if edited_flags and all(flag is False for flag in edited_flags):
                    start_idx = i + 1
                    continue

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
    """
    Creates a flat dictionary of discrete features for a single decision point.

    This function takes a `TokenRow` (representing the boundary between two
    tokens) and transforms its rich, continuous data into a set of discrete,
    string-based features that can be used by the statistical model. It uses
    the various `bin_*` and `*_class` helper functions to perform this
    discretization. It also creates interaction features by combining
    individual features.

    Args:
        row: The `TokenRow` object containing the current and next tokens as dicts.
        cfg: The main `Config` object (currently unused but kept for future
             compatibility).

    Returns:
        A dictionary where keys are feature names and values are the
        discretized feature values for the given decision point.
    """
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
        "rel_pos_bin": bin_rel_pos(token.get("relative_position")),
        "break_before_glue": str(token.get("num_unit_glue", False)),
        "is_dangling_eos": str(token.get("is_dangling_eos", False)),
        "outcome": str(token.get("break_type") or "O"),
        "speaker_change": str(token.get("speaker_change", False)),
        "starts_with_dash": str(token.get("starts_with_dialogue_dash", False)),
    }

    base_features.update(
        {
            "lemma_bigram": lemma_bigram_key(token, nxt),
            "tag_bigram": tag_bigram_key(token, nxt),
            "morph_bigram": morph_bigram_key(token, nxt),
            "dep_bigram": dep_bigram_key(token, nxt),
            "head_position": head_position_key(token),
            "head_link": dependency_link_key(token, nxt),
        }
    )

    base_features['interact_punct_pause'] = f"pp:{base_features['punct_class']}_{base_features['pause_z_bin']}"
    if base_features['punct_class'] in ['p:comma', 'p:final']:
        base_features['interact_punct_syntax'] = f"ps:{base_features['punct_class']}_{base_features['pos_bigram']}"
    else:
        base_features['interact_punct_syntax'] = "ps:none"
        
    return base_features

def build_weights(df: pd.DataFrame, cfg: Config, alpha: float = 0.1, sample_weights: pd.Series = None) -> Dict[str, Any]:
    """
    Builds the statistical model weights from a featurized DataFrame.

    This is the core training function. It takes a DataFrame where each row
    represents a decision point and each column represents a discrete feature.
    It then calculates the conditional probability of each outcome (`O`, `LB`,
    `SB`) given each feature value. These probabilities are converted to
    log-odds to create the final weights model.

    Args:
        df: The input DataFrame containing the featurized training data.
        cfg: The main `Config` object (currently unused).
        alpha: A smoothing factor (Laplace smoothing) to prevent zero probabilities.
        sample_weights: An optional pandas Series to assign different weights to
                        specific training examples, used for iterative reweighting.

    Returns:
        A nested dictionary representing the statistical model, where keys are
        feature groups, feature values, and outcomes, and the final values are
        the log-odds weights.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
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
        "lemma": ["lemma_bigram"],
        "tag": ["tag_bigram"],
        "morphology": ["morph_bigram"],
        "dependency": ["dep_bigram", "head_position", "head_link"],
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
                key_name = feature_value
                if group == "structural_heuristics":
                    key_name = f"{feature}:{feature_value}"
                weights[group][key_name] = {
                    outcome: log_odds(row.get(outcome, 0.0)) for outcome in outcomes
                }
                
    return weights