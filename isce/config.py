# C:\dev\Captions_Formatter\Formatter_machine\isce\config.py
"""Manages the loading and validation of application configuration.

This module defines the `Config` dataclass, which serves as a centralized,
type-safe container for all pipeline settings. It also provides the
`load_config` function, which is responsible for reading settings from the main
`config.yaml` file and intelligently merging them with learned constraints from
a `constraints.json` file generated during model training.
"""
from __future__ import annotations
from dataclasses import dataclass
import yaml
import json
from pathlib import Path

@dataclass
class Config:
    """
    A typed configuration object that holds all settings for the captioning engine.

    This dataclass centralizes configuration parameters, ensuring that different
    parts of the application access settings in a consistent and type-safe way.
    It is typically instantiated by the `load_config` function.

    Attributes:
        beam_width: The number of hypotheses to keep at each step of the beam search.
        min_block_duration_s: The minimum duration a subtitle block can have, in seconds.
        max_block_duration_s: The maximum duration a subtitle block can have, in seconds.
        line_length_constraints: A nested dictionary defining the soft and hard character
                                 limits for each line of a subtitle block, including
                                 soft minimum targets and penalty scales.
        min_chars_for_single_word_block: The minimum character length required for a
                                         block that contains only a single word.
        sliders: A dictionary of user-adjustable floating-point values that tune the
                 behavior of the scoring model.
        paths: A dictionary containing the relative paths to model files like weights
               and constraints.
        enable_bidirectional_pass: If True, runs a second beam search in reverse.
        lookahead_width: The number of future tokens to consider for heuristics.
        enable_reflow: If True, enables a post-processing pass to reflow tokens.
        min_line_length_for_break: Minimum characters for a line to be broken.
        min_last_word_len_for_break: Minimum length of the last word for a line break.
        single_word_line_penalty: Penalty for a line with a single word.
        extreme_balance_penalty: Penalty for extremely unbalanced lines.
        enable_refinement_pass: If True, enables a refinement pass for low-quality cues.
        min_block_length_char: Minimum character length for a block (legacy fallback when
                               short block penalties are disabled).
        min_line_length_char: Minimum character length for a line (legacy fallback when
                              short line penalties are disabled).
        line_length_soft_min: Preferred minimum character count for each line before
                              underflow penalties are applied.
        line_length_overflow_scale: Quadratic penalty scale applied when a line exceeds
                                    the soft target.
        line_length_underflow_scale: Quadratic penalty scale applied when a line falls
                                     short of the soft minimum.
        min_total_chars_per_block: Preferred minimum character count for a block when
                                   applying the short block penalty slider.
        min_last_line_chars: Preferred minimum character count for the final line.
        short_block_penalty: Penalty multiplier for under-filled blocks.
        short_line_penalty: Penalty multiplier for under-filled final lines.
        extreme_balance_threshold: Ratio threshold where the extreme balance penalty
                                   begins to escalate.
        allowed_single_word_proper_nouns: Lower-cased set of proper nouns that are
                                          exempt from single-word penalties.
    """
    beam_width: int
    min_block_duration_s: float
    max_block_duration_s: float
    line_length_constraints: dict[str, dict[str, int]]
    min_chars_for_single_word_block: int
    sliders: dict[str, float]
    paths: dict[str, str]
    enable_bidirectional_pass: bool
    lookahead_width: int
    enable_reflow: bool
    min_line_length_for_break: int
    min_last_word_len_for_break: int
    single_word_line_penalty: float
    extreme_balance_penalty: float
    enable_refinement_pass: bool
    min_block_length_char: int
    min_line_length_char: int
    line_length_soft_min: int
    line_length_overflow_scale: float
    line_length_underflow_scale: float
    min_total_chars_per_block: int
    min_last_line_chars: int
    short_block_penalty: float
    short_line_penalty: float
    extreme_balance_threshold: float
    allowed_single_word_proper_nouns: set[str]

def load_config(path: str = "config.yaml") -> Config:
    """
    Loads, merges, and validates configuration files into a single Config object.

    This function is the primary entry point for loading all application settings.
    It reads the base settings from the user--editable `config.yaml` file. It then
    intelligently loads the `constraints.json` file (which is generated during
    model training) and merges its values, prioritizing the learned constraints
    over the fallback values in the YAML file.

    Args:
        path: The path to the main `config.yaml` file.

    Returns:
        A fully populated and validated `Config` object.

    Raises:
        FileNotFoundError: If the specified `config.yaml` file cannot be found.
        ValueError: If there is an error parsing the YAML file.
        TypeError: If the root of the YAML file is not a dictionary.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file at {path}: {e}")

    if not isinstance(y, dict):
        raise TypeError(f"Configuration file {path} must be a dictionary.")

    # Default fallback values from the main config.yaml
    constraints_yaml = y.get("constraints", {})
    line1_soft = int(constraints_yaml.get("line_length_soft_target", 37))
    line1_hard = int(constraints_yaml.get("line_length_hard_limit", 42))
    soft_min = int(constraints_yaml.get("line_length_soft_min", 0))
    over_scale = float(constraints_yaml.get("line_length_overflow_scale", 0.1))
    under_scale = float(constraints_yaml.get("line_length_underflow_scale", 0.05))
    min_total_chars_per_block = int(constraints_yaml.get("min_total_chars_per_block", 0))
    min_last_line_chars = int(constraints_yaml.get("min_last_line_chars", 0))

    sliders_yaml = y.get("sliders", {})
    
    # Attempt to load the learned constraints.json file
    constraints_json = {}
    constraints_path_str = y.get("paths", {}).get("constraints")
    if constraints_path_str:
        full_constraints_path = Path(path).parent / constraints_path_str
        if full_constraints_path.exists():
            with open(full_constraints_path, "r", encoding="utf-8") as f:
                constraints_json = json.load(f)
        else:
            print(f"Warning: Could not load constraints file from {full_constraints_path}. Using fallbacks from config.yaml.")

    line_defaults = {
        "soft_target": line1_soft,
        "hard_limit": line1_hard,
        "soft_min": soft_min,
        "soft_over_penalty_scale": over_scale,
        "soft_under_penalty_scale": under_scale,
    }
    line1_constraints = dict(line_defaults)
    line1_constraints.update(constraints_json.get("line1", {}))
    line2_constraints = dict(line_defaults)
    line2_constraints.update(constraints_json.get("line2", {}))
    block_constraints = {
        "min_total_chars": min_total_chars_per_block,
        "min_last_line_chars": min_last_line_chars,
    }
    block_constraints.update(constraints_json.get("block", {}))

    allowed_single_words = {
        str(item).strip().lower()
        for item in y.get("allowed_single_word_proper_nouns", [])
        if str(item).strip()
    }

    return Config(
        beam_width=int(y.get("beam_width", 7)),
        min_block_duration_s=float(constraints_json.get("min_block_duration_s", constraints_yaml.get("min_block_duration_s", 1.0))),
        max_block_duration_s=float(constraints_json.get("max_block_duration_s", constraints_yaml.get("max_block_duration_s", 8.0))),
        line_length_constraints={
            "line1": line1_constraints,
            "line2": line2_constraints,
            "block": block_constraints,
        },
        min_chars_for_single_word_block=int(constraints_yaml.get("min_chars_for_single_word_block", 10)),
        sliders=dict(sliders_yaml),
        paths=dict(y.get("paths", {})),
        enable_bidirectional_pass=bool(y.get("enable_bidirectional_pass", False)),
        lookahead_width=int(y.get("lookahead_width", 0)),
        enable_reflow=bool(y.get("enable_reflow", False)),
        min_line_length_for_break=int(constraints_yaml.get("min_line_length_for_break", 15)),
        min_last_word_len_for_break=int(constraints_yaml.get("min_last_word_len_for_break", 5)),
        single_word_line_penalty=float(sliders_yaml.get("single_word_line_penalty", 10.0)),
        extreme_balance_penalty=float(sliders_yaml.get("extreme_balance_penalty", 20.0)),
        enable_refinement_pass=bool(y.get("enable_refinement_pass", False)),
        min_block_length_char=int(constraints_yaml.get("min_block_length_char", 10)),
        min_line_length_char=int(constraints_yaml.get("min_line_length_char", 5)),
        line_length_soft_min=soft_min,
        line_length_overflow_scale=over_scale,
        line_length_underflow_scale=under_scale,
        min_total_chars_per_block=min_total_chars_per_block,
        min_last_line_chars=min_last_line_chars,
        short_block_penalty=float(sliders_yaml.get("short_block_penalty", 0.0)),
        short_line_penalty=float(sliders_yaml.get("short_line_penalty", 0.0)),
        extreme_balance_threshold=float(sliders_yaml.get("extreme_balance_threshold", 2.5)),
        allowed_single_word_proper_nouns=allowed_single_words,
    )