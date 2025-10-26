# C:\\dev\\Captions_Formatter\\Formatter_machine\\isce\\config.py
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
    """Typed configuration object holding all captioning settings.

    The dataclass centralises configuration parameters so different parts of the
    application consume a consistent, validated view of the settings.  It is
    typically instantiated by :func:`load_config`.

    Attributes
    ----------
    beam_width:
        Number of hypotheses to keep at each step of the beam search.
    min_block_duration_s / max_block_duration_s:
        Permitted duration range for subtitle cues, in seconds.
    line_length_constraints:
        Nested mapping describing soft and hard character limits per line.
    min_chars_for_single_word_block:
        Minimum visual length required when a block contains only a single word.
    min_block_length_char:
        Minimum number of characters (including spaces) expected for any block.
    min_line_length_char:
        Minimum character count for each line when the configuration elects to
        police multi-word lines as well as single-word captions.
    min_line_length_for_break:
        Threshold used by lookahead heuristics to discourage inserting a line
        break if the projected next line would be extremely short.
    min_last_word_len_for_break:
        Threshold used by lookahead heuristics to avoid ending a line on a very
        short final word.
    sliders:
        Dictionary of user-adjustable multipliers that tune the statistical
        model's behaviour.
    paths:
        Mapping containing the relative paths to model assets such as weights and
        constraints.
    lookahead_width:
        Number of future tokens the segmenter exposes to the scorer.  ``0``
        disables lookahead heuristics entirely.
    enable_reflow:
        When ``True`` run an additional post-processing pass that reflows awkward
        short or imbalanced cues.
    enable_bidirectional_pass:
        When ``True`` execute both forward and reverse beam searches and reconcile
        their boundaries.
    allowed_single_word_proper_nouns:
        Tuple of proper nouns that may appear as single-word captions without
        triggering hard rejections.
    enable_refinement_pass:
        Whether to run a localized follow-up search that re-scores low quality
        cues after the main beam search finishes.
    enforce_short_line_limit_for_multi_word_lines:
        Optional guardrail that, when enabled, treats multi-word lines falling
        below :attr:`min_line_length_char` as violations.
    """

    beam_width: int
    min_block_duration_s: float
    max_block_duration_s: float
    line_length_constraints: dict[str, dict[str, int]]
    min_chars_for_single_word_block: int
    sliders: dict[str, float]
    paths: dict[str, str]
    lookahead_width: int = 0
    enable_reflow: bool = False
    enable_bidirectional_pass: bool = False
    allowed_single_word_proper_nouns: tuple[str, ...] = ()
    enable_refinement_pass: bool = False
    min_line_length_for_break: int = 0
    min_last_word_len_for_break: int = 0
    min_block_length_char: int = 0
    min_line_length_char: int = 0
    enforce_short_line_limit_for_multi_word_lines: bool = False


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
    min_line_length_for_break = int(constraints_yaml.get("min_line_length_for_break", 0))
    min_last_word_len_for_break = int(constraints_yaml.get("min_last_word_len_for_break", 0))
    min_block_length_char = int(constraints_yaml.get("min_block_length_char", 0))
    min_line_length_char = int(constraints_yaml.get("min_line_length_char", 0))
    enforce_multi_word_short_line = bool(
        constraints_yaml.get("enforce_short_line_limit_for_multi_word_lines", False)
    )

    # Attempt to load the learned constraints.json file
    constraints_json = {}
    constraints_path_str = y.get("paths", {}).get("constraints")
    if constraints_path_str:
        full_constraints_path = Path(path).parent / constraints_path_str
        if full_constraints_path.exists():
            with open(full_constraints_path, "r", encoding="utf-8") as f:
                constraints_json = json.load(f)
        else:
            print(
                f"Warning: Could not load constraints file from {full_constraints_path}. "
                "Using fallbacks from config.yaml."
            )

    allowed_single_word_proper_nouns = tuple(
        str(item) for item in y.get("allowed_single_word_proper_nouns", [])
    )

    return Config(
        beam_width=int(y.get("beam_width", 7)),
        min_block_duration_s=float(
            constraints_json.get(
                "min_block_duration_s",
                constraints_yaml.get("min_block_duration_s", 1.0),
            )
        ),
        max_block_duration_s=float(
            constraints_json.get(
                "max_block_duration_s",
                constraints_yaml.get("max_block_duration_s", 8.0),
            )
        ),
        line_length_constraints={
            "line1": constraints_json.get(
                "line1", {"soft_target": line1_soft, "hard_limit": line1_hard}
            ),
            "line2": constraints_json.get(
                "line2", {"soft_target": line1_soft, "hard_limit": line1_hard}
            ),
        },
        min_chars_for_single_word_block=int(
            constraints_yaml.get("min_chars_for_single_word_block", 10)
        ),
        sliders=dict(y.get("sliders", {})),
        paths=dict(y.get("paths", {})),
        lookahead_width=int(y.get("lookahead_width", 0)),
        enable_reflow=bool(y.get("enable_reflow", False)),
        enable_bidirectional_pass=bool(y.get("enable_bidirectional_pass", False)),
        allowed_single_word_proper_nouns=allowed_single_word_proper_nouns,
        enable_refinement_pass=bool(y.get("enable_refinement_pass", False)),
        min_line_length_for_break=int(
            constraints_json.get(
                "min_line_length_for_break", min_line_length_for_break
            )
        ),
        min_last_word_len_for_break=int(
            constraints_json.get(
                "min_last_word_len_for_break", min_last_word_len_for_break
            )
        ),
        min_block_length_char=int(
            constraints_json.get("min_block_length_char", min_block_length_char)
        ),
        min_line_length_char=int(
            constraints_json.get("min_line_length_char", min_line_length_char)
        ),
        enforce_short_line_limit_for_multi_word_lines=bool(
            constraints_json.get(
                "enforce_short_line_limit_for_multi_word_lines",
                enforce_multi_word_short_line,
            )
        ),
    )
