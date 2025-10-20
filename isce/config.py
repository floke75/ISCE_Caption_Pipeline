# C:\dev\Captions_Formatter\Formatter_machine\isce\config.py

from __future__ import annotations
from dataclasses import dataclass
import yaml
import json
from pathlib import Path

@dataclass
class Config:
    """A typed configuration object for the captioning engine."""
    beam_width: int
    min_block_duration_s: float
    max_block_duration_s: float
    line_length_constraints: dict[str, dict[str, int]]
    min_chars_for_single_word_block: int
    sliders: dict[str, float]
    paths: dict[str, str]

def load_config(path: str = "config.yaml") -> Config:
    """Load the YAML configuration and merge it with any learned constraint files.

    The loader keeps the inline defaults from `config.yaml`, but, when the
    referenced `constraints` JSON is present on disk, its dynamic limits take
    precedence. File paths are resolved relative to the configuration file so a
    copied config continues to work from arbitrary locations.
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

    return Config(
      beam_width=int(y.get("beam_width", 7)),
      min_block_duration_s=float(constraints_json.get("min_block_duration_s", constraints_yaml.get("min_block_duration_s", 1.0))),
      max_block_duration_s=float(constraints_json.get("max_block_duration_s", constraints_yaml.get("max_block_duration_s", 8.0))),
      line_length_constraints={
          "line1": constraints_json.get("line1", {"soft_target": line1_soft, "hard_limit": line1_hard}),
          "line2": constraints_json.get("line2", {"soft_target": line1_soft, "hard_limit": line1_hard})
      },
      min_chars_for_single_word_block=int(constraints_yaml.get("min_chars_for_single_word_block", 10)),
      sliders=dict(y.get("sliders", {})),
      paths=dict(y.get("paths", {})),
    )