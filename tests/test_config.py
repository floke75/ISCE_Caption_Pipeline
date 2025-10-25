import json
from pathlib import Path

import pytest

from isce.config import load_config


def test_load_config_merges_learned_constraints(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    constraints_path = tmp_path / "constraints.json"

    config_path.write_text(
        """
beam_width: 5
constraints:
  min_block_duration_s: 1.2
  max_block_duration_s: 6.0
  line_length_soft_target: 35
  line_length_hard_limit: 40
min_chars_for_single_word_block: 9
sliders:
  flow: 1.1
paths:
  constraints: constraints.json
""".strip(),
        encoding="utf-8",
    )

    constraints_path.write_text(
        json.dumps(
            {
                "min_block_duration_s": 2.0,
                "max_block_duration_s": 7.0,
                "ideal_cps_iqr": [8.0, 16.0],
                "line1": {"soft_target": 30, "hard_limit": 36},
                "line2": {"soft_target": 32, "hard_limit": 38},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(config_path))

    assert cfg.beam_width == 5
    assert cfg.min_block_duration_s == 2.0
    assert cfg.max_block_duration_s == 7.0
    assert cfg.line_length_constraints["line1"]["soft_target"] == 30
    assert cfg.sliders["flow"] == 1.1
    assert cfg.paths["constraints"] == "constraints.json"


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))


def test_load_config_rejects_non_dict_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not a mapping", encoding="utf-8")

    with pytest.raises(TypeError):
        load_config(str(config_path))
