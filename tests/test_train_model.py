import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isce.config import Config
from isce.model_builder import derive_constraints
from scripts.train_model import partition_corpus_paths


def _write_tokens(path: Path, tokens: list[dict]) -> None:
    path.write_text(json.dumps({"tokens": tokens}), encoding="utf-8")


def _fallback_config() -> Config:
    return Config(
        beam_width=7,
        min_block_duration_s=0.5,
        max_block_duration_s=10.0,
        line_length_constraints={"line1": {"soft_target": 37, "hard_limit": 42}, "line2": {"soft_target": 37, "hard_limit": 42}},
        min_chars_for_single_word_block=4,
        sliders={},
        paths={},
        lookahead_width=0,
        allowed_single_word_proper_nouns=(),
    )


def test_partition_corpus_paths_identifies_raw(tmp_path: Path) -> None:
    human_file = tmp_path / "clip.train.words.json"
    raw_file = tmp_path / "clip.train.raw.words.json"
    other_file = tmp_path / "notes.json"

    for path in (human_file, raw_file, other_file):
        path.write_text("{}", encoding="utf-8")

    human_paths, raw_paths = partition_corpus_paths(tmp_path)

    assert human_file in human_paths
    assert raw_file in raw_paths
    assert other_file not in human_paths + raw_paths


def test_constraints_ignore_raw_duplicates(tmp_path: Path) -> None:
    human_file = tmp_path / "episode.train.words.json"
    raw_file = tmp_path / "episode.train.raw.words.json"

    _write_tokens(
        human_file,
        [
            {"w": "Hello", "start": 0.0, "end": 0.5, "break_type": "O", "pause_after_ms": 0},
            {"w": "world", "start": 0.5, "end": 1.0, "break_type": "SB", "pause_after_ms": 0},
        ],
    )

    _write_tokens(
        raw_file,
        [
            {"w": "hello", "start": 0.0, "end": 0.2, "break_type": "O", "pause_after_ms": 0},
            {"w": "world", "start": 0.2, "end": 0.4, "break_type": "SB", "pause_after_ms": 0},
        ],
    )

    cfg = _fallback_config()

    expected = derive_constraints([str(human_file)], cfg)

    human_paths, raw_paths = partition_corpus_paths(tmp_path)
    assert raw_paths, "Raw duplicate should be detected for regression coverage."

    filtered_constraints = derive_constraints([str(p) for p in human_paths], cfg)
    assert filtered_constraints["ideal_cps_median"] == pytest.approx(expected["ideal_cps_median"])
    assert filtered_constraints["min_block_duration_s"] == pytest.approx(expected["min_block_duration_s"])

    polluted_constraints = derive_constraints([str(p) for p in human_paths + raw_paths], cfg)
    assert polluted_constraints["ideal_cps_median"] != pytest.approx(expected["ideal_cps_median"])
