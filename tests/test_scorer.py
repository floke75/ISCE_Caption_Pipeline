import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isce.config import Config
from isce.scorer import Scorer


def _base_config() -> Config:
    return Config(
        beam_width=1,
        min_block_duration_s=0.1,
        max_block_duration_s=10.0,
        line_length_constraints={
            "line1": {
                "soft_target": 37,
                "hard_limit": 42,
                "soft_min": 16,
                "soft_over_penalty_scale": 0.1,
                "soft_under_penalty_scale": 0.05,
            },
            "line2": {
                "soft_target": 37,
                "hard_limit": 42,
                "soft_min": 16,
                "soft_over_penalty_scale": 0.1,
                "soft_under_penalty_scale": 0.05,
            },
            "block": {"min_total_chars": 12, "min_last_line_chars": 7},
        },
        min_chars_for_single_word_block=4,
        sliders={
            "density": 0.0,
            "balance": 0.0,
            "short_block_penalty": 2.0,
            "short_line_penalty": 3.0,
        },
        paths={},
    )


def _make_scorer(cfg: Config) -> Scorer:
    constraints = {
        "ideal_cps_iqr": [0.0, 100.0],
        "ideal_cps_median": 10.0,
        "min_block_duration_s": 0.05,
        "max_block_duration_s": 20.0,
    }
    return Scorer(weights={}, constraints=constraints, sliders=cfg.sliders, cfg=cfg)


def test_short_block_penalty_applies_for_underfilled_block():
    cfg = _base_config()
    scorer = _make_scorer(cfg)

    block_tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.4, "pause_after_ms": 0, "is_sentence_final": False},
        {"w": "there", "start": 0.4, "end": 0.8, "pause_after_ms": 0, "is_sentence_final": False},
    ]
    block_breaks = ["O", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score == pytest.approx(-8.0)


def test_sentence_final_block_skips_short_penalty():
    cfg = _base_config()
    scorer = _make_scorer(cfg)

    block_tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.4, "pause_after_ms": 0, "is_sentence_final": False},
        {"w": "there", "start": 0.4, "end": 0.8, "pause_after_ms": 0, "is_sentence_final": True},
    ]
    block_breaks = ["O", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score == pytest.approx(0.0)
