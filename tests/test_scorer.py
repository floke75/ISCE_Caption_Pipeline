import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isce.config import Config
from isce.scorer import Scorer


def _base_config(**overrides) -> Config:
    defaults = dict(
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
        min_chars_for_single_word_block=10,
        sliders={
            "density": 0.0,
            "balance": 0.0,
            "short_block_penalty": 2.0,
            "short_line_penalty": 3.0,
        },
        paths={},
        allowed_single_word_proper_nouns=(),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _base_constraints() -> dict:
    return {
        "ideal_cps_iqr": [0.0, 100.0],
        "ideal_cps_median": 10.0,
        "ideal_balance_iqr": [0.7, 1.4],
        "min_block_duration_s": 0.05,
        "max_block_duration_s": 20.0,
    }


def _make_scorer(cfg: Config, sliders: dict | None = None) -> Scorer:
    effective_sliders = dict(cfg.sliders)
    if sliders:
        effective_sliders.update(sliders)
    return Scorer(weights={}, constraints=_base_constraints(), sliders=effective_sliders, cfg=cfg)


def test_single_word_penalty_applies():
    cfg = _base_config()
    scorer = _make_scorer(
        cfg,
        {
            "single_word_line_penalty": 5.0,
            "short_block_penalty": 0.0,
            "short_line_penalty": 0.0,
        },
    )

    block_tokens = [{"w": "Hello", "start": 0.0, "end": 0.5, "pos": "NOUN"}]
    block_breaks = ["SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score == pytest.approx(-5.0, rel=1e-3)


def test_single_word_penalty_ignored_for_whitelist():
    cfg = _base_config(allowed_single_word_proper_nouns=("NASA",))
    baseline = _make_scorer(cfg, {"short_block_penalty": 0.0, "short_line_penalty": 0.0}).score_block(
        [{"w": "NASA", "start": 0.0, "end": 0.5, "pos": "PROPN"}],
        ["SB"],
    )

    scorer = _make_scorer(
        cfg,
        {
            "single_word_line_penalty": 5.0,
            "short_block_penalty": 0.0,
            "short_line_penalty": 0.0,
        },
    )
    score = scorer.score_block(
        [{"w": "NASA", "start": 0.0, "end": 0.5, "pos": "PROPN"}],
        ["SB"],
    )

    assert score == pytest.approx(baseline)


def test_extreme_balance_penalty_applies():
    cfg = _base_config()
    base_sliders = {
        "extreme_balance_penalty": 0.0,
        "extreme_balance_threshold": 1.5,
        "single_word_line_penalty": 0.0,
    }

    balanced_block = [
        {"w": "Hello", "start": 0.0, "end": 0.4, "pos": "NOUN"},
        {"w": "world", "start": 0.4, "end": 0.8, "pos": "NOUN"},
        {"w": "again", "start": 0.8, "end": 1.2, "pos": "NOUN"},
        {"w": "today", "start": 1.2, "end": 1.6, "pos": "NOUN"},
    ]
    block_breaks = ["O", "LB", "O", "SB"]

    base_score = _make_scorer(
        cfg,
        dict(base_sliders, short_block_penalty=0.0, short_line_penalty=0.0),
    ).score_block(balanced_block, block_breaks)

    skewed_block = balanced_block + [{"w": "friends", "start": 1.6, "end": 2.0, "pos": "NOUN"}]
    skewed_breaks = ["O", "LB", "O", "O", "SB"]

    penalized_score = _make_scorer(
        cfg,
        dict(base_sliders, extreme_balance_penalty=3.0, short_block_penalty=0.0, short_line_penalty=0.0),
    ).score_block(
        skewed_block, skewed_breaks
    )

    assert penalized_score < base_score


def test_short_block_penalty_applies_for_underfilled_block():
    cfg = _base_config()
    scorer = _make_scorer(cfg, {"short_line_penalty": 0.0})

    block_tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.4, "pause_after_ms": 0},
        {"w": "there", "start": 0.4, "end": 0.8, "pause_after_ms": 0},
    ]
    block_breaks = ["O", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score == pytest.approx(-8.0)


def test_sentence_final_block_skips_short_penalty():
    cfg = _base_config()
    scorer = _make_scorer(cfg, {"short_line_penalty": 0.0})

    block_tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.4, "pause_after_ms": 0, "is_sentence_final": False},
        {"w": "there", "start": 0.4, "end": 0.8, "pause_after_ms": 0, "is_sentence_final": True},
    ]
    block_breaks = ["O", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score == pytest.approx(0.0)


def test_short_last_line_penalty_triggers():
    cfg = _base_config()
    scorer = _make_scorer(cfg)

    block_tokens = [
        {"w": "Hello", "start": 0.0, "end": 0.4},
        {"w": "world", "start": 0.4, "end": 0.8},
        {"w": "!", "start": 0.8, "end": 0.9},
    ]
    block_breaks = ["O", "LB", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score < 0.0
