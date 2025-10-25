import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.config import Config
from isce.scorer import Scorer


def _make_config(**overrides) -> Config:
    defaults = dict(
        beam_width=7,
        min_block_duration_s=0.5,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
        },
        min_chars_for_single_word_block=10,
        sliders={},
        paths={},
        allowed_single_word_proper_nouns=(),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_constraints() -> dict:
    return {
        "ideal_cps_iqr": [10.0, 18.0],
        "ideal_cps_median": 14.0,
        "ideal_balance_iqr": [0.7, 1.4],
        "min_block_duration_s": 0.5,
        "max_block_duration_s": 8.0,
    }


def test_single_word_penalty_applies() -> None:
    cfg = _make_config()
    scorer = Scorer(
        weights={},
        constraints=_make_constraints(),
        sliders={"single_word_line_penalty": 5.0},
        cfg=cfg,
    )

    block_tokens = [{"w": "Hello", "start": 0.0, "end": 0.5, "pos": "NOUN"}]
    block_breaks = ["SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score == pytest.approx(-4.0, rel=1e-3)


def test_single_word_penalty_ignored_for_whitelist() -> None:
    cfg = _make_config(allowed_single_word_proper_nouns=("NASA",))
    constraints = _make_constraints()

    baseline = Scorer(weights={}, constraints=constraints, sliders={}, cfg=cfg).score_block(
        [{"w": "NASA", "start": 0.0, "end": 0.5, "pos": "PROPN"}],
        ["SB"],
    )

    penalised = Scorer(
        weights={},
        constraints=constraints,
        sliders={"single_word_line_penalty": 5.0},
        cfg=cfg,
    ).score_block(
        [{"w": "NASA", "start": 0.0, "end": 0.5, "pos": "PROPN"}],
        ["SB"],
    )

    assert penalised == pytest.approx(baseline)


def test_extreme_balance_penalty_applies() -> None:
    cfg = _make_config()
    constraints = _make_constraints()

    block_tokens = [
        {"w": "Short", "start": 0.0, "end": 1.0, "pos": "ADJ"},
        {"w": "Supercalifragilisticexpialidocious", "start": 1.0, "end": 2.5, "pos": "NOUN"},
    ]
    block_breaks = ["LB", "SB"]

    base_sliders = {
        "extreme_balance_penalty": 0.0,
        "extreme_balance_threshold": 1.5,
        "single_word_line_penalty": 0.0,
    }
    penalised_sliders = dict(base_sliders, extreme_balance_penalty=3.0)

    baseline = Scorer({}, constraints, base_sliders, cfg).score_block(block_tokens, block_breaks)
    penalised = Scorer({}, constraints, penalised_sliders, cfg).score_block(block_tokens, block_breaks)

    assert penalised < baseline
    len1 = len(block_tokens[0]["w"])
    len2 = len(block_tokens[1]["w"])
    ratio = max(len1, len2) / min(len1, len2)
    expected_penalty = 3.0 * (1.0 + (ratio - 1.5) / 1.5)
    assert math.isclose(baseline - penalised, expected_penalty, rel_tol=1e-6)
