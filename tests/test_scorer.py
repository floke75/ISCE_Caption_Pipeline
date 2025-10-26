import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.config import Config
from isce.scorer import Scorer
from isce.types import TokenRow, TransitionContext


def _make_config(**overrides) -> Config:
    defaults = dict(
        beam_width=7,
        min_block_duration_s=0.5,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42, "soft_min": 21},
            "line2": {"soft_target": 37, "hard_limit": 42, "soft_min": 21},
            "block": {"min_total_chars": 0, "min_last_line_chars": 0},
        },
        min_chars_for_single_word_block=10,
        sliders={},
        paths={},
        allowed_single_word_proper_nouns=(),
        min_block_length_char=0,
        min_line_length_char=0,
        min_line_length_for_break=0,
        min_last_word_len_for_break=0,
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


def test_block_penalty_for_short_total_chars() -> None:
    constraints = _make_constraints()
    tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.3, "pos": "INTJ"},
        {"w": "there", "start": 0.3, "end": 0.8, "pos": "PRON"},
    ]
    block_breaks = ["LB", "SB"]

    base_cfg = _make_config()
    tuned_cfg = _make_config(min_block_length_char=12)

    baseline = Scorer({}, constraints, {}, base_cfg).score_block(tokens, block_breaks)
    penalised = Scorer({}, constraints, {}, tuned_cfg).score_block(tokens, block_breaks)

    assert penalised < baseline

    def count_chars(token_slice: list[dict]) -> int:
        if not token_slice:
            return 0
        return sum(len(t.get("w", "")) for t in token_slice) + (len(token_slice) - 1)

    lines: list[list[dict]] = []
    current: list[dict] = []
    for idx, token in enumerate(tokens):
        current.append(token)
        if block_breaks[idx] in {"LB", "SB"}:
            lines.append(current)
            current = []
    if current:
        lines.append(current)

    total_chars = sum(count_chars(line) for line in lines)
    expected_penalty = 0.5 * (12 - total_chars)
    assert pytest.approx(baseline - penalised, rel=1e-6) == expected_penalty


def test_short_block_penalty_applies_for_underfilled_block() -> None:
    constraints = _make_constraints()
    cfg = _make_config(
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42, "soft_min": 21},
            "line2": {"soft_target": 37, "hard_limit": 42, "soft_min": 21},
            "block": {"min_total_chars": 18, "min_last_line_chars": 0},
        }
    )

    tokens = [
        {"w": "I", "start": 0.0, "end": 0.2, "pos": "PRON"},
        {"w": "agree", "start": 0.2, "end": 0.8, "pos": "VERB", "is_sentence_final": False},
    ]
    block_breaks = ["O", "SB"]

    baseline = Scorer({}, constraints, {"short_block_penalty": 0.0}, cfg).score_block(tokens, block_breaks)
    penalised = Scorer({}, constraints, {"short_block_penalty": 2.0}, cfg).score_block(tokens, block_breaks)

    assert penalised < baseline

    total_chars = len("I") + len("agree") + 1  # include space between words
    expected_penalty = 2.0 * (18 - total_chars)
    assert pytest.approx(baseline - penalised, rel=1e-6) == expected_penalty


def test_short_line_penalty_targets_last_line() -> None:
    constraints = _make_constraints()
    cfg = _make_config(
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42, "soft_min": 21},
            "line2": {"soft_target": 37, "hard_limit": 42, "soft_min": 21},
            "block": {"min_total_chars": 0, "min_last_line_chars": 12},
        }
    )

    tokens = [
        {"w": "Absolutely", "start": 0.0, "end": 0.6, "pos": "ADV"},
        {"w": "not.", "start": 0.6, "end": 1.0, "pos": "PART", "is_sentence_final": True},
    ]
    block_breaks = ["LB", "SB"]

    baseline = Scorer({}, constraints, {"short_line_penalty": 0.0}, cfg).score_block(tokens, block_breaks)
    penalised = Scorer({}, constraints, {"short_line_penalty": 1.2}, cfg).score_block(tokens, block_breaks)

    assert penalised < baseline

    last_line_len = len("not.")
    expected_penalty = 1.2 * (12 - last_line_len)
    assert pytest.approx(baseline - penalised, rel=1e-6) == expected_penalty


def test_lookahead_penalises_short_projection_and_last_word() -> None:
    constraints = _make_constraints()
    base_cfg = _make_config()
    tuned_cfg = _make_config(
        min_line_length_for_break=10,
        min_last_word_len_for_break=4,
    )

    row = TokenRow(
        token={"w": "Hi", "pause_before_ms": 0, "pause_after_ms": 0, "pos": "INTJ"},
        nxt={"w": "there"},
        lookahead=(
            {"w": "there", "pause_before_ms": 0, "pause_after_ms": 0},
        ),
    )

    baseline = Scorer({}, constraints, {}, base_cfg).score_transition(row)
    penalised = Scorer({}, constraints, {}, tuned_cfg).score_transition(row)

    assert penalised["LB"] < baseline["LB"]


def test_transition_context_penalises_fragmented_second_line() -> None:
    constraints = _make_constraints()
    sliders = {"fragment_penalty": 8.0, "fragment_char_threshold": 6.0}
    cfg = _make_config()
    scorer = Scorer({}, constraints, sliders, cfg)

    token = {
        "w": "Hello",
        "pause_z": 0.0,
        "relative_position": 0.4,
        "pos": "INTJ",
        "is_sentence_initial": False,
    }
    nxt = {"w": "world", "pos": "NOUN", "is_sentence_initial": False}
    row = TokenRow(token=token, nxt=nxt)

    context = TransitionContext(
        pending_tokens=(dict(token),),
        current_line_num=1,
        current_line_len=len(token["w"]),
        projected_second_line_chars=3,
        projected_second_line_words=1,
    )

    baseline = scorer.score_transition(row)
    penalised = scorer.score_transition(row, context)

    threshold = sliders["fragment_char_threshold"]
    deficit = threshold - context.projected_second_line_chars
    expected_penalty = sliders["fragment_penalty"] * (deficit / threshold)

    assert pytest.approx(baseline["LB"] - penalised["LB"], rel=1e-6) == expected_penalty


def test_transition_context_ignores_non_first_line_fragments() -> None:
    constraints = _make_constraints()
    sliders = {"fragment_penalty": 7.5, "fragment_char_threshold": 5.0}
    cfg = _make_config()
    scorer = Scorer({}, constraints, sliders, cfg)

    token = {
        "w": "Fine",
        "pause_z": 0.0,
        "relative_position": 0.5,
        "pos": "ADJ",
        "is_sentence_initial": False,
    }
    nxt = {"w": "then", "pos": "ADV", "is_sentence_initial": False}
    row = TokenRow(token=token, nxt=nxt)

    context = TransitionContext(
        pending_tokens=(dict(token),),
        current_line_num=2,
        current_line_len=len(token["w"]),
        projected_second_line_chars=0,
        projected_second_line_words=0,
    )

    baseline = scorer.score_transition(row)
    penalised = scorer.score_transition(row, context)

    assert penalised["LB"] == pytest.approx(baseline["LB"])
