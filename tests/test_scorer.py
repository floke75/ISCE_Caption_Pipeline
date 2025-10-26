import pytest

from isce.scorer import Scorer
from isce.config import Config
from isce.types import TokenRow, TransitionContext

def approx_equal(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _make_cfg(**overrides) -> Config:
    cfg = Config(
        beam_width=5,
        min_block_duration_s=1.0,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {
                "soft_target": 37,
                "hard_limit": 42,
                "soft_min": 0,
                "soft_over_penalty_scale": 0.1,
                "soft_under_penalty_scale": 0.05,
            },
            "line2": {
                "soft_target": 37,
                "hard_limit": 42,
                "soft_min": 0,
                "soft_over_penalty_scale": 0.1,
                "soft_under_penalty_scale": 0.05,
            },
            "block": {"min_total_chars": 0, "min_last_line_chars": 0},
        },
        min_chars_for_single_word_block=10,
        sliders={},
        paths={},
        enable_bidirectional_pass=False,
        lookahead_width=0,
        enable_reflow=False,
        min_line_length_for_break=1,
        min_last_word_len_for_break=1,
        single_word_line_penalty=0.0,
        extreme_balance_penalty=0.0,
        enable_refinement_pass=False,
        min_block_length_char=1,
        min_line_length_char=1,
        line_length_soft_min=0,
        line_length_overflow_scale=0.1,
        line_length_underflow_scale=0.05,
        min_total_chars_per_block=0,
        min_last_line_chars=0,
        short_block_penalty=0.0,
        short_line_penalty=0.0,
        extreme_balance_threshold=3.0,
        allowed_single_word_proper_nouns=set(),
    )

    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


def _make_scorer(cfg: Config | None = None, sliders: dict | None = None) -> Scorer:
    cfg = cfg or _make_cfg()
    base_constraints = {
        "ideal_cps_iqr": [8.0, 16.0],
        "ideal_cps_median": 12.0,
        "ideal_balance_iqr": [0.8, 1.2],
        "min_block_duration_s": 0.5,
        "max_block_duration_s": 5.0,
    }
    effective_sliders = dict(cfg.sliders)
    if sliders:
        effective_sliders.update(sliders)
    return Scorer({}, base_constraints, effective_sliders, cfg)


def test_score_transition_applies_weights_and_structure_boost():
    weights = {
        "prosody": {"pz:0-0.5": {"O": 0.5, "LB": 0.2, "SB": -0.1}},
        "punctuation": {"p:comma": {"O": 0.1, "LB": 0.05, "SB": 0.3}},
        "position": {"rp:mid": {"O": 0.2, "LB": 0.1, "SB": 0.0}},
        "syntax": {"pb:NOUN|VERB": {"O": 0.3, "LB": 0.4, "SB": 0.2}},
        "capitalization": {"cap:split": {"O": -0.2, "LB": 0.1, "SB": 0.5}},
        "cohesion": {"True": {"O": -0.1, "LB": 0.0, "SB": 0.2}},
        "structural_heuristics": {
            "is_dangling_eos:True": {"O": -0.3, "LB": 0.0, "SB": 0.4},
            "starts_with_dash:False": {"O": 0.0, "LB": 0.0, "SB": 0.1},
        },
        "speaker_change_feature": {"True": {"O": 0.0, "LB": 0.0, "SB": 0.2}},
        "interaction_punct_pause": {
            "pp:p:comma_pz:0-0.5": {"O": 0.05, "LB": 0.0, "SB": 0.1}
        },
        "interaction_punct_syntax": {
            "ps:p:comma_pb:NOUN|VERB": {"O": -0.1, "LB": 0.0, "SB": 0.2}
        },
    }

    constraints = {
        "ideal_cps_iqr": [8.0, 16.0],
        "ideal_cps_median": 12.0,
        "ideal_balance_iqr": [0.8, 1.2],
        "min_block_duration_s": 1.0,
        "max_block_duration_s": 6.0,
    }

    scorer = Scorer(
        weights,
        constraints,
        {"flow": 2.0, "structure": 1.5, "structure_boost": 4.0},
        _make_cfg(),
    )

    token = {
        "w": "Hello,",
        "pause_z": 0.2,
        "relative_position": 0.3,
        "pos": "NOUN",
        "num_unit_glue": True,
        "is_dangling_eos": True,
        "speaker_change": True,
        "starts_with_dialogue_dash": False,
        "is_llm_structural_break": True,
        "is_sentence_initial": False,
    }
    nxt = {"w": "World", "pos": "VERB", "is_sentence_initial": False}

    scores = scorer.score_transition(TokenRow(token=token, nxt=nxt))

    assert approx_equal(scores["O"], -7.1)
    assert approx_equal(scores["LB"], 1.7)
    assert approx_equal(scores["SB"], 12.05)


def test_score_block_balances_density_and_duration():
    cfg = _make_cfg()
    scorer = _make_scorer(cfg)

    block_tokens = [
        {"w": "Hello", "start": 0.0, "end": 0.5, "pause_after_ms": 100},
        {"w": "world", "start": 0.5, "end": 1.0, "pause_after_ms": 200},
        {"w": "!", "start": 1.1, "end": 1.3, "pause_after_ms": 0},
    ]
    block_breaks = ["O", "LB", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert approx_equal(score, 0.5)


def test_single_word_penalty_respects_whitelist():
    cfg = _make_cfg(
        single_word_line_penalty=5.0,
        allowed_single_word_proper_nouns={"nasa"},
    )
    baseline = _make_scorer(cfg).score_block(
        [{"w": "NASA", "start": 0.0, "end": 0.5, "pos": "PROPN"}],
        ["SB"],
    )

    score = _make_scorer(cfg).score_block(
        [{"w": "NASA", "start": 0.0, "end": 0.5, "pos": "PROPN"}],
        ["SB"],
    )

    assert approx_equal(score, baseline)

    penalized = _make_scorer(cfg).score_block(
        [{"w": "Hi", "start": 0.0, "end": 0.5, "pos": "INTJ"}],
        ["SB"],
    )

    assert penalized < baseline


def test_short_block_penalty_applies_for_nonfinal_block():
    cfg = _make_cfg(short_block_penalty=2.0, min_total_chars_per_block=12)
    scorer = _make_scorer(cfg, {"short_block_penalty": 2.0})

    nonfinal_tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.3, "pause_after_ms": 0, "is_sentence_final": False},
        {"w": "there", "start": 0.3, "end": 0.6, "pause_after_ms": 0, "is_sentence_final": False},
    ]
    block_breaks = ["O", "SB"]

    nonfinal_score = scorer.score_block(nonfinal_tokens, block_breaks)

    sentence_final_tokens = [
        {"w": "Hi", "start": 0.0, "end": 0.3, "pause_after_ms": 0, "is_sentence_final": False},
        {"w": "there", "start": 0.3, "end": 0.6, "pause_after_ms": 0, "is_sentence_final": True},
    ]

    final_score = scorer.score_block(sentence_final_tokens, block_breaks)

    assert nonfinal_score < final_score


def test_transition_context_penalizes_projected_fragments():
    cfg = _make_cfg()
    sliders = {"fragment_penalty": 6.0, "fragment_char_threshold": 8.0}
    scorer = _make_scorer(cfg, sliders)

    row = TokenRow(
        token={"w": "Hello", "pause_z": 0.0, "relative_position": 0.5, "pos": "INTJ"},
        nxt={"w": "world", "pos": "NOUN", "is_sentence_initial": False},
    )

    baseline_lb = scorer.score_transition(row)["LB"]
    context = TransitionContext(
        pending_tokens=(),
        current_line_num=1,
        current_line_len=5,
        projected_second_line_chars=4,
        projected_second_line_words=1,
    )
    penalized_lb = scorer.score_transition(row, context)["LB"]

    assert penalized_lb < baseline_lb


def test_short_last_line_penalty_triggers():
    cfg = _make_cfg(short_line_penalty=1.5, min_last_line_chars=6)
    scorer = _make_scorer(cfg, {"short_line_penalty": 1.5})

    block_tokens = [
        {"w": "This", "start": 0.0, "end": 0.3, "pause_after_ms": 0},
        {"w": "is", "start": 0.3, "end": 0.5, "pause_after_ms": 0},
        {"w": "fine", "start": 0.5, "end": 0.8, "pause_after_ms": 0},
    ]
    block_breaks = ["O", "LB", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert score < 0.0
