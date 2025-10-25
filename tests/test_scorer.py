import pytest

from isce.scorer import Scorer
from isce.config import Config
from isce.types import TokenRow

def approx_equal(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _make_cfg() -> Config:
    return Config(
        beam_width=5,
        min_block_duration_s=1.0,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
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
    )


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
    constraints = {
        "ideal_cps_iqr": [8.0, 16.0],
        "ideal_cps_median": 12.0,
        "ideal_balance_iqr": [0.8, 1.2],
        "min_block_duration_s": 0.5,
        "max_block_duration_s": 5.0,
    }

    scorer = Scorer({}, constraints, {}, _make_cfg())

    block_tokens = [
        {"w": "Hello", "start": 0.0, "end": 0.5, "pause_after_ms": 100},
        {"w": "world", "start": 0.5, "end": 1.0, "pause_after_ms": 200},
        {"w": "!", "start": 1.1, "end": 1.3, "pause_after_ms": 0},
    ]
    block_breaks = ["O", "LB", "SB"]

    score = scorer.score_block(block_tokens, block_breaks)

    assert approx_equal(score, 0.5)
