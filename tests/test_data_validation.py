from isce.data_validation import iter_blocks, validate
from isce.config import Config
from isce.types import Token


def _cfg(min_block_duration: float = 2.0) -> Config:
    return Config(
        beam_width=5,
        min_block_duration_s=min_block_duration,
        max_block_duration_s=6.0,
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


def test_iter_blocks_yields_block_boundaries():
    tokens = [
        Token("one", 0.0, 0.4, None, break_type="O"),
        Token("two", 0.4, 0.8, None, break_type="SB"),
        Token("three", 1.0, 1.5, None, break_type="LB"),
        Token("four", 1.5, 2.0, None, break_type="SB"),
    ]

    assert list(iter_blocks(tokens)) == [(0, 1), (2, 3)]


def test_validate_flags_temporal_and_structural_issues():
    tokens = [
        Token("bad", 0.5, 0.3, None, break_type="O"),
        Token("pause", 0.3, 0.6, None, pause_after_ms=-10, break_type="LB"),
        Token("line", 0.6, 0.8, None, break_type="LB"),
        Token("end", 0.8, 1.0, None, break_type="SB"),
    ]

    report = validate(tokens, _cfg())

    issue_types = {issue["type"] for issue in report["issues"]}

    assert report["issue_count"] == 4
    assert {
        "time_order_error",
        "negative_pause_error",
        "short_block_warning",
        "too_many_line_breaks_error",
    } == issue_types
