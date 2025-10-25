from isce.srt_writer import tokens_to_srt, format_time
from isce.types import Token


def test_tokens_to_srt_formats_single_and_multiline_blocks():
    tokens = [
        Token("Hello", 0.0, 0.5, None, break_type="O"),
        Token("world", 0.5, 1.0, None, break_type="SB"),
        Token("Line", 2.0, 2.3, None, break_type="LB"),
        Token("break", 2.3, 2.6, None, break_type="SB"),
    ]

    srt = tokens_to_srt(tokens)

    assert srt == (
        "1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n"
        "2\n00:00:02,000 --> 00:00:02,600\nLine\nbreak"
    )


def test_tokens_to_srt_handles_empty_input():
    assert tokens_to_srt([]) == ""


def test_format_time_clamps_and_formats():
    assert format_time(-0.5) == "00:00:00,000"
    assert format_time(3661.789) == "01:01:01,789"
