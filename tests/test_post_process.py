from __future__ import annotations
from dataclasses import replace
import pytest
from isce.post_process import reflow_tokens
from isce.scorer import Scorer
from isce.types import Token, BreakType
from isce.config import Config

@pytest.fixture
def mock_config() -> Config:
    """A mock config for testing."""
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

@pytest.fixture
def mock_scorer(mock_config: Config) -> Scorer:
    """A mock scorer that provides deterministic scores for testing."""
    return Scorer(weights={}, constraints={}, sliders={}, cfg=mock_config)


@pytest.fixture
def mock_tokens() -> List[Token]:
    """A default list of mock tokens for testing."""
    tokens = [
        Token(w="This", start=0.0, end=0.2, speaker="S1", break_type="O"),
        Token(w="is", start=0.2, end=0.4, speaker="S1", break_type="O"),
        Token(w="a", start=0.4, end=0.6, speaker="S1", break_type="LB"),
        Token(w="test", start=0.6, end=0.8, speaker="S1", break_type="SB"),
        Token(w=".", start=0.8, end=1.0, speaker="S1", break_type="SB"),
        Token(w="Another", start=1.0, end=1.2, speaker="S2", break_type="O"),
        Token(w="one", start=1.2, end=1.4, speaker="S2", break_type="SB"),
    ]
    return tokens


def test_reflow_tokens_merge_short_blocks(mock_scorer: Scorer, mock_tokens: List[Token]):
    """
    Test that reflow_tokens correctly merges short blocks.
    """
    # The "." token is a short block that should be merged with the previous block.
    # The "one" token is a short block but is spoken by a different speaker so it should not be merged.
    expected_breaks: List[BreakType] = ["O", "O", "LB", "O", "SB", "O", "SB"]

    reflowed_tokens = reflow_tokens(mock_tokens, mock_scorer)
    final_breaks = [token.break_type for token in reflowed_tokens]

    assert final_breaks == expected_breaks


def test_reflow_tokens_rebalance_line_breaks(mock_scorer: Scorer):
    """
    Test that reflow_tokens correctly rebalances line breaks.
    """
    tokens = [
        Token(w="This", start=0.0, end=0.2, speaker="S1", break_type="LB"),
        Token(w="is", start=0.2, end=0.4, speaker="S1", break_type="O"),
        Token(w="a", start=0.4, end=0.6, speaker="S1", break_type="O"),
        Token(w="very", start=0.6, end=0.8, speaker="S1", break_type="O"),
        Token(w="long", start=0.8, end=1.0, speaker="S1", break_type="O"),
        Token(w="line", start=1.0, end=1.2, speaker="S1", break_type="SB"),
    ]
    # The initial line break is at a suboptimal position.
    # The reflow function should move the line break for better balance.
    expected_breaks: List[BreakType] = ["O", "O", "LB", "O", "O", "SB"]

    reflowed_tokens = reflow_tokens(tokens, mock_scorer)
    final_breaks = [token.break_type for token in reflowed_tokens]

    assert final_breaks == expected_breaks
