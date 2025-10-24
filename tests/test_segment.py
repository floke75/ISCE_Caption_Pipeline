import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import MagicMock
from isce.beam_search import segment
from isce.types import Token, Engineered

from isce.scorer import Scorer
from isce.config import Config

def test_segment_happy_path():
    """Test that the segment function correctly assigns break types."""
    # Arrange
    tokens = [
        Token(w="Hello", start=0.0, end=0.5, speaker="A"),
        Token(w="world", start=0.5, end=1.0, speaker="A"),
    ]
    config = Config(
        beam_width=1,
        min_block_duration_s=0.5,
        max_block_duration_s=10.0,
        line_length_constraints={"line1": {"hard_limit": 42}, "line2": {"hard_limit": 42}},
        min_chars_for_single_word_block=4,
        sliders={},
        paths={},
    )
    scorer = Scorer(weights={}, constraints={}, sliders={}, cfg=config)

    # Act
    segmented_tokens = segment(tokens, scorer, config)

    # Assert
    assert segmented_tokens[0].break_type == "O"
    assert segmented_tokens[1].break_type == "SB"
