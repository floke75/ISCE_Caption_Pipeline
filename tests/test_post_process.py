"""Unit tests for the ``isce.post_process`` helpers.

The tests in this module deliberately use tiny fake scorer implementations.  By
hand-rolling the scorers we can precisely target the behavior under test (block
merging and line-break shuffling) without depending on the full statistical
model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Sequence

# Add the repository root to ``sys.path`` so the tests can import local modules
# when executed directly via ``python tests/test_post_process.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.config import Config
from isce.post_process import reflow_tokens
from isce.types import Token


class MergeFriendlyScorer:
    """Score toy blocks to prefer merging the first two cues.

    The scorer penalizes single-word blocks, mildly rewards two-word blocks, and
    strongly prefers the case where the first three tokens stay on one line.  It
    mirrors the situations the merge heuristic should catch.
    """

    def score_block(self, block_tokens: Sequence[dict], block_breaks: Sequence[str]) -> float:
        length = len(block_tokens)
        if length == 1:
            return -5.0
        if length == 2:
            return 1.0
        if length == 3:
            # Prefer keeping the bridge token inline rather than a manual line break.
            return 10.0 if block_breaks[0] == "O" else 5.0
        return 0.0


class BalanceScorer:
    """Score toy blocks to prefer shifting the line break to the right."""

    def score_block(self, block_tokens: Sequence[dict], block_breaks: Sequence[str]) -> float:
        if len(block_tokens) != 3:
            return 0.0
        if block_breaks[0] == "LB":
            return -5.0
        if block_breaks[1] == "LB":
            return 8.0
        return 0.0


def _make_config() -> Config:
    """Return a minimal :class:`~isce.config.Config` for the tests."""

    return Config(
        beam_width=1,
        min_block_duration_s=1.0,
        max_block_duration_s=10.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
        },
        min_chars_for_single_word_block=3,
        sliders={},
        paths={},
        enable_reflow=True,
    )


class TestReflowTokens(unittest.TestCase):
    """Regression tests for the ``reflow_tokens`` helper."""

    def test_merges_short_block_with_following_block(self) -> None:
        """Short, low-score cues should be merged into the following block."""

        tokens = [
            Token(w="Hi", start=0.0, end=0.4, speaker="A", break_type="SB"),
            Token(w="there", start=0.4, end=0.8, speaker="A", break_type="O"),
            Token(w="friend", start=0.8, end=1.2, speaker="A", break_type="SB"),
        ]

        cfg = _make_config()
        merged = reflow_tokens(tokens, MergeFriendlyScorer(), cfg)
        breaks = [token.break_type for token in merged]

        self.assertEqual(breaks, ["O", "O", "SB"])

    def test_shifts_line_break_to_balance_lines(self) -> None:
        """A lopsided block should shift the manual line break rightward."""

        tokens = [
            Token(w="Tiny", start=0.0, end=0.4, speaker="A", break_type="LB"),
            Token(w="sentence", start=0.4, end=0.9, speaker="A", break_type="O"),
            Token(w="here", start=0.9, end=1.4, speaker="A", break_type="SB"),
        ]

        cfg = _make_config()
        balanced = reflow_tokens(tokens, BalanceScorer(), cfg)
        breaks = [token.break_type for token in balanced]

        self.assertEqual(breaks, ["O", "LB", "SB"])


if __name__ == "__main__":
    unittest.main()
