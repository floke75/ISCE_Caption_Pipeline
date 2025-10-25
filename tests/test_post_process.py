import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.config import Config
from isce.post_process import reflow_tokens
from isce.types import Token


class MergeFriendlyScorer:
    def score_block(self, block_tokens, block_breaks):
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
    def score_block(self, block_tokens, block_breaks):
        if len(block_tokens) != 3:
            return 0.0
        if block_breaks[0] == "LB":
            return -5.0
        if block_breaks[1] == "LB":
            return 8.0
        return 0.0


def _make_config() -> Config:
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
    def test_merges_short_block_with_following_block(self):
        tokens = [
            Token(w="Hi", start=0.0, end=0.4, speaker="A", break_type="SB"),
            Token(w="there", start=0.4, end=0.8, speaker="A", break_type="O"),
            Token(w="friend", start=0.8, end=1.2, speaker="A", break_type="SB"),
        ]

        cfg = _make_config()
        merged = reflow_tokens(tokens, MergeFriendlyScorer(), cfg)
        breaks = [token.break_type for token in merged]

        self.assertEqual(breaks, ["O", "O", "SB"])

    def test_shifts_line_break_to_balance_lines(self):
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
