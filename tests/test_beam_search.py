import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.beam_search import segment
from isce.config import Config
from isce.types import Token


class DummyScorer:
    def __init__(self):
        self.sl = {"line_length_leniency": 1.0, "orphan_leniency": 1.0}

    def score_transition(self, row):
        word = row.token.get("w", "")
        if word.endswith("0"):
            return {"O": -5.0, "LB": 5.0, "SB": -5.0}
        if word.endswith("2"):
            return {"O": -5.0, "LB": 10.0, "SB": -1.0}
        return {"O": -5.0, "LB": -5.0, "SB": -5.0}

    def score_block(self, block_tokens, block_breaks):
        return 0.0


def make_token(word: str, start: float) -> Token:
    return Token(w=word, start=start, end=start + 0.2, speaker="A")


class TestBeamSearch(unittest.TestCase):
    def test_fallback_candidate_keeps_beam_alive(self):
        tokens = [
            make_token("AAAAA0", 0.0),
            make_token("BBBBB1", 0.2),
            make_token("CCCCC2", 0.4),
            make_token("DDDDD3", 0.6),
        ]

        cfg = Config(
            beam_width=1,
            min_block_duration_s=10.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 5, "hard_limit": 5, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 5, "hard_limit": 5, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "SB", "LB", "SB"])

    def test_short_line_underflow_penalty_discourages_extension(self):
        tokens = [
            make_token("Hi", 0.0),
            make_token("a", 0.2),
            make_token("buddy", 0.4),
        ]

        class UnderflowScorer:
            def __init__(self):
                self.sl = {"line_length_leniency": 1.0, "orphan_leniency": 1.0}

            def score_transition(self, row):
                return {"O": 5.0, "LB": 0.0, "SB": -1.0}

            def score_block(self, block_tokens, block_breaks):
                return 0.0

        cfg = Config(
            beam_width=2,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {
                    "soft_target": 12,
                    "hard_limit": 20,
                    "soft_min": 8,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 5.0,
                },
                "line2": {
                    "soft_target": 12,
                    "hard_limit": 20,
                    "soft_min": 8,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 5.0,
                },
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        segmented = segment(tokens, UnderflowScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "O", "SB"])

if __name__ == "__main__":
    unittest.main()
