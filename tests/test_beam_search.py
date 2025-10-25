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
                "line1": {"soft_target": 5, "hard_limit": 5},
                "line2": {"soft_target": 5, "hard_limit": 5},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "SB", "LB", "SB"])

    def test_refinement_pass_merges_single_word_cue(self):
        class RefinementScorer:
            def __init__(self):
                self.sl = {
                    "line_length_leniency": 1.0,
                    "orphan_leniency": 1.0,
                    "flow": 1.0,
                    "density": 1.0,
                    "balance": 1.0,
                    "structure": 1.0,
                }

            def score_transition(self, row):
                return {"O": -40.0, "LB": -100.0, "SB": 0.0}

            def score_block(self, block_tokens, block_breaks):
                return 100.0 if len(block_tokens) >= 2 else -5.0

        tokens = [
            make_token("one", 0.0),
            make_token("two", 0.4),
            make_token("three", 0.8),
            make_token("four", 1.2),
        ]

        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 42, "hard_limit": 42},
                "line2": {"soft_target": 42, "hard_limit": 42},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            enable_refinement_pass=True,
        )

        segmented = segment(tokens, RefinementScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["O", "SB", "O", "SB"])

if __name__ == "__main__":
    unittest.main()
