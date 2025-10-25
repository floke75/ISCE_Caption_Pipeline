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

    def test_bidirectional_breaks_before_trailing_straggler(self):
        class PauseScorer:
            def __init__(self):
                self.sl = {"line_length_leniency": 1.0, "orphan_leniency": 1.0}

            def score_transition(self, row):
                pause_after = row.token.get("pause_after_ms", 0)
                base = {"O": 0.0, "LB": -10.0, "SB": -1.0}
                if pause_after >= 500:
                    base["SB"] = 12.0
                return base

            def score_block(self, block_tokens, block_breaks):
                return 0.0

        tokens = [
            Token(w="alpha", start=0.0, end=0.5, speaker="A", pause_after_ms=100, pause_before_ms=0, relative_position=0.0),
            Token(w="beta", start=0.5, end=1.0, speaker="A", pause_after_ms=120, pause_before_ms=100, relative_position=0.5),
            Token(w="gamma", start=1.0, end=1.6, speaker="A", pause_after_ms=0, pause_before_ms=900, relative_position=1.0),
        ]

        cfg = Config(
            beam_width=3,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 25, "hard_limit": 30},
                "line2": {"soft_target": 25, "hard_limit": 30},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        scorer = PauseScorer()
        forward_only = segment(tokens, scorer, cfg)
        forward_breaks = [token.break_type for token in forward_only]

        bidirectional = segment(tokens, scorer, cfg, bidirectional=True)
        bidir_breaks = [token.break_type for token in bidirectional]

        self.assertEqual(forward_breaks[:2], ["O", "O"])
        self.assertEqual(bidir_breaks[:2], ["O", "SB"])
        self.assertEqual(bidir_breaks[-1], "SB")

if __name__ == "__main__":
    unittest.main()
