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

    def score_transition(self, row, ctx=None):
        word = row.token.get("w", "")
        if word.endswith("0"):
            scores = {"O": -5.0, "LB": 5.0, "SB": -5.0}
        if word.endswith("2"):
            scores = {"O": -5.0, "LB": 10.0, "SB": -1.0}
        else:
            scores = {"O": -5.0, "LB": -5.0, "SB": -5.0}

        if ctx and getattr(ctx, "projected_second_line_words", None) == 1:
            projected_chars = ctx.projected_second_line_chars or 0
            if projected_chars < 5:
                scores["LB"] -= 20.0

        return scores

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

    def test_lookahead_discourages_orphan_second_line(self):
        tokens = [
            make_token("AAAAA0", 0.0),
            make_token("I", 0.2),
            make_token("BBBBB3", 0.4),
        ]
        tokens[1] = Token(
            w="I",
            start=0.2,
            end=0.4,
            speaker="A",
            is_sentence_final=True,
        )

        cfg = Config(
            beam_width=2,
            min_block_duration_s=0.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 15, "hard_limit": 20},
                "line2": {"soft_target": 15, "hard_limit": 20},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks[0], "O")
        self.assertNotIn("LB", breaks[:2])

if __name__ == "__main__":
    unittest.main()
