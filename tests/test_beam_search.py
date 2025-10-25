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

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "SB", "LB", "SB"])

    def test_bidirectional_pass_improves_segmentation(self):
        tokens = [
            make_token("short", 0.0),
            make_token("line", 0.2),
            make_token("then", 0.4),
            make_token("a", 0.6),
            make_token("veryveryveryverylongline", 0.8),
        ]

        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 10, "hard_limit": 30},
                "line2": {"soft_target": 10, "hard_limit": 30},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            enable_bidirectional_pass=True,
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

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        # This is a simplified example. A real-world test would require a more sophisticated scorer.
        self.assertEqual(breaks, ["O", "LB", "O", "O", "SB"])

if __name__ == "__main__":
    unittest.main()
