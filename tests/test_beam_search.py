import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.beam_search import Segmenter, PathState, segment
from isce.config import Config
from isce.types import Token


class DummyScorer:
    def __init__(self):
        self.sl = {
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "single_word_line_penalty": 0.0,
            "extreme_balance_penalty": 0.0,
            "extreme_balance_threshold": 2.5,
        }

    def score_transition(self, row):
        word = row.token.get("w", "")
        if word.endswith("0"):
            return {"O": -5.0, "LB": 5.0, "SB": -5.0}
        if word.endswith("2"):
            return {"O": -5.0, "LB": 10.0, "SB": -1.0}
        return {"O": -5.0, "LB": -5.0, "SB": -5.0}

    def score_block(self, block_tokens, block_breaks):
        return 0.0


def make_token(word: str, start: float, pos: str | None = None) -> Token:
    return Token(w=word, start=start, end=start + 0.2, speaker="A", pos=pos)


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
            allowed_single_word_proper_nouns=(),
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "SB", "LB", "SB"])

    def test_single_word_line_rejected_without_whitelist(self):
        tokens = [make_token("Hello", 0.0), make_token("world", 0.2)]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=2,
            sliders={},
            paths={},
            allowed_single_word_proper_nouns=(),
        )
        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(score=0.0, line_num=1, line_len=len(tokens[0].w), block_start_idx=0, breaks=())
        self.assertFalse(segmenter._is_hard_ok_SB(state, 0))

    def test_whitelisted_proper_noun_allowed(self):
        tokens = [make_token("NASA", 0.0, pos="PROPN")]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=8,
            sliders={},
            paths={},
            allowed_single_word_proper_nouns=("NASA",),
        )
        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(score=0.0, line_num=1, line_len=len(tokens[0].w), block_start_idx=0, breaks=())
        self.assertTrue(segmenter._is_hard_ok_SB(state, 0))

if __name__ == "__main__":
    unittest.main()
