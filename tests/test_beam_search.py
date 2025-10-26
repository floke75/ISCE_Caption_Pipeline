import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.beam_search import segment, _refine_blocks
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


class RefinementScorer:
    def __init__(self):
        self.sl = {
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "fallback_sb_penalty": 25.0,
        }
        self.low_score_calls = 0

    def score_transition(self, row):
        return {"O": 0.0, "LB": 0.0, "SB": 0.0}

    def score_block(self, block_tokens, block_breaks):
        if block_tokens and block_tokens[0].get("w", "").startswith("bad"):
            self.low_score_calls += 1
            return -10.0
        return 0.0


def make_token(word: str, start: float) -> Token:
    return Token(w=word, start=start, end=start + 0.2, speaker="A")


def make_cfg(**overrides) -> Config:
    base_constraints = {
        "line1": {
            "soft_target": 37,
            "hard_limit": 42,
            "soft_min": 0,
            "soft_over_penalty_scale": 0.1,
            "soft_under_penalty_scale": 0.05,
        },
        "line2": {
            "soft_target": 37,
            "hard_limit": 42,
            "soft_min": 0,
            "soft_over_penalty_scale": 0.1,
            "soft_under_penalty_scale": 0.05,
        },
        "block": {"min_total_chars": 0, "min_last_line_chars": 0},
    }

    cfg_kwargs = dict(
        beam_width=1,
        min_block_duration_s=1.0,
        max_block_duration_s=8.0,
        line_length_constraints=base_constraints,
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
        line_length_soft_min=0,
        line_length_overflow_scale=0.1,
        line_length_underflow_scale=0.05,
        min_total_chars_per_block=0,
        min_last_line_chars=0,
        short_block_penalty=0.0,
        short_line_penalty=0.0,
        extreme_balance_threshold=3.0,
        allowed_single_word_proper_nouns=set(),
    )
    cfg_kwargs.update(overrides)
    return Config(**cfg_kwargs)


class TestBeamSearch(unittest.TestCase):
    def test_fallback_candidate_keeps_beam_alive(self):
        tokens = [
            make_token("AAAAA0", 0.0),
            make_token("BBBBB1", 0.2),
            make_token("CCCCC2", 0.4),
            make_token("DDDDD3", 0.6),
        ]

        cfg = make_cfg(
            min_block_duration_s=10.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {
                    "soft_target": 5,
                    "hard_limit": 5,
                    "soft_min": 0,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.05,
                },
                "line2": {
                    "soft_target": 5,
                    "hard_limit": 5,
                    "soft_min": 0,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.05,
                },
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
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

        cfg = make_cfg(
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {
                    "soft_target": 10,
                    "hard_limit": 30,
                    "soft_min": 0,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.05,
                },
                "line2": {
                    "soft_target": 10,
                    "hard_limit": 30,
                    "soft_min": 0,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.05,
                },
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            enable_bidirectional_pass=True,
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        # This is a simplified example. A real-world test would require a more sophisticated scorer.
        self.assertEqual(breaks, ["O", "LB", "O", "O", "SB"])

    def test_refinement_preserves_context_breaks(self):
        scorer = RefinementScorer()
        tokens = []
        words = [
            ("good0", "O"),
            ("good1", "O"),
            ("good2", "SB"),
            ("bad0", "O"),
            ("bad1", "O"),
            ("bad2", "SB"),
            ("mid0", "O"),
            ("mid1", "O"),
            ("mid2", "SB"),
            ("tail0", "O"),
            ("tail1", "O"),
            ("tail2", "SB"),
        ]

        for idx, (word, br) in enumerate(words):
            tokens.append(Token(w=word, start=idx * 0.5, end=idx * 0.5 + 0.4, speaker="A", break_type=br))

        cfg = make_cfg(
            beam_width=2,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {
                    "soft_target": 50,
                    "hard_limit": 60,
                    "soft_min": 0,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.05,
                },
                "line2": {
                    "soft_target": 50,
                    "hard_limit": 60,
                    "soft_min": 0,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.05,
                },
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            enable_refinement_pass=True,
        )

        refined = _refine_blocks(tokens, scorer, cfg)

        self.assertGreaterEqual(scorer.low_score_calls, 1)
        self.assertEqual(refined[9].break_type, "O")
        self.assertEqual(refined[-1].break_type, "SB")

if __name__ == "__main__":
    unittest.main()
