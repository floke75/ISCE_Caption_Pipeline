import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.beam_search import Segmenter, PathState, segment
from isce.config import Config
from isce.scorer import Scorer
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

    def score_transition(self, row, ctx=None):
        word = row.token.get("w", "")
        if word.endswith("0"):
            scores = {"O": -5.0, "LB": 5.0, "SB": -5.0}
        elif word.endswith("2"):
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


def make_token(word: str, start: float, **overrides) -> Token:
    defaults = dict(w=word, start=start, end=start + 0.2, speaker="A")
    defaults.update(overrides)
    return Token(**defaults)


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
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "SB", "LB", "SB"])

    def test_lookahead_prefers_break_before_rapid_turn(self):
        def make_tokens() -> list[Token]:
            return [
                make_token("Hello", 0.0, pause_after_ms=40, speaker="A"),
                make_token("there,", 0.2, pause_after_ms=60, speaker="A", speaker_change=True),
                make_token(
                    "Yeah.",
                    0.4,
                    pause_after_ms=800,
                    speaker="B",
                    is_sentence_initial=True,
                    is_sentence_final=True,
                ),
            ]

        constraints = {
            "ideal_cps_iqr": [10.0, 18.0],
            "ideal_cps_median": 14.0,
            "ideal_balance_iqr": [0.7, 1.4],
        }

        sliders = {
            "flow": 1.0,
            "density": 0.0,
            "balance": 0.0,
            "structure": 1.0,
            "structure_boost": 12.0,
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
        }

        base_cfg = Config(
            beam_width=5,
            min_block_duration_s=0.05,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 42, "hard_limit": 50, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 42, "hard_limit": 50, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
        )
        lookahead_cfg = Config(
            beam_width=5,
            min_block_duration_s=0.05,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 42, "hard_limit": 50, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 42, "hard_limit": 50, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=2,
        )

        base_scorer = Scorer({}, constraints, sliders, base_cfg)
        lookahead_scorer = Scorer({}, constraints, sliders, lookahead_cfg)

        base_breaks = [token.break_type for token in segment(make_tokens(), base_scorer, base_cfg)]
        lookahead_breaks = [token.break_type for token in segment(make_tokens(), lookahead_scorer, lookahead_cfg)]

        self.assertEqual(base_breaks[0], "O")
        self.assertIn(lookahead_breaks[0], {"LB", "SB"})
        self.assertNotEqual(base_breaks, lookahead_breaks)

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
                "line1": {"soft_target": 15, "hard_limit": 20, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 15, "hard_limit": 20, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=2,
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks[0], "O")
        self.assertNotIn("LB", breaks[:2])

    def test_single_word_line_rejected_without_whitelist(self):
        tokens = [make_token("Hello", 0.0), make_token("world", 0.2)]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 12, "hard_limit": 20, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=6,
            sliders={},
            paths={},
            lookahead_width=0,
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
                "line1": {"soft_target": 12, "hard_limit": 20, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 12, "hard_limit": 20, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=8,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=("NASA",),
        )
        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(score=0.0, line_num=1, line_len=len(tokens[0].w), block_start_idx=0, breaks=())
        self.assertTrue(segmenter._is_hard_ok_SB(state, 0))

    def test_bidirectional_breaks_before_trailing_straggler(self):
        class PauseScorer:
            def __init__(self):
                self.sl = {"line_length_leniency": 1.0, "orphan_leniency": 1.0}

            def score_transition(self, row, ctx=None):
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
                "line1": {"soft_target": 25, "hard_limit": 30, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 25, "hard_limit": 30, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        scorer = PauseScorer()
        forward_breaks = [token.break_type for token in segment(tokens, scorer, cfg)]

        cfg.enable_bidirectional_pass = True
        bidirectional_breaks = [token.break_type for token in segment(tokens, scorer, cfg)]

        self.assertLessEqual(bidirectional_breaks.index("SB"), forward_breaks.index("SB"))

    def test_short_line_underflow_penalty_discourages_extension(self):
        tokens = [
            make_token("Hi", 0.0),
            make_token("a", 0.2),
            make_token("buddy", 0.4),
        ]

        class UnderflowScorer:
            def __init__(self):
                self.sl = {"line_length_leniency": 1.0, "orphan_leniency": 1.0}

            def score_transition(self, row, ctx=None):
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
                    "soft_min": 10,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.3,
                },
                "line2": {
                    "soft_target": 12,
                    "hard_limit": 20,
                    "soft_min": 10,
                    "soft_over_penalty_scale": 0.1,
                    "soft_under_penalty_scale": 0.3,
                },
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
        )

        breaks = [token.break_type for token in segment(tokens, UnderflowScorer(), cfg)]

        self.assertEqual(breaks[0], "LB")

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

            def score_transition(self, row, ctx=None):
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
                "line1": {"soft_target": 42, "hard_limit": 42, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "line2": {"soft_target": 42, "hard_limit": 42, "soft_min": 0, "soft_over_penalty_scale": 0.1, "soft_under_penalty_scale": 0.05},
                "block": {"min_total_chars": 0, "min_last_line_chars": 0},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            enable_refinement_pass=True,
        )

        breaks = [token.break_type for token in segment(tokens, RefinementScorer(), cfg)]

        self.assertEqual(breaks, ["O", "SB", "O", "SB"])


if __name__ == "__main__":
    unittest.main()
