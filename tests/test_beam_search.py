import unittest
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.beam_search import (
    Segmenter,
    PathState,
    _map_reversed_breaks,
    _reverse_tokens_for_bidirectional,
    _reconcile_bidirectional_breaks,
    _score_path,
    refine_blocks,
    segment,
)
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


class BlockPreferenceScorer:
    def __init__(self):
        self.sl = {
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "single_word_line_penalty": 0.0,
            "extreme_balance_penalty": 0.0,
            "extreme_balance_threshold": 2.5,
            "fragment_char_threshold": 8.0,
            "fragment_penalty": 6.0,
        }

    def score_transition(self, row, ctx=None):
        return {"O": 0.0, "LB": 0.0, "SB": 0.0}

    def score_block(self, block_tokens, block_breaks):
        words = [token.get("w") for token in block_tokens]
        if words == ["a", "b", "c", "d"]:
            return -10.0
        if words == ["a", "b"]:
            return 6.0
        if words == ["c", "d"]:
            return 6.0
        return 0.0


class RefinementScorer:
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
        if word == "alpha":
            return {"O": 0.0, "LB": -999.0, "SB": 8.0}
        return {"O": -999.0, "LB": -999.0, "SB": 0.0}

    def score_block(self, block_tokens, block_breaks):
        words = [token.get("w") for token in block_tokens]
        if words == ["alpha"]:
            return -5.0
        if words == ["alpha", "beta"]:
            return 5.0
        return 0.0


def make_token(word: str, start: float, **overrides) -> Token:
    defaults = dict(w=word, start=start, end=start + 0.2, speaker="A")
    defaults.update(overrides)
    return Token(**defaults)


class TestBeamSearch(unittest.TestCase):
    def test_reverse_tokens_flip_speaker_change(self) -> None:
        tokens = [
            make_token("hello", 0.0, speaker="A", speaker_change=True),
            make_token("there", 0.5, speaker="B", speaker_change=False),
            make_token("friend", 1.0, speaker="B", speaker_change=False),
        ]

        reversed_tokens = _reverse_tokens_for_bidirectional(tokens)

        self.assertEqual(len(reversed_tokens), 3)
        self.assertFalse(reversed_tokens[0].speaker_change)
        self.assertTrue(reversed_tokens[1].speaker_change)
        self.assertFalse(reversed_tokens[2].speaker_change)

    def test_segmenter_records_last_path_score(self):
        class ConstantScorer:
            def __init__(self):
                self.sl = {
                    "line_length_leniency": 1.0,
                    "orphan_leniency": 1.0,
                    "single_word_line_penalty": 0.0,
                    "extreme_balance_penalty": 0.0,
                    "extreme_balance_threshold": 2.5,
                }

            def score_transition(self, row, ctx=None):
                return {"O": -1.0, "LB": 4.0, "SB": 0.0}

            def score_block(self, block_tokens, block_breaks):
                return 0.0

        tokens = [
            make_token("alpha", 0.0, pos="PROPN"),
            make_token("beta", 0.4, pos="PROPN"),
        ]

        cfg = Config(
            beam_width=2,
            min_block_duration_s=0.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 20, "hard_limit": 30},
                "line2": {"soft_target": 20, "hard_limit": 30},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=("alpha", "beta"),
        )

        scorer = ConstantScorer()
        segmenter = Segmenter(list(tokens), scorer, cfg)
        breaks = segmenter.run()

        self.assertEqual(breaks, ["LB", "SB"])
        self.assertIsNotNone(segmenter.last_path_score)
        rescored = _score_path(tokens, breaks, scorer, cfg)
        self.assertAlmostEqual(segmenter.last_path_score, rescored)

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
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks, ["LB", "SB", "LB", "SB"])

    def test_lookahead_prefers_break_before_rapid_turn(self):
        def make_tokens() -> list[Token]:
            return [
                make_token("Hello", 0.0, pause_after_ms=40, speaker="A", pos="PROPN"),
                make_token("there,", 0.2, pause_after_ms=60, speaker="A", speaker_change=True, pos="PROPN"),
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
                "line1": {"soft_target": 42, "hard_limit": 50},
                "line2": {"soft_target": 42, "hard_limit": 50},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=("Hello", "there"),
        )
        lookahead_cfg = Config(
            beam_width=5,
            min_block_duration_s=0.05,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 42, "hard_limit": 50},
                "line2": {"soft_target": 42, "hard_limit": 50},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=2,
            allowed_single_word_proper_nouns=("Hello", "there"),
        )

        base_scorer = Scorer({}, constraints, sliders, base_cfg)
        lookahead_scorer = Scorer({}, constraints, sliders, lookahead_cfg)

        base_breaks = [token.break_type for token in segment(make_tokens(), base_scorer, base_cfg)]
        lookahead_breaks = [token.break_type for token in segment(make_tokens(), lookahead_scorer, lookahead_cfg)]

        self.assertEqual(base_breaks[0], "O")
        self.assertIn(lookahead_breaks[0], {"LB", "SB"})
        self.assertNotEqual(base_breaks, lookahead_breaks)

    def test_short_single_word_line_rejected_without_whitelist(self):
        tokens = [make_token("Go", 0.0), make_token("now", 0.2)]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=4,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )
        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(score=0.0, line_num=1, line_len=len(tokens[0].w), block_start_idx=0, breaks=())
        self.assertFalse(segmenter._is_hard_ok_SB(state, 0))

    def test_single_word_above_threshold_allowed_without_whitelist(self):
        tokens = [make_token("Wonderful", 0.0)]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=6,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )
        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(score=0.0, line_num=1, line_len=len(tokens[0].w), block_start_idx=0, breaks=())
        self.assertTrue(segmenter._is_hard_ok_SB(state, 0))

    def test_multi_word_short_line_allowed(self):
        tokens = [make_token("Go", 0.0), make_token("now", 0.2)]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=10,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(
            score=0.0,
            line_num=1,
            line_len=len(tokens[0].w) + 1 + len(tokens[1].w),
            block_start_idx=0,
            breaks=("O",),
        )

        self.assertTrue(segmenter._is_hard_ok_SB(state, 1))

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
            lookahead_width=0,
            allowed_single_word_proper_nouns=("NASA",),
        )
        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        state = PathState(score=0.0, line_num=1, line_len=len(tokens[0].w), block_start_idx=0, breaks=())
        self.assertTrue(segmenter._is_hard_ok_SB(state, 0))

    def test_lookahead_discourages_orphan_second_line(self):
        tokens = [
            make_token("AAAAA0", 0.0),
            make_token("I", 0.2, is_sentence_final=True),
            make_token("BBBBB3", 0.4),
        ]

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
            lookahead_width=0,
        )

        segmented = segment(tokens, DummyScorer(), cfg)
        breaks = [token.break_type for token in segmented]

        self.assertEqual(breaks[0], "O")
        self.assertNotIn("LB", breaks[:2])

    def test_map_reversed_breaks_preserves_backward_boundaries(self):
        reversed_breaks = ["SB", "O", "SB", "SB"]
        mapped = _map_reversed_breaks(reversed_breaks)
        self.assertEqual(mapped, ["SB", "O", "SB", "SB"])

    def test_bidirectional_reconciliation_prefers_higher_score(self):
        tokens = [
            make_token("a", 0.0),
            make_token("b", 0.2),
            make_token("c", 0.4),
            make_token("d", 0.6),
        ]

        forward_breaks = ["O", "O", "O", "SB"]
        backward_breaks = ["O", "SB", "O", "SB"]

        cfg = Config(
            beam_width=2,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        scorer = BlockPreferenceScorer()
        reconciled = _reconcile_bidirectional_breaks(
            forward_breaks, backward_breaks, scorer, tokens, cfg
        )

        self.assertEqual(reconciled, ["O", "SB", "O", "SB"])

    def test_refinement_pass_merges_single_word_block(self):
        def make_tokens():
            return [
                make_token("alpha", 0.0, pos="PROPN"),
                make_token("beta", 0.5),
            ]

        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=("alpha",),
        )

        scorer = RefinementScorer()

        baseline_breaks = [token.break_type for token in segment(make_tokens(), scorer, cfg)]
        self.assertEqual(baseline_breaks, ["SB", "SB"])

        refined_cfg = replace(cfg, enable_refinement_pass=True)
        refined_breaks = [
            token.break_type for token in segment(make_tokens(), scorer, refined_cfg)
        ]
        self.assertEqual(refined_breaks, ["O", "SB"])

    def test_refine_blocks_clamps_single_word_window(self):
        tokens = [
            make_token("alpha", 0.0, pos="PROPN"),
            make_token("beta", 0.5),
        ]

        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=("alpha",),
        )

        scorer = RefinementScorer()
        baseline_breaks = ["SB", "SB"]

        refined = refine_blocks(tokens, baseline_breaks, scorer, cfg)

        self.assertEqual(refined, ["O", "SB"])


class TestLineViolations(unittest.TestCase):
    def test_flags_short_single_word_lines(self):
        tokens = [make_token("Hi", 0.0, pos="INTJ")]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=6,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        violations = segmenter._line_violations([tokens])

        assert violations == Counter({"single_word": 1, "short_line": 1})

    def test_ignores_multi_word_short_lines(self):
        tokens = [make_token("Go", 0.0), make_token("now", 0.2)]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=10,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        violations = segmenter._line_violations([tokens])

        assert violations == Counter()

    def test_respects_whitelisted_single_words(self):
        tokens = [make_token("NASA", 0.0, pos="PROPN")]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=12,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=("NASA",),
        )

        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        violations = segmenter._line_violations([tokens])

        assert violations == Counter()

    def test_allows_long_single_word_without_whitelist(self):
        tokens = [make_token("Outstanding", 0.0, pos="ADJ")]
        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.1,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 12, "hard_limit": 20},
                "line2": {"soft_target": 12, "hard_limit": 20},
            },
            min_chars_for_single_word_block=6,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        violations = segmenter._line_violations([tokens])

        assert violations == Counter()


if __name__ == "__main__":
    unittest.main()
