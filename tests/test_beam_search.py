import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.beam_search import (
    PathState,
    Segmenter,
    _reverse_tokens_for_bidirectional,
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

    def score_transition(self, row):
        word = row.token.get("w", "")
        if word.endswith("0"):
            return {"O": -5.0, "LB": 5.0, "SB": -5.0}
        if word.endswith("2"):
            return {"O": -5.0, "LB": 10.0, "SB": -1.0}
        return {"O": -5.0, "LB": -5.0, "SB": -5.0}

    def score_block(self, block_tokens, block_breaks):
        return 0.0


def make_token(word: str, start: float, **overrides) -> Token:
    defaults = dict(
        w=word,
        start=start,
        end=start + 0.2,
        speaker="A",
        pause_after_ms=0,
        pause_before_ms=0,
    )
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
                "line1": {"soft_target": 5, "hard_limit": 5},
                "line2": {"soft_target": 5, "hard_limit": 5},
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
                make_token("Hello", 0.0, pause_after_ms=40, speaker="A", pos="PROPN"),
                make_token(
                    "there,",
                    0.2,
                    pause_after_ms=60,
                    speaker="A",
                    speaker_change=True,
                ),
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
            allowed_single_word_proper_nouns=("Hello",),
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
            allowed_single_word_proper_nouns=("Hello",),
        )

        base_scorer = Scorer({}, constraints, sliders, base_cfg)
        lookahead_scorer = Scorer({}, constraints, sliders, lookahead_cfg)

        base_breaks = [token.break_type for token in segment(make_tokens(), base_scorer, base_cfg)]
        lookahead_breaks = [token.break_type for token in segment(make_tokens(), lookahead_scorer, lookahead_cfg)]

        self.assertEqual(base_breaks[0], "O")
        self.assertIn(lookahead_breaks[0], {"LB", "SB"})
        self.assertNotEqual(base_breaks, lookahead_breaks)

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
        state = PathState(
            score=0.0,
            line_num=1,
            line_len=len(tokens[0].w),
            block_start_idx=0,
            breaks=(),
        )
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
        state = PathState(
            score=0.0,
            line_num=1,
            line_len=len(tokens[0].w),
            block_start_idx=0,
            breaks=(),
        )
        self.assertTrue(segmenter._is_hard_ok_SB(state, 0))

    def test_reverse_tokens_shift_boundary_features(self):
        tokens = [
            make_token(
                "one",
                0.0,
                speaker="A",
                speaker_change=True,
                starts_with_dialogue_dash=True,
                num_unit_glue=True,
                is_llm_structural_break=True,
                is_dangling_eos=True,
            ),
            make_token(
                "two",
                0.2,
                speaker="B",
                speaker_change=False,
                starts_with_dialogue_dash=True,
                num_unit_glue=False,
                is_llm_structural_break=False,
                is_dangling_eos=False,
            ),
            make_token(
                "three",
                0.4,
                speaker="B",
                speaker_change=True,
                starts_with_dialogue_dash=False,
                num_unit_glue=True,
                is_llm_structural_break=False,
                is_dangling_eos=True,
            ),
        ]

        reversed_tokens = _reverse_tokens_for_bidirectional(tokens)

        self.assertEqual(
            [t.speaker_change for t in reversed_tokens],
            [tokens[1].speaker_change, tokens[0].speaker_change, False],
        )
        self.assertEqual(
            [t.starts_with_dialogue_dash for t in reversed_tokens],
            [tokens[1].starts_with_dialogue_dash, tokens[0].starts_with_dialogue_dash, False],
        )
        self.assertEqual(
            [t.num_unit_glue for t in reversed_tokens],
            [tokens[1].num_unit_glue, tokens[0].num_unit_glue, False],
        )
        self.assertEqual(
            [t.is_llm_structural_break for t in reversed_tokens],
            [tokens[1].is_llm_structural_break, tokens[0].is_llm_structural_break, False],
        )
        self.assertEqual(
            [t.is_dangling_eos for t in reversed_tokens],
            [tokens[1].is_dangling_eos, tokens[0].is_dangling_eos, False],
        )

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

    def test_refinement_retries_next_block_after_failed_merge(self):
        class RefinementScorer:
            def __init__(self):
                self.sl = {
                    "line_length_leniency": 1.0,
                    "orphan_leniency": 1.0,
                    "single_word_line_penalty": 0.0,
                    "fallback_sb_penalty": 25.0,
                }

            def score_transition(self, row):
                return {"O": 0.0, "LB": 0.0, "SB": 0.0}

            def score_block(self, block_tokens, block_breaks):
                if len(block_tokens) == 1:
                    return -2.0
                if "LB" in block_breaks:
                    return 5.0
                return -1.0

        tokens = [
            make_token("Hi", 0.0, end=0.2, speaker="A"),
            make_token("there", 0.2, end=0.6, speaker="A"),
            make_token("friend", 0.6, end=1.2, speaker="A"),
        ]
        breaks = ["SB", "O", "SB"]

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

        first_segmenter = MagicMock()
        first_segmenter.run.return_value = ["SB", "O", "SB"]
        first_segmenter.last_path_score = 0.0

        second_segmenter = MagicMock()
        second_segmenter.run.return_value = ["LB", "SB"]
        second_segmenter.last_path_score = 5.0

        scorer = RefinementScorer()

        with patch("isce.beam_search.Segmenter", side_effect=[first_segmenter, second_segmenter]) as mock_segmenter:
            refined = refine_blocks(tokens, breaks, scorer, cfg)

        self.assertEqual(refined, ["SB", "LB", "SB"])
        self.assertEqual(mock_segmenter.call_count, 2)
        self.assertEqual(
            mock_segmenter.call_args_list,
            [
                call(tokens, scorer, ANY),
                call(tokens[1:], scorer, ANY),
            ],
        )


if __name__ == "__main__":
    unittest.main()
