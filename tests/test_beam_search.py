import unittest
from collections import Counter
from dataclasses import replace

from isce.beam_search import (
    Segmenter,
    PathState,
    _token_to_row_dict,
    _map_reversed_breaks,
    _reverse_tokens_for_bidirectional,
    _reconcile_bidirectional_breaks,
    _score_path,
    _run_forward_breaks,
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


class RecordingScorer:
    def __init__(self, single_word_line_penalty: float = 0.0):
        self.sl = {
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "single_word_line_penalty": single_word_line_penalty,
            "extreme_balance_penalty": 0.0,
            "extreme_balance_threshold": 2.5,
        }
        self.lookahead_payloads: list[tuple[dict, ...] | None] = []
        self.block_token_indices: list[list[int | None]] = []

    def score_transition(self, row, ctx=None):
        self.lookahead_payloads.append(row.lookahead)
        return {"O": 0.0, "LB": 0.0, "SB": 0.0}

    def score_block(self, block_tokens, block_breaks):
        self.block_token_indices.append([token.get("token_index") for token in block_tokens])
        return 0.0


class TokenIndexRecordingScorer:
    def __init__(self):
        self.sl = {
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "single_word_line_penalty": 0.0,
            "extreme_balance_penalty": 0.0,
            "extreme_balance_threshold": 2.5,
        }
        self.reset()

    def reset(self) -> None:
        self.transition_indices: list[int | None] = []
        self.pending_indices: list[tuple[int | None, ...]] = []
        self.lookahead_indices: list[tuple[int | None, ...] | None] = []
        self.block_indices: list[tuple[int | None, ...]] = []

    def score_transition(self, row, ctx=None):
        self.transition_indices.append(row.token.get("token_index"))
        if ctx is not None:
            self.pending_indices.append(
                tuple(token.get("token_index") for token in ctx.pending_tokens)
            )
        if row.lookahead is None:
            self.lookahead_indices.append(None)
        else:
            self.lookahead_indices.append(
                tuple(token.get("token_index") for token in row.lookahead)
            )
        return {"O": 0.0, "LB": 0.0, "SB": 0.0}

    def score_block(self, block_tokens, block_breaks):
        self.block_indices.append(
            tuple(token.get("token_index") for token in block_tokens)
        )
        return 0.0


def make_token(word: str, start: float, **overrides) -> Token:
    defaults = dict(w=word, start=start, end=start + 0.2, speaker="A")
    defaults.update(overrides)
    return Token(**defaults)


def test_token_to_row_dict_copies_payload_and_assigns_index() -> None:
    token = make_token("hello", 0.0)
    payload = _token_to_row_dict(token, idx=5)

    assert payload["token_index"] == 5
    assert payload["w"] == "hello"

    # Mutating the returned payload should not affect the original dataclass.
    payload["w"] = "changed"
    assert token.w == "hello"


def test_token_to_row_dict_normalises_index_and_word() -> None:
    source = {"w": 123, "token_index": "7"}
    payload = _token_to_row_dict(source, idx=3)

    assert payload["token_index"] == 7
    assert payload["w"] == "123"

    # Invalid indexes should fall back to the provided idx.
    broken = {"w": "ok", "token_index": "not-a-number"}
    payload = _token_to_row_dict(broken, idx=9)
    assert payload["token_index"] == 9

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

    def test_reverse_tokens_backfill_missing_indices(self) -> None:
        tokens = [
            make_token("one", 0.0),
            replace(make_token("two", 0.4), token_index=42),
            make_token("three", 0.8),
        ]

        reversed_tokens = _reverse_tokens_for_bidirectional(tokens)
        indices = [token.token_index for token in reversed_tokens]

        self.assertEqual(indices, [2, 42, 0])

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

    def test_transition_context_projects_second_line_from_first_line(self):
        tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
            make_token("gamma", 0.4),
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
            allowed_single_word_proper_nouns=(),
        )

        segmenter = Segmenter(tokens, DummyScorer(), cfg)
        first_line_state = PathState(
            score=0.0,
            line_num=1,
            line_len=len(tokens[0].w) + 1 + len(tokens[1].w),
            block_start_idx=0,
            breaks=("O",),
        )
        context = segmenter._build_transition_context(first_line_state, 1)
        self.assertEqual(context.current_line_num, 1)
        self.assertEqual(context.current_line_len, len(tokens[0].w) + 1 + len(tokens[1].w))
        self.assertIsNotNone(context.projected_second_line_chars)
        self.assertIsNotNone(context.projected_second_line_words)

        second_line_state = PathState(
            score=0.0,
            line_num=2,
            line_len=len(tokens[2].w),
            block_start_idx=0,
            breaks=("O", "LB"),
        )
        second_context = segmenter._build_transition_context(second_line_state, 2)
        self.assertIsNone(second_context.projected_second_line_chars)
        self.assertIsNone(second_context.projected_second_line_words)

    def test_transition_context_pending_tokens_respect_start_offset(self) -> None:
        tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
        ]

        cfg = Config(
            beam_width=1,
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
            allowed_single_word_proper_nouns=(),
        )

        segmenter = Segmenter(list(tokens), DummyScorer(), cfg, start_offset=7)
        state = PathState(
            score=0.0,
            line_num=1,
            line_len=len(tokens[0].w) + 1 + len(tokens[1].w),
            block_start_idx=0,
            breaks=("O",),
        )

        context = segmenter._build_transition_context(state, 1)
        indices = [token.get("token_index") for token in context.pending_tokens]
        self.assertEqual(indices, [7, 8])

    def test_score_path_exposes_lookahead(self):
        tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
            make_token("gamma", 0.4),
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
            lookahead_width=2,
            allowed_single_word_proper_nouns=(),
        )

        scorer = RecordingScorer()
        breaks = ["O", "O", "SB"]
        _score_path(tokens, breaks, scorer, cfg)

        self.assertEqual(len(scorer.lookahead_payloads), 3)
        self.assertIsNone(scorer.lookahead_payloads[-1])
        self.assertIsNotNone(scorer.lookahead_payloads[0])
        self.assertEqual(len(scorer.lookahead_payloads[0]), 2)

    def test_score_path_assigns_block_indices_with_offset(self) -> None:
        tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
        ]

        cfg = Config(
            beam_width=1,
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
            allowed_single_word_proper_nouns=(),
        )

        scorer = RecordingScorer()
        breaks = ["O", "SB"]

        _score_path(tokens, breaks, scorer, cfg, start_offset=11)

        self.assertEqual(scorer.block_token_indices, [[11, 12]])

    def test_token_index_propagates_through_all_scoring_paths(self) -> None:
        def chunk_sequence(seq, size):
            return [tuple(seq[i : i + size]) for i in range(0, len(seq), size)]

        base_tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
            make_token("gamma", 0.4),
        ]

        cfg = Config(
            beam_width=1,
            min_block_duration_s=0.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 20, "hard_limit": 30},
                "line2": {"soft_target": 20, "hard_limit": 30},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=2,
            allowed_single_word_proper_nouns=(),
        )

        recorder = TokenIndexRecordingScorer()
        _run_forward_breaks(base_tokens, recorder, cfg)

        self.assertEqual(recorder.transition_indices[:3], [0, 1, 2])
        self.assertEqual(
            recorder.pending_indices[:3], [(0,), (0, 1), (0, 1, 2)]
        )
        self.assertEqual(
            recorder.lookahead_indices[:3], [(1, 2), (2,), None]
        )
        for block_payload in recorder.block_indices:
            self.assertTrue(all(idx in {0, 1, 2} for idx in block_payload))
            self.assertTrue(all(idx is not None for idx in block_payload))

        recorder.reset()
        forward_breaks = ["O", "LB", "SB"]
        backward_breaks = ["LB", "O", "SB"]
        _reconcile_bidirectional_breaks(
            forward_breaks, backward_breaks, recorder, base_tokens, cfg
        )

        transition_chunks = chunk_sequence(recorder.transition_indices, len(base_tokens))
        for chunk in transition_chunks:
            self.assertEqual(chunk, (0, 1, 2))

        pending_chunks = chunk_sequence(recorder.pending_indices, len(base_tokens))
        for chunk in pending_chunks:
            self.assertEqual(chunk, ((0,), (0, 1), (0, 1, 2)))

        lookahead_chunks = chunk_sequence(recorder.lookahead_indices, len(base_tokens))
        for chunk in lookahead_chunks:
            self.assertEqual(chunk, ((1, 2), (2,), None))

        self.assertTrue(
            all(block == (0, 1, 2) for block in recorder.block_indices)
        )

        recorder.reset()
        refinement_tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
            make_token("gamma", 0.4),
            make_token("delta", 0.6),
        ]
        refinement_breaks = ["O", "SB", "SB", "SB"]
        refinement_cfg = Config(
            beam_width=1,
            min_block_duration_s=0.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 20, "hard_limit": 30},
                "line2": {"soft_target": 20, "hard_limit": 30},
            },
            min_chars_for_single_word_block=1,
            sliders={},
            paths={},
            lookahead_width=1,
            allowed_single_word_proper_nouns=(),
        )

        refine_blocks(refinement_tokens, refinement_breaks, recorder, refinement_cfg)

        self.assertGreaterEqual(len(recorder.transition_indices), 2)
        self.assertEqual(recorder.transition_indices[0:2], [2, 3])
        self.assertEqual(recorder.pending_indices[0:2], [(2,), (3,)])
        self.assertEqual(recorder.lookahead_indices[0], (3,))
        self.assertIsNone(recorder.lookahead_indices[1])
        flattened_block_indices = [idx for block in recorder.block_indices for idx in block]
        self.assertIn(2, flattened_block_indices)
        self.assertIn(3, flattened_block_indices)
        self.assertTrue(all(idx is not None for idx in flattened_block_indices))

        recorder.reset()
        refine_blocks(
            refinement_tokens,
            refinement_breaks,
            recorder,
            refinement_cfg,
            start_offset=5,
        )

        self.assertGreaterEqual(len(recorder.transition_indices), 2)
        self.assertEqual(recorder.transition_indices[0:2], [7, 8])
        self.assertEqual(recorder.pending_indices[0:2], [(7,), (8,)])
        self.assertEqual(recorder.lookahead_indices[0], (8,))
        self.assertIsNone(recorder.lookahead_indices[1])
        offset_block_indices = [idx for block in recorder.block_indices for idx in block]
        self.assertIn(7, offset_block_indices)
        self.assertIn(8, offset_block_indices)
        self.assertTrue(all(idx is not None for idx in offset_block_indices))

    def test_fallback_uses_single_word_slider_penalty(self):
        tokens = [make_token("Hi", 0.0)]
        base_cfg = Config(
            beam_width=1,
            min_block_duration_s=5.0,
            max_block_duration_s=10.0,
            line_length_constraints={
                "line1": {"soft_target": 5, "hard_limit": 10},
                "line2": {"soft_target": 5, "hard_limit": 10},
            },
            min_chars_for_single_word_block=4,
            sliders={},
            paths={},
            lookahead_width=0,
            allowed_single_word_proper_nouns=(),
        )

        zero_slider = RecordingScorer(single_word_line_penalty=0.0)
        slider_segmenter = Segmenter(tokens, zero_slider, base_cfg)
        zero_breaks = slider_segmenter.run()
        self.assertEqual(zero_breaks, ["SB"])
        zero_score = slider_segmenter.last_path_score

        custom_slider = RecordingScorer(single_word_line_penalty=7.0)
        custom_segmenter = Segmenter(tokens, custom_slider, base_cfg)
        custom_breaks = custom_segmenter.run()
        self.assertEqual(custom_breaks, ["SB"])
        custom_score = custom_segmenter.last_path_score

        self.assertIsNotNone(zero_score)
        self.assertIsNotNone(custom_score)
        self.assertGreater(custom_score, zero_score)
        self.assertAlmostEqual(custom_score - zero_score, 18.0, places=3)

    def test_segmenter_fallback_block_scoring_respects_start_offset(self) -> None:
        tokens = [
            make_token("alpha", 0.0),
            make_token("beta", 0.2),
        ]

        cfg = Config(
            beam_width=1,
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
            allowed_single_word_proper_nouns=(),
        )

        class ForcedSegmenter(Segmenter):
            def _is_hard_ok_O(self, *args, **kwargs):
                return False

            def _is_hard_ok_LB(self, *args, **kwargs):
                return False

            def _is_hard_ok_SB(self, *args, **kwargs):
                return False

        scorer = RecordingScorer()
        segmenter = ForcedSegmenter(list(tokens), scorer, cfg, start_offset=9)
        segmenter.run()

        self.assertEqual(scorer.block_token_indices, [[9], [10]])

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

    def test_multi_line_single_word_second_line_allowed(self):
        tokens = [
            make_token("Keep", 0.0),
            make_token("going", 0.3),
            make_token("now", 0.6),
        ]
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
            line_num=2,
            line_len=len(tokens[2].w),
            block_start_idx=0,
            breaks=("O", "LB"),
        )

        self.assertTrue(segmenter._is_hard_ok_SB(state, 2))

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

    def test_allows_single_word_second_line_when_block_has_context(self):
        tokens = [
            make_token("Please", 0.0),
            make_token("keep", 0.2),
            make_token("going", 0.4),
        ]
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
        lines = [tokens[:2], tokens[2:]]
        violations = segmenter._line_violations(lines)

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
