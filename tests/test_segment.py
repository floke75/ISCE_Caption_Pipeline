"""Smoke tests for the beam-search segmentation wrapper."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from isce.beam_search import segment
from isce.config import Config
from isce.types import Token, TokenRow


class DummyScorer:
    """Minimal scorer with deterministic scores for test coverage."""

    def __init__(self) -> None:
        # Surface the slider knobs accessed by ``Segmenter`` with sensible defaults.
        self.sl = {
            "line_length_leniency": 1.0,
            "orphan_leniency": 1.0,
            "fallback_sb_penalty": 5.0,
        }

    def score_transition(self, row: TokenRow, ctx=None) -> Dict[str, float]:
        if row.nxt is None:
            # Always close the subtitle when we reach the final token.
            return {"O": -5.0, "LB": -5.0, "SB": 20.0}

        # Encourage a line break after the sentence-final token and otherwise keep flowing.
        if row.token.get("is_sentence_final"):
            return {"O": -2.0, "LB": 10.0, "SB": -10.0}

        return {"O": 5.0, "LB": -5.0, "SB": -10.0}

    def score_block(self, block_tokens: List[dict], block_breaks: List[str]) -> float:
        # Keep the arithmetic simple: reward emitting a block that ends the sentence.
        last_word = block_tokens[-1]["w"]
        return 5.0 if last_word.endswith(".") else 0.0


def make_config() -> Config:
    return Config(
        beam_width=3,
        min_block_duration_s=0.1,
        max_block_duration_s=10.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
        },
        min_chars_for_single_word_block=1,
        sliders={},
        paths={},
    )


def make_tokens() -> List[Token]:
    base = dict(start=0.0, end=0.5, speaker="Narrator")
    return [
        Token(w="Hello", **base),
        replace(Token(w="world.", **base), is_sentence_final=True, end=1.0),
        replace(Token(w="Again", **base), start=1.2, end=1.6),
        replace(Token(w="here.", **base), start=1.7, end=2.2, is_sentence_final=True),
    ]


def test_segment_assigns_breaks():
    tokens = make_tokens()
    scorer = DummyScorer()
    cfg = make_config()

    segmented = segment(tokens, scorer, cfg)

    assert [t.break_type for t in segmented] == ["O", "LB", "O", "SB"]


def test_segment_handles_empty_token_list():
    cfg = make_config()
    scorer = DummyScorer()

    assert segment([], scorer, cfg) == []
