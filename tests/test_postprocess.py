from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isce.postprocess import postprocess
from isce.types import Token


class DummyScorer:
    """Minimal scorer that favors balanced, adequately long blocks."""

    min_duration = 0.5

    def score_block(self, block_tokens, block_breaks):
        if not block_tokens:
            return 0.0

        def count_chars(slice_tokens):
            if not slice_tokens:
                return 0
            return sum(len(tok.get("w", "")) for tok in slice_tokens) + max(0, len(slice_tokens) - 1)

        score = 0.0
        total_chars = count_chars(block_tokens)
        duration = max(1e-6, block_tokens[-1].get("end", 0.0) - block_tokens[0].get("start", 0.0))
        cps = total_chars / duration
        score += 1.0 - 0.05 * abs(cps - 15.0)

        lb_idx = next((i for i, br in enumerate(block_breaks) if br == "LB"), -1)
        if lb_idx != -1:
            len1 = count_chars(block_tokens[: lb_idx + 1])
            len2 = count_chars(block_tokens[lb_idx + 1 :])
            ratio = max(len1, len2) / max(1, min(len1, len2))
            score -= 0.1 * max(0.0, ratio - 1.0)
            if min(len1, len2) <= 1:
                score -= 1.0

        if duration < self.min_duration:
            score -= 3.0

        return score


@pytest.fixture
def dummy_scorer():
    return DummyScorer()


@pytest.fixture
def token_factory():
    def _factory(word, start, end=None, break_type=None):
        token = Token(w=word, start=start, end=end if end is not None else start + 0.4, speaker="A")
        if break_type:
            token = replace(token, break_type=break_type)
        return token

    return _factory


def test_rebalances_one_word_line(dummy_scorer, token_factory):
    tokens = [
        token_factory("Hello", 0.0, 0.3, break_type="LB"),
        token_factory("wonderful", 0.3, 0.6, break_type="O"),
        token_factory("audience", 0.6, 0.9, break_type="O"),
        token_factory("today", 0.9, 1.2, break_type="SB"),
    ]

    adjusted = postprocess(tokens, dummy_scorer)

    assert adjusted[0].break_type == "O"
    assert adjusted[1].break_type == "LB"


def test_merges_short_single_word_block(dummy_scorer, token_factory):
    tokens = [
        token_factory("Yes", 0.0, 0.3, break_type="SB"),
        token_factory("We", 0.3, 0.6, break_type="LB"),
        token_factory("really", 0.6, 0.9, break_type="O"),
        token_factory("agree", 0.9, 1.4, break_type="SB"),
    ]

    adjusted = postprocess(tokens, dummy_scorer)

    assert [tok.break_type for tok in adjusted] == ["O", "LB", "O", "SB"]
