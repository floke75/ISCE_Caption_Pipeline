import math

from isce.beam_search import _token_to_row_dict
from isce.types import Token


def _make_token(**overrides) -> Token:
    base = dict(w="hi", start=0.0, end=0.5, speaker="narrator")
    base.update(overrides)
    return Token(**base)


def test_token_to_row_dict_sets_index_for_dataclass_without_token_index():
    token = _make_token(token_index=None)

    result = _token_to_row_dict(token, 7)

    assert result is not None
    assert result["token_index"] == 7
    # Original token remains untouched.
    assert token.token_index is None


def test_token_to_row_dict_preserves_existing_index_on_dict():
    payload = {"w": "hi", "token_index": 3}

    result = _token_to_row_dict(payload, 99)

    assert result is not None
    assert result["token_index"] == 3
    assert payload["token_index"] == 3
    assert result is not payload


def test_token_to_row_dict_coerces_string_indexes():
    payload = {"w": "hi", "token_index": "12"}

    result = _token_to_row_dict(payload, 5)

    assert result is not None
    assert result["token_index"] == 12


def test_token_to_row_dict_falls_back_for_invalid_indexes():
    payload = {"w": "hi", "token_index": "oops"}

    result = _token_to_row_dict(payload, 4)

    assert result is not None
    assert result["token_index"] == 4
    # Upstream payload is not mutated when we correct the index.
    assert payload["token_index"] == "oops"


def test_token_to_row_dict_handles_nan_indexes():
    payload = {"w": "hi", "token_index": math.nan}

    result = _token_to_row_dict(payload, 2)

    assert result is not None
    assert result["token_index"] == 2
    assert math.isnan(payload["token_index"])


def test_token_to_row_dict_returns_copy_that_can_be_mutated_safely():
    payload = {"w": "hi"}

    result = _token_to_row_dict(payload, 1)
    result["w"] = "bye"

    assert payload["w"] == "hi"
