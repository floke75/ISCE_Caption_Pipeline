from isce.token_normalization import normalize_token_payload
from isce.types import Token


def test_normalizes_dict_payload_with_string_index_and_numeric_word():
    payload = {"w": 123, "token_index": "4", "start": 0.0, "end": 1.0, "speaker": None}

    result = normalize_token_payload(payload)

    assert result == {"w": "123", "token_index": 4, "start": 0.0, "end": 1.0, "speaker": None}


def test_falls_back_to_idx_when_token_index_missing_or_invalid():
    token = Token(w="hello", start=0.0, end=1.0, speaker=None, token_index=None)
    invalid = {"w": "world", "start": 1.0, "end": 2.0, "speaker": None, "token_index": "abc"}

    result_token = normalize_token_payload(token, idx=5)
    result_invalid = normalize_token_payload(invalid, idx=6)

    assert result_token["token_index"] == 5
    assert result_invalid["token_index"] == 6


def test_none_payload_passthrough():
    assert normalize_token_payload(None) is None
