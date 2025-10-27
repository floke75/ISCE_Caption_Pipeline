import pytest

from isce.token_utils import normalize_token_payload
from isce.types import Token


def test_normalize_token_payload_sanitizes_mixed_types() -> None:
    token_dict = {
        "w": None,
        "pause_z": "1.5",
        "pause_before_ms": "250",
        "pause_after_ms": None,
        "speaker_change": "true",
        "starts_with_dialogue_dash": "False",
        "num_unit_glue": "0",
        "is_llm_structural_break": "",
        "is_dangling_eos": "no",
        "head_idx": "10",
    }

    normalized = normalize_token_payload(token_dict, idx=7)

    assert normalized["w"] == ""
    assert normalized["pause_z"] == pytest.approx(1.5)
    assert normalized["pause_before_ms"] == 250
    assert normalized["pause_after_ms"] == 0
    assert normalized["speaker_change"] is True
    assert normalized["starts_with_dialogue_dash"] is False
    assert normalized["num_unit_glue"] is False
    assert normalized["is_llm_structural_break"] is False
    assert normalized["is_dangling_eos"] is False
    assert normalized["token_index"] == 7
    assert normalized["head_idx"] == 10

    # Original dictionary should remain untouched (deep copy)
    assert token_dict["w"] is None
    assert token_dict["pause_before_ms"] == "250"
    assert "token_index" not in token_dict


def test_normalize_token_payload_from_dataclass_preserves_original() -> None:
    token = Token(w="hello", start=0.0, end=0.5, speaker="A")

    normalized = normalize_token_payload(token, idx=3)

    assert normalized["token_index"] == 3
    assert normalized["w"] == "hello"
    # Dataclass instance remains unchanged
    assert token.token_index is None
    assert token.w == "hello"
