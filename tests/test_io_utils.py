import json
from pathlib import Path

import pytest

from isce.io_utils import load_tokens, save_tokens
from isce.types import Token


def test_load_tokens_filters_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps(
            {
                "tokens": [
                    {
                        "w": "Hello",
                        "start": 0.0,
                        "end": 0.5,
                        "speaker": None,
                        "extra_field": "ignored",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tokens = load_tokens(str(path))

    assert len(tokens) == 1
    assert tokens[0].w == "Hello"
    assert not hasattr(tokens[0], "extra_field")


def test_load_tokens_reports_structure_errors(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"not_tokens": []}), encoding="utf-8")

    with pytest.raises(TypeError):
        load_tokens(str(bad_path))


def test_load_tokens_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tokens(str(tmp_path / "missing.json"))


def test_save_tokens_writes_expected_structure(tmp_path: Path) -> None:
    out_path = tmp_path / "out.json"
    tokens = [Token("Hi", 0.0, 0.5, None)]

    save_tokens(str(out_path), tokens)

    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert data["tokens"][0]["w"] == "Hi"
    assert data["tokens"][0]["start"] == 0.0
