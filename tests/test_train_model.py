import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isce.config import Config
from isce.model_builder import derive_constraints
from isce.scorer import Scorer
from isce.types import Engineered, TokenRow
from scripts.train_model import (
    partition_corpus_paths,
    get_full_feature_table_and_rows,
    sanitize_row_for_reweighting,
)


def _write_tokens(path: Path, tokens: list[dict]) -> None:
    path.write_text(json.dumps({"tokens": tokens}), encoding="utf-8")


def _fallback_config() -> Config:
    return Config(
        beam_width=7,
        min_block_duration_s=0.5,
        max_block_duration_s=10.0,
        line_length_constraints={"line1": {"soft_target": 37, "hard_limit": 42}, "line2": {"soft_target": 37, "hard_limit": 42}},
        min_chars_for_single_word_block=4,
        sliders={},
        paths={},
        lookahead_width=0,
        allowed_single_word_proper_nouns=(),
    )


def test_partition_corpus_paths_identifies_raw(tmp_path: Path) -> None:
    human_file = tmp_path / "clip.train.words.json"
    raw_file = tmp_path / "clip.train.raw.words.json"
    other_file = tmp_path / "notes.json"

    for path in (human_file, raw_file, other_file):
        path.write_text("{}", encoding="utf-8")

    human_paths, raw_paths = partition_corpus_paths(tmp_path)

    assert human_file in human_paths
    assert raw_file in raw_paths
    assert other_file not in human_paths + raw_paths


def test_constraints_ignore_raw_duplicates(tmp_path: Path) -> None:
    human_file = tmp_path / "episode.train.words.json"
    raw_file = tmp_path / "episode.train.raw.words.json"

    _write_tokens(
        human_file,
        [
            {"w": "Hello", "start": 0.0, "end": 0.5, "break_type": "O", "pause_after_ms": 0},
            {"w": "world", "start": 0.5, "end": 1.0, "break_type": "SB", "pause_after_ms": 0},
        ],
    )

    _write_tokens(
        raw_file,
        [
            {"w": "hello", "start": 0.0, "end": 0.2, "break_type": "O", "pause_after_ms": 0},
            {"w": "world", "start": 0.2, "end": 0.4, "break_type": "SB", "pause_after_ms": 0},
        ],
    )

    cfg = _fallback_config()

    expected = derive_constraints([str(human_file)], cfg)

    human_paths, raw_paths = partition_corpus_paths(tmp_path)
    assert raw_paths, "Raw duplicate should be detected for regression coverage."

    filtered_constraints = derive_constraints([str(p) for p in human_paths], cfg)
    assert filtered_constraints["ideal_cps_median"] == pytest.approx(expected["ideal_cps_median"])
    assert filtered_constraints["min_block_duration_s"] == pytest.approx(expected["min_block_duration_s"])

    polluted_constraints = derive_constraints([str(p) for p in human_paths + raw_paths], cfg)
    assert polluted_constraints["ideal_cps_median"] != pytest.approx(expected["ideal_cps_median"])


def test_get_full_feature_table_includes_spacy_features(tmp_path: Path) -> None:
    cfg = _fallback_config()

    rich_tokens = [
        {
            "w": "Running",
            "start": 0.0,
            "end": 0.5,
            "break_type": "O",
            "pause_after_ms": 120,
            "lemma": "run",
            "tag": "VBG",
            "morph": "Tense=Pres|VerbForm=Part",
            "dep": "advcl",
            "head_idx": 1,
        },
        {
            "w": "fast",
            "start": 0.5,
            "end": 0.9,
            "break_type": "LB",
            "pause_after_ms": 30,
            "lemma": "fast",
            "tag": "RB",
            "morph": "Degree=Pos",
            "dep": "advmod",
            "head_idx": 0,
        },
    ]

    sparse_tokens = [
        {
            "w": "We",
            "start": 1.0,
            "end": 1.2,
            "break_type": "O",
            "pause_after_ms": 40,
        },
        {
            "w": "go",
            "start": 1.2,
            "end": 1.5,
            "break_type": "SB",
            "pause_after_ms": 0,
        },
    ]

    rich_path = tmp_path / "rich.train.words.json"
    sparse_path = tmp_path / "sparse.train.words.json"
    _write_tokens(rich_path, rich_tokens)
    _write_tokens(sparse_path, sparse_tokens)

    df, token_rows = get_full_feature_table_and_rows([
        str(rich_path),
        str(sparse_path),
    ], cfg)

    assert len(token_rows) == 2
    assert {"lemma_bigram", "tag_bigram", "morph_bigram", "dep_bigram", "head_position", "head_link"}.issubset(df.columns)

    rich_row = df.iloc[0]
    assert rich_row["lemma_bigram"] == "lb:run|fast"
    assert rich_row["tag_bigram"] == "tb:vbg|rb"
    assert rich_row["morph_bigram"] == "mb:tense=pres+verbform=part|degree=pos"
    assert rich_row["dep_bigram"] == "db:advcl|advmod"
    assert rich_row["head_position"] == "head_pos:next"
    assert rich_row["head_link"] == "dep_link:token->next"

    sparse_row = df.iloc[1]
    assert sparse_row["lemma_bigram"] == "lb:none|none"
    assert sparse_row["tag_bigram"] == "tb:none|none"
    assert sparse_row["morph_bigram"] == "mb:none|none"
    assert sparse_row["dep_bigram"] == "db:none|none"
    assert sparse_row["head_position"] == "head_pos:unknown"
    assert sparse_row["head_link"] == "dep_link:none"


def _reweighting_config() -> Config:
    return Config(
        beam_width=1,
        min_block_duration_s=0.0,
        max_block_duration_s=10.0,
        line_length_constraints={
            "line": {"soft": 42, "hard": 45},
            "block": {"soft": 84, "hard": 90},
        },
        min_chars_for_single_word_block=0,
        sliders={},
        paths={},
        enable_bidirectional_pass=False,
        lookahead_width=0,
        enable_reflow=False,
        allowed_single_word_proper_nouns=(),
        enable_refinement_pass=False,
    )


def _base_token(**overrides):
    token = {
        "w": "Hello",
        "pause_z": 0.0,
        "relative_position": 0.5,
        "pos": "INTJ",
        "lemma": "hello",
        "tag": "UH",
        "morph": "",
        "dep": "ROOT",
        "head_idx": 0,
        "num_unit_glue": False,
        "is_dangling_eos": False,
        "punct_after": None,
        "speaker_change": False,
        "starts_with_dialogue_dash": False,
        "is_sentence_initial": False,
    }
    token.update(overrides)
    return token


def test_reweighting_sanitizes_structural_hint():
    token = _base_token(is_llm_structural_break=True)
    nxt = _base_token(w="world", lemma="world", is_llm_structural_break=True)
    lookahead_token = _base_token(w="friend", lemma="friend", is_llm_structural_break=True)

    row = TokenRow(
        token=token,
        nxt=nxt,
        feats=Engineered(),
        lookahead=(lookahead_token,),
    )

    sanitized = sanitize_row_for_reweighting(row)

    # Ensure the sanitized payloads are clones with the hint disabled.
    assert sanitized.token is not row.token
    assert sanitized.nxt is not row.nxt
    assert sanitized.lookahead is not None
    assert sanitized.lookahead[0] is not lookahead_token
    assert row.token["is_llm_structural_break"] is True
    assert sanitized.token["is_llm_structural_break"] is False
    assert sanitized.nxt["is_llm_structural_break"] is False
    assert sanitized.lookahead[0]["is_llm_structural_break"] is False

    cfg = _reweighting_config()
    scorer = Scorer(weights={}, constraints={}, sliders={}, cfg=cfg)

    boosted_scores = scorer.score_transition(row)
    sanitized_scores = scorer.score_transition(sanitized)

    boosted_delta = boosted_scores["SB"] - boosted_scores["O"]
    sanitized_delta = sanitized_scores["SB"] - sanitized_scores["O"]

    # Removing the hint should drop the SB-vs-O spread by twice the structure boost.
    assert boosted_delta - sanitized_delta == pytest.approx(2 * scorer.structure_boost)
