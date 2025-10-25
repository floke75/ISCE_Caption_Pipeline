from __future__ import annotations

import json
from pathlib import Path

from isce.model_builder import derive_constraints
from isce.config import Config


def make_config() -> Config:
    return Config(
        beam_width=7,
        min_block_duration_s=1.0,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
        },
        min_chars_for_single_word_block=10,
        sliders={},
        paths={},
        enable_bidirectional_pass=False,
        lookahead_width=0,
        enable_reflow=False,
        min_line_length_for_break=1,
        min_last_word_len_for_break=1,
        single_word_line_penalty=0.0,
        extreme_balance_penalty=0.0,
        enable_refinement_pass=False,
        min_block_length_char=1,
        min_line_length_char=1,
    )


def write_corpus_file(path: Path, tokens: list[dict]) -> None:
    path.write_text(json.dumps({"tokens": tokens}), encoding="utf-8")


def test_derive_constraints_skips_simulated_blocks(tmp_path):
    cfg = make_config()

    edited_tokens = [
        {
            "w": "Hello,",
            "start": 0.0,
            "end": 0.3,
            "break_type": "LB",
            "pause_after_ms": 120,
            "is_edited_transcript": True,
        },
        {
            "w": "world",
            "start": 0.3,
            "end": 0.6,
            "break_type": "O",
            "pause_after_ms": 80,
            "is_edited_transcript": True,
        },
        {
            "w": "again!",
            "start": 0.6,
            "end": 1.1,
            "break_type": "SB",
            "pause_after_ms": 0,
            "is_edited_transcript": True,
        },
    ]

    simulated_tokens = [dict(token, is_edited_transcript=False) for token in edited_tokens]

    edited_path = tmp_path / "edited.json"
    raw_path = tmp_path / "raw_copy.train.raw.words.json"
    write_corpus_file(edited_path, edited_tokens)
    write_corpus_file(raw_path, simulated_tokens)

    constraints_from_edited = derive_constraints([str(edited_path)], cfg)
    constraints_with_raw = derive_constraints([str(edited_path), str(raw_path)], cfg)

    assert constraints_with_raw["line1"]["soft_target"] == constraints_from_edited["line1"]["soft_target"]
    assert constraints_with_raw["line2"]["soft_target"] == constraints_from_edited["line2"]["soft_target"]
    assert constraints_with_raw["ideal_cps_median"] == constraints_from_edited["ideal_cps_median"]
