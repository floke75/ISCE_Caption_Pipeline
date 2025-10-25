import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_training_pair_standalone import engineer_features, serialize_token

def approx_equal(a, b, rel_tol=1e-9, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def make_token(word: str, start: float, end: float) -> dict:
    return {"w": word, "start": start, "end": end, "speaker": "A"}


def test_engineer_features_sets_relative_position_and_dangling_flags():
    tokens = [
        make_token("Hello", 0.0, 0.3),
        make_token("world.", 0.3, 0.6),
        make_token("Next", 0.8, 0.95),
        make_token("sentence", 0.95, 1.1),
        make_token("again!", 1.1, 1.3),
        make_token("Okay.", 1.3, 1.5),
        make_token("but", 1.52, 1.65),
    ]

    settings = {"spacy_enable": False, "dangling_eos_max_pause_ms": 250, "round_seconds": 3}
    engineer_features(tokens, settings)

    relative_positions = [tokens[i]["relative_position"] for i in range(len(tokens))]
    assert approx_equal(relative_positions[0], 0.0)
    assert approx_equal(relative_positions[1], 1.0)
    assert approx_equal(relative_positions[3], 0.5)
    assert approx_equal(relative_positions[6], 1.0)

    assert tokens[1]["is_dangling_eos"] is False
    assert tokens[5]["is_dangling_eos"] is True

    serialized = [serialize_token(t, settings) for t in tokens]
    assert approx_equal(serialized[3]["relative_position"], 0.5)
    assert serialized[5]["is_dangling_eos"] is True
