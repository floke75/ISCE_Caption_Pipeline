from isce.config import Config
from isce.model_builder import create_feature_row
from isce.types import TokenRow


def make_cfg() -> Config:
    return Config(
        beam_width=1,
        min_block_duration_s=0.5,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
            "block": {"min_total_chars": 0, "min_last_line_chars": 0},
        },
        min_chars_for_single_word_block=10,
        sliders={},
        paths={},
    )


def test_create_feature_row_sanitises_payload_and_defaults_outcome() -> None:
    row = TokenRow(
        token={
            "w": "Hello",
            "pause_z": None,
            "relative_position": None,
            "pos": "NOUN",
            "lemma": 123,
            "tag": None,
            "morph": "Number=Sing|Case=Nom",
            "dep": None,
            "head_idx": "4",
            "token_index": "3",
            "num_unit_glue": 0,
            "is_dangling_eos": False,
            "speaker_change": True,
            "starts_with_dialogue_dash": False,
            "break_type": None,
        },
        nxt={"lemma": None, "tag": None, "morph": None, "dep": None, "token_index": "4", "pos": "VERB"},
    )

    features = create_feature_row(row, make_cfg())

    assert features["outcome"] == "O"
    assert features["lemma_bigram"] == "lb:123|none"
    assert features["tag_bigram"] == "tb:none|none"
    assert features["morph_bigram"] == "mb:case=nom+number=sing|none"
    assert features["dep_bigram"] == "db:none|none"
    assert features["head_position"] == "head_pos:next"
    assert features["head_link"] == "dep_link:token->next"
    assert features["interact_punct_pause"].startswith("pp:")
    assert features["interact_punct_syntax"].startswith("ps:")
