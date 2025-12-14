import json
from pathlib import Path

import pytest

import build_training_pair_standalone as btp


FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Start from the default build_pair configuration to keep alignment settings stable
TEST_SETTINGS = {
    **btp.DEFAULT_SETTINGS.get("build_pair", {}),
    "spacy_enable": False,  # Disable spaCy to speed up tests and avoid model loading
    "time_tolerance_s": 0.5,  # Looser tolerance for test alignment
    "dangling_eos_max_pause_ms": 250,
    "round_seconds": 3,
    "txt_match_close": 0.82,
    "txt_match_weak": 0.65,
    "speaker_correction_window_size": 5,
    "emit_asr_style_training_copy": True,
    "num_regex": r"\\d+[.,]?\\d*",
    "unit_vocab": ["%"],
}


@pytest.fixture(scope="module")
def setup_test_environment():
    """Create a clean output directory for test runs."""
    if TEST_OUTPUT_DIR.exists():
        import shutil

        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    import shutil

    shutil.rmtree(TEST_OUTPUT_DIR)


def test_srt_training_signal_integrity(setup_test_environment):
    """
    Integration test to ensure that training signals from an SRT file
    are correctly processed and translated into features.
    """
    primary_input = FIXTURES_DIR / "test.srt"
    asr_reference = FIXTURES_DIR / "test.asr.json"
    output_basename = "test_srt_integrity_expanded"

    expected_edited_output = TEST_OUTPUT_DIR / f"{output_basename}.train.words.json"
    expected_simulated_output = TEST_OUTPUT_DIR / f"{output_basename}.train.raw.words.json"

    paths = {
        "out_training_dir": TEST_OUTPUT_DIR,
        "out_inference_dir": TEST_OUTPUT_DIR,
    }

    btp.process_file(
        primary_path=primary_input,
        asr_reference_path=asr_reference,
        paths=paths,
        settings=TEST_SETTINGS,
        asr_only_mode=False,
        output_basename=output_basename,
    )

    assert expected_edited_output.exists(), "Edited training file was not created."
    assert expected_simulated_output.exists(), "Simulated ASR training file was not created."

    with open(expected_edited_output, "r", encoding="utf-8") as f:
        edited_data = json.load(f)

    edited_tokens = edited_data.get("tokens", [])
    assert len(edited_tokens) == 44, (
        "Expected 44 tokens in the edited output after alignment, "
        f"but got {len(edited_tokens)}."
    )

    expected_words = [
        "Okej,",
        "det",
        "här",
        "är",
        "ett",
        "lite",
        "längre",
        "test",
        "för",
        "att",
        "se",
        "hur",
        "systemet",
        "hanterar",
        "mer",
        "realistisk",
        "data.",
        "Nu",
        "byter",
        "vi",
        "talare.",
        "Jag",
        "undrar",
        "om",
        "det",
        "här",
        "kommer",
        "att",
        "fungera",
        "som",
        "det",
        "ska.",
        "Det",
        "är",
        "en",
        "bra",
        "fråga.",
        "Absolut,",
        "vi",
        "får",
        "se",
        "vad",
        "som",
        "händer.",
    ]
    actual_words = [t["w"] for t in edited_tokens]
    assert actual_words == expected_words, "The words from the SRT were not correctly aligned."

    # Cue 1: "...för att se\nhur..." -> break on 'se' (index 10)
    # Cue 3: "...att fungera\nsom..." -> break on 'fungera' (index 28)
    # Cue 5: "...får se\nvad..." -> break on the second 'se' (index 40)
    assert edited_tokens[10]["w"] == "se" and edited_tokens[10]["is_llm_structural_break"]
    assert edited_tokens[28]["w"] == "fungera" and edited_tokens[28]["is_llm_structural_break"]
    assert edited_tokens[40]["w"] == "se" and edited_tokens[40]["is_llm_structural_break"]

    assert not edited_tokens[1]["is_llm_structural_break"], (
        "Token 'det' was incorrectly marked with a structural break."
    )
    assert all(t["is_edited_transcript"] for t in edited_tokens), (
        "Not all tokens in the edited output were marked as 'is_edited_transcript': True."
    )

    with open(expected_simulated_output, "r", encoding="utf-8") as f:
        simulated_data = json.load(f)

    simulated_tokens = simulated_data.get("tokens", [])
    assert len(simulated_tokens) == 44, (
        "Expected 44 tokens in the simulated ASR output, "
        f"but got {len(simulated_tokens)}."
    )

    # Verify that the text has been normalized (lowercase, no punctuation)
    assert simulated_tokens[16]["w"] == "data"
    assert simulated_tokens[0]["w"] == "okej"
    assert all(
        not t["is_edited_transcript"] for t in simulated_tokens
    ), "Not all tokens in the simulated output were marked as 'is_edited_transcript': False."

    for i in range(len(edited_tokens)):
        e_tok = edited_tokens[i]
        s_tok = simulated_tokens[i]
        assert e_tok["start"] == s_tok["start"], f"Timestamp mismatch at index {i}"
        assert e_tok["speaker"] == s_tok["speaker"], f"Speaker mismatch at index {i}"
        assert e_tok["break_type"] == s_tok["break_type"], f"Break type label mismatch at index {i}"
        assert e_tok["is_llm_structural_break"] == s_tok["is_llm_structural_break"], (
            f"Structural break hint mismatch at index {i}"
        )


def test_pipeline_robustness_with_noisy_txt(setup_test_environment):
    """
    Tests that the pipeline can gracefully handle a 'noisy' TXT file
    containing mixed languages and unusual characters without crashing.
    """
    primary_input = FIXTURES_DIR / "noisy.txt"
    asr_reference = FIXTURES_DIR / "test.asr.json"
    output_basename = "test_noisy_txt_robustness"
    expected_output = TEST_OUTPUT_DIR / f"{output_basename}.enriched.json"

    paths = {
        "out_training_dir": TEST_OUTPUT_DIR,
        "out_inference_dir": TEST_OUTPUT_DIR,
    }

    btp.process_file(
        primary_path=primary_input,
        asr_reference_path=asr_reference,
        paths=paths,
        settings=TEST_SETTINGS,
        asr_only_mode=False,
        output_basename=output_basename,
    )

    assert expected_output.exists(), "Inference file was not created for the noisy input."

    with open(expected_output, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data.get("tokens", [])) > 0, "The output for the noisy file contains no tokens."
