import json
import random
from pathlib import Path

def generate_mock_data():
    # Define a sample text structure
    # Sentence 1: "Hello world, this is a test." (Split across 2 lines)
    # Sentence 2: "We are verifying the inference alignment." (One line)
    # Sentence 3: "It should handle newlines correctly." (One line)

    text_structure = [
        # Line 1
        [
            ("Hello", True), ("world,", False), ("this", False), ("is", False), ("a", False), ("test.", True)
        ],
        # Line 2 (Sentence 1 continued? No, "test." ended it. New sentence "We...")
        [
            ("We", True), ("are", False), ("verifying", False), ("the", False), ("inference", False), ("alignment.", True)
        ],
        # Line 3
        [
            ("It", True), ("should", False), ("handle", False), ("newlines", False), ("correctly.", True)
        ]
    ]

    # Wait, structural breaks usually happen at line ends.
    # Let's define it as: List of Lines. Each Line has Words.
    # Last word of line has is_llm_structural_break = True.

    lines = [
        ["Hello", "world,", "this", "is", "a", "test."],
        ["We", "are", "verifying", "the"],
        ["inference", "alignment."],
        ["It", "should", "handle", "newlines", "correctly."]
    ]

    tokens = []
    asr_words = []

    current_time = 1.0

    token_idx = 0

    for line_idx, line in enumerate(lines):
        for word_idx, word_text in enumerate(line):
            duration = random.uniform(0.3, 0.6)
            start = current_time
            end = current_time + duration
            current_time = end + random.uniform(0.05, 0.2) # Small gap

            # Token
            is_sentence_final = word_text.endswith(".")
            is_sentence_initial = token_idx == 0 or (tokens[-1]["is_sentence_final"] if tokens else True)

            # Structural break on last word of line
            is_llm_structural_break = (word_idx == len(line) - 1)

            tokens.append({
                "w": word_text,
                "start": round(start, 3),
                "end": round(end, 3),
                "is_llm_structural_break": is_llm_structural_break,
                "is_sentence_final": is_sentence_final,
                "is_sentence_initial": is_sentence_initial,
                "speaker": "SPEAKER_00",
                "token_index": token_idx
            })

            # ASR Word (Similar but simpler)
            clean_word = word_text.replace(".", "").replace(",", "")
            asr_words.append({
                "w": clean_word,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": "SPEAKER_00",
                "score": random.uniform(0.7, 0.99)
            })

            token_idx += 1

        current_time += 0.5 # Pause between lines

    # Introduce a slight drift or mismatch in ASR to make it realistic
    # Add a hallucinated word in ASR
    asr_words.insert(3, {
        "w": "um",
        "start": asr_words[2]["end"] + 0.1,
        "end": asr_words[3]["start"] - 0.1,
        "speaker": "SPEAKER_00",
        "score": 0.4
    })

    # Save files
    out_dir = Path("ui_data/mocks")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "mock.enriched.json").write_text(json.dumps({"tokens": tokens}, indent=2), encoding="utf-8")
    (out_dir / "mock.asr.visual.words.diar.json").write_text(json.dumps({"words": asr_words}, indent=2), encoding="utf-8")

    print("Mock data generated in ui_data/mocks/")

if __name__ == "__main__":
    generate_mock_data()
