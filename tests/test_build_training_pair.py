import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import pysrt

from build_training_pair_standalone import (
    _process_srt,
    _process_txt,
    generate_labels_from_cues,
    engineer_features,
    load_asr_words,
    tokenize_srt_cues,
)
import spacy
from spacy.vocab import Vocab


class TestBuildTrainingPair(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)

        # Mock ASR data
        self.mock_asr_data = {
            "words": [
                {"w": "Hello", "start": 0.1, "end": 0.5, "speaker": "A"},
                {"w": "world", "start": 0.6, "end": 1.0, "speaker": "A"},
                {"w": "this", "start": 1.1, "end": 1.4, "speaker": "A"},
                {"w": "is", "start": 1.5, "end": 1.7, "speaker": "A"},
                {"w": "a", "start": 1.8, "end": 1.9, "speaker": "A"},
                {"w": "test", "start": 2.0, "end": 2.4, "speaker": "A"},
            ]
        }
        self.asr_file = self.test_dir / "test.asr.json"
        with open(self.asr_file, "w") as f:
            json.dump(self.mock_asr_data, f)

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

    def test_process_srt(self):
        # Mock SRT file with internal newlines
        srt_content = """1
00:00:00,100 --> 00:00:01,000
Hello
world

2
00:00:01,100 --> 00:00:02,400
this is a
test
"""
        srt_file = self.test_dir / "test.srt"
        with open(srt_file, "w") as f:
            f.write(srt_content)

        # Run the processing
        asr_words = load_asr_words(self.asr_file)
        tokens, cues, alignment_sources, cue_ids = _process_srt(
            primary_path=srt_file,
            asr_words=asr_words,
            settings={},
        )

        # Verify the alignment and structural breaks
        self.assertEqual(len(tokens), 6)
        self.assertEqual(tokens[0]["w"], "Hello")
        self.assertTrue(tokens[0]["is_llm_structural_break"])
        self.assertEqual(tokens[1]["w"], "world")
        self.assertFalse(tokens[1]["is_llm_structural_break"])
        self.assertEqual(tokens[4]["w"], "a")
        self.assertTrue(tokens[4]["is_llm_structural_break"])
        self.assertEqual(tokens[5]["w"], "test")
        self.assertFalse(tokens[5]["is_llm_structural_break"])
        self.assertEqual(len(alignment_sources), len(tokens))
        self.assertEqual(len(cue_ids), len(tokens))

    def test_process_txt(self):
        # Mock TXT file
        txt_content = "Hello world this is a test"
        txt_file = self.test_dir / "test.txt"
        with open(txt_file, "w") as f:
            f.write(txt_content)

        # Run the processing
        asr_words = load_asr_words(self.asr_file)
        tokens, cues, alignment_sources, cue_ids = _process_txt(
            primary_path=txt_file,
            asr_words=asr_words,
            settings={},
        )

        # Verify the alignment
        self.assertEqual(len(tokens), 6)
        self.assertEqual(tokens[0]["w"], "Hello")
        self.assertEqual(tokens[5]["w"], "test")
        self.assertFalse(tokens[1]["is_llm_structural_break"])
        self.assertEqual(len(alignment_sources), len(tokens))
        self.assertEqual(cue_ids, [])

    def test_alignment_based_labeling_resists_timestamp_drift(self):
        cues = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "This is\nA test"},
        ]
        processed_tokens, cue_ids = tokenize_srt_cues(cues)

        tokens = []
        for idx, source_token in enumerate(processed_tokens):
            tokens.append(
                {
                    "w": source_token["w"],
                    "start": 100.0 + idx,
                    "end": 100.5 + idx,
                    "is_llm_structural_break": source_token.get("is_llm_structural_break", False),
                }
            )

        alignment_sources = list(range(len(processed_tokens)))
        settings = {"time_tolerance_s": 0.01}

        generate_labels_from_cues(tokens, cues, settings, alignment_sources, cue_ids)

        cue_id_sequence = [token.get("cue_id") for token in tokens]
        break_type_sequence = [token.get("break_type") for token in tokens]

        self.assertEqual(cue_id_sequence, [0, 0, 1, 1, 1, 1])
        self.assertEqual(
            break_type_sequence,
            ["O", "SB", "O", "LB", "O", "SB"],
        )

    @patch("build_training_pair_standalone.spacy.load")
    def test_engineer_features(self, mock_spacy_load):
        # Mock spacy model and its return values
        mock_nlp = MagicMock()
        mock_nlp.vocab = Vocab()  # Add a real Vocab object to the mock
        mock_doc = MagicMock()
        mock_token = MagicMock()
        mock_token.pos_ = "NN"
        mock_token.lemma_ = "hello"
        mock_token.tag_ = "NN"
        mock_token.morph.to_string.return_value = "Case=Nom"
        mock_token.has_dep.return_value = True
        mock_token.dep_ = "ROOT"
        mock_token.head.i = 0
        mock_doc.__iter__.return_value = [mock_token] * 2
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        # Mock tokens with a speaker change
        tokens = [
            {"w": "Hello", "speaker": "A", "start": 0.1, "end": 0.5},
            {"w": "world", "speaker": "B", "start": 0.8, "end": 1.0}, # Added a pause
        ]

        # Mock settings
        settings = {
            "spacy_enable": True,
            "spacy_model": "sv_core_news_lg",
            "spacy_add_dependencies": True,
        }

        # Apply feature engineering
        engineer_features(tokens, settings)

        # Verify enrichment
        self.assertEqual(len(tokens), 2)
        # Verify spacy features
        self.assertEqual(tokens[0].get("pos"), "NN")
        self.assertEqual(tokens[0].get("lemma"), "hello")
        # Verify guardrail features
        self.assertTrue(tokens[0].get("speaker_change"))


if __name__ == "__main__":
    unittest.main()
