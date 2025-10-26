import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import pysrt

from build_training_pair_standalone import (
    align_text_to_asr,
    _apply_spacy,
    _apply_guardrails,
    _load_asr_json,
    _process_srt,
    _process_txt,
)
from isce.types import TokenRow


class TestBuildTrainingPair(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)

        # Mock ASR data
        self.mock_asr_data = [
            {"word": "Hello", "start": 0.1, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0},
            {"word": "this", "start": 1.1, "end": 1.4},
            {"word": "is", "start": 1.5, "end": 1.7},
            {"word": "a", "start": 1.8, "end": 1.9},
            {"word": "test", "start": 2.0, "end": 2.4},
        ]
        self.asr_file = self.test_dir / "test.asr.json"
        with open(self.asr_file, "w") as f:
            json.dump(self.mock_asr_data, f)

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

    def test_align_text_to_asr_with_srt(self):
        # Mock SRT file
        srt_content = """1
00:00:00,100 --> 00:00:01,000
Hello world

2
00:00:01,100 --> 00:00:02,400
this is a test
"""
        srt_file = self.test_dir / "test.srt"
        with open(srt_file, "w") as f:
            f.write(srt_content)

        # Run the alignment
        aligned_tokens = align_text_to_asr(
            media_file=MagicMock(),
            transcript_file=srt_file,
            asr_file=self.asr_file,
            output_file=MagicMock(),
            is_training_data=False,
        )

        # Verify the alignment
        self.assertEqual(len(aligned_tokens), 6)
        self.assertEqual(aligned_tokens[0].token.w, "Hello")
        self.assertEqual(aligned_tokens[1].token.w, "world")
        self.assertTrue(aligned_tokens[1].token.is_llm_structural_break)
        self.assertEqual(aligned_tokens[2].token.w, "this")
        self.assertEqual(aligned_tokens[5].token.w, "test")
        self.assertTrue(aligned_tokens[5].token.is_llm_structural_break)

    def test_align_text_to_asr_with_txt(self):
        # Mock TXT file
        txt_content = "Hello world this is a test"
        txt_file = self.test_dir / "test.txt"
        with open(txt_file, "w") as f:
            f.write(txt_content)

        # Run the alignment
        aligned_tokens = align_text_to_asr(
            media_file=MagicMock(),
            transcript_file=txt_file,
            asr_file=self.asr_file,
            output_file=MagicMock(),
            is_training_data=False,
        )

        # Verify the alignment
        self.assertEqual(len(aligned_tokens), 6)
        self.assertEqual(aligned_tokens[0].token.w, "Hello")
        self.assertEqual(aligned_tokens[5].token.w, "test")
        # In TXT processing, structural breaks are not inferred
        self.assertFalse(hasattr(aligned_tokens[1].token, 'is_llm_structural_break') and aligned_tokens[1].token.is_llm_structural_break)

    @patch("build_training_pair_standalone.spacy.load")
    def test_apply_spacy(self, mock_spacy_load):
        # Mock spacy model
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_token = MagicMock()
        mock_token.pos_ = "NN"
        mock_token.is_sent_start = True
        mock_doc.__iter__.return_value = [mock_token] * 2
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp

        # Mock tokens
        tokens = [
            TokenRow(token={"w": "Hello"}, engineered={}),
            TokenRow(token={"w": "world"}, engineered={}),
        ]

        # Apply spacy
        enriched_tokens = _apply_spacy(tokens)

        # Verify enrichment
        self.assertEqual(len(enriched_tokens), 2)
        self.assertEqual(enriched_tokens[0].token.pos, "NN")
        self.assertTrue(enriched_tokens[0].token.is_sentence_initial)

    def test_apply_guardrails(self):
        # Mock tokens with a speaker change
        tokens = [
            TokenRow(token={"w": "Hello", "speaker": "A"}, engineered={}),
            TokenRow(token={"w": "world", "speaker": "B"}, engineered={}),
        ]

        # Apply guardrails
        enriched_tokens = _apply_guardrails(tokens)

        # Verify enrichment
        self.assertEqual(len(enriched_tokens), 2)
        self.assertTrue(enriched_tokens[1].token.speaker_change)


if __name__ == "__main__":
    unittest.main()
