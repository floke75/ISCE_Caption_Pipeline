import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import importlib

# Add repo root to path to import align_make
sys.path.append(".")
import align_make

def create_dummy_wav(filename, duration=0.1, rate=44100):
    t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    sf.write(filename, audio_int16, rate)

@pytest.fixture
def dummy_wav(tmp_path):
    p = tmp_path / "test_input.wav"
    create_dummy_wav(str(p))
    return p

def test_load_audio_native(dummy_wav):
    """Test that load_audio_native loads and resamples correctly."""
    audio = align_make.load_audio_native(str(dummy_wav))
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (16000 * 0.1,) # 0.1s duration
    assert audio.dtype == np.float32

@patch("align_make.ffmpeg")
def test_extract_and_convert_audio_fallback(mock_ffmpeg, dummy_wav, tmp_path):
    """Test that extract_and_convert_audio falls back to native loader if ffmpeg fails."""
    # Ensure ffmpeg.Error is a class we can catch
    mock_ffmpeg.Error = Exception

    # Simulate ffmpeg not found or failing.
    # We raise FileNotFoundError when .run() is called, or earlier.
    # The code catches (ffmpeg.Error, FileNotFoundError).
    # Since we are mocking the module, we need to ensure the catch block works.
    mock_ffmpeg.input.side_effect = FileNotFoundError

    output = align_make.extract_and_convert_audio(dummy_wav, tmp_path)

    assert output.exists()
    assert output.name == "test_input_16khz_mono.wav"

    # Verify content
    data, rate = sf.read(str(output))
    assert rate == 16000
    assert len(data) == 16000 * 0.1

@patch("align_make._load_dependency")
def test_process_file_uses_native_loader(mock_load_dep, dummy_wav, tmp_path):
    """Test that process_file uses the native loader and passes array to whisperx."""

    # Mock whisperx module and torch
    mock_whisperx = MagicMock()
    # Configure mock return values to satisfy unpacking and dictionary access
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())

    # Mock transcription result
    mock_model = mock_whisperx.load_model.return_value
    mock_model.transcribe.return_value = {"language": "en", "segments": []}

    # Mock alignment result
    mock_whisperx.align.return_value = {"language": "en", "segments": [], "word_segments": []}

    mock_torch = MagicMock()

    # We want to use real torchaudio for the spy, but mock others
    real_torchaudio = importlib.import_module("torchaudio")

    def side_effect(name, reason):
        if name == "whisperx": return mock_whisperx
        if name == "torch": return mock_torch
        if name == "torchaudio": return real_torchaudio
        return MagicMock()

    mock_load_dep.side_effect = side_effect

    # Mock settings and paths
    paths = {
        "asr_dir": tmp_path / "asr",
        "align_dir": tmp_path / "align"
    }
    # Create the directories as process_file expects them to exist or creates subdirs inside them
    paths["asr_dir"].mkdir(parents=True, exist_ok=True)
    paths["align_dir"].mkdir(parents=True, exist_ok=True)

    settings = {
        "whisper_model_id": "tiny",
        "align_model_id": "dummy_align",
        "batch_size": 1,
        "compute_type": "float32",
        "hf_token": "dummy",
        "do_diarization": False
    }

    # Mock extract_and_convert_audio to return our dummy wav (or a converted one)
    # We can rely on the real one or mock it. Let's mock it to keep unit test isolated.
    with patch("align_make.extract_and_convert_audio") as mock_extract:
        mock_extract.return_value = dummy_wav

        # We also need to mock load_audio_native to verify it is called,
        # OR we rely on the real one and check what is passed to transcribe.
        # Let's spy on load_audio_native
        with patch("align_make.load_audio_native", wraps=align_make.load_audio_native) as spy_load:

            align_make.process_file(dummy_wav, "cpu", paths, settings)

            # Verify extract_and_convert_audio was called
            mock_extract.assert_called_once()

            # Verify native loader was called
            spy_load.assert_called_once_with(str(dummy_wav))

            # Verify whisperx.load_audio was NOT called
            mock_whisperx.load_audio.assert_not_called()

            # Verify model.transcribe was called with the audio array
            # Get the model mock returned by load_model
            mock_model = mock_whisperx.load_model.return_value
            mock_model.transcribe.assert_called_once()
            args, kwargs = mock_model.transcribe.call_args
            audio_arg = args[0]
            assert isinstance(audio_arg, np.ndarray)
            assert audio_arg.shape == (1600, ) # 0.1s * 16000
