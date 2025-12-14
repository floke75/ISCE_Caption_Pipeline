import shutil
import pytest
import importlib

def check_import(module_name):
    return importlib.import_module(module_name)

def test_core_dependencies():
    """Test that core data processing libraries are installed."""
    check_import("pandas")
    check_import("numpy")
    check_import("rapidfuzz")
    check_import("tqdm")
    check_import("pysrt")
    check_import("yaml")  # pyyaml

def test_nlp_dependencies():
    """Test that SpaCy and the Swedish model are installed and loadable."""
    spacy = check_import("spacy")
    assert spacy.util.is_package("sv_core_news_lg"), "Swedish SpaCy model not installed"
    nlp = spacy.load("sv_core_news_lg")
    assert nlp("Hej världen").has_annotation("TAG"), "SpaCy model should support tagging"

def test_speech_dependencies():
    """Test that speech processing libraries are installed."""
    check_import("torch")
    check_import("whisperx")
    check_import("pyannote.audio")
    check_import("ffmpeg") # ffmpeg-python wrapper

def test_web_dependencies():
    """Test that web backend libraries are installed."""
    check_import("fastapi")
    check_import("pydantic")
    check_import("uvicorn")
    check_import("httpx")

def test_ffmpeg_binary():
    """Test that audio processing capability exists (ffmpeg or native fallback)."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        # Check for fallback dependencies
        if not check_import("torchaudio") or not check_import("soundfile"):
             pytest.fail("Neither ffmpeg binary nor torchaudio/soundfile fallback found. Audio processing will fail.")
        else:
             print("ffmpeg not found, but fallback dependencies are present.")
