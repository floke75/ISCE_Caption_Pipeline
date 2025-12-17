#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained audio processing and speech recognition engine.

This script is responsible for the first major stage of the ISCE pipeline. It
takes a single audio or video file as input and performs the following steps:
1.  Extracts the audio stream and converts it to a standardized format (16kHz
    mono WAV) required by the speech recognition models.
2.  Uses a WhisperX model to transcribe the audio into text.
3.  Performs forced alignment to obtain precise word-level timestamps.
4.  Applies speaker diarization to identify and label different speakers.

The final output is a single JSON file containing a flat list of words, each
with its start time, end time, and assigned speaker label. This file serves as
the foundational timing reference for all subsequent steps in the main ISCE
pipeline.

This script is designed to be executable as a standalone command-line tool and
is called by the main orchestrator (`run_pipeline.py`).

Attributes:
    DEFAULT_SETTINGS (Dict[str, Any]): A dictionary holding the default
        configuration for the script, which can be overridden by an external
        YAML file. This includes model identifiers, language settings, and paths.
"""
import os
import sys
import json
import traceback
import gc
from pathlib import Path
from typing import Any, Dict
import argparse
import importlib
import importlib.util
import ffmpeg
import warnings
import numpy as np
from pipeline_config import load_pipeline_config

# =========================
# DEFAULT SETTINGS (Self-Contained)
# =========================
DEFAULT_SETTINGS: Dict[str, Any] = {
    "project_root": ".",
    "pipeline_root": "{project_root}/pipeline_data",
    "align_make": {
        "out_root":     "{pipeline_root}/_intermediate",
        "cache_dir":    "{project_root}/cache",
        "whisper_model_id": "KBLab/kb-whisper-large",
        "align_model_id": "KBLab/wav2vec2-large-voxrex-swedish",
        "language": "sv",
        "compute_type": "float16",
        "batch_size": 16,
        "hf_token": "",
        "do_diarization": True,
        "diar_min_spk": None,
        "diar_max_spk": None,
        "skip_if_asr_exists": False,
    }
}

# --- QUIET noisy 3rd-party warnings ---
warnings.filterwarnings("ignore", message=r".*TorchCodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*torchaudio._backend.list_audio_backends has been deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*torchaudio.load_with_torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*torio.io._streaming_media_decoder.StreamingMediaDecoder.*", category=UserWarning)

# =========================
# Utilities
# =========================
def ensure_dirs(p: Path):
    """
    Ensures that the directory for a given path exists.

    Args:
        p: A Path object representing the file or directory.
    """
    p.mkdir(parents=True, exist_ok=True)


def _resource_error(stage: str, exc: Exception) -> RuntimeError:
    """Create a descriptive error when WhisperX resources are missing."""

    hint = (
        f"Failed to {stage}. WhisperX model assets may not be installed yet. "
        "Run `python scripts/install.py --skip-frontend` (see README) to pre-download "
        "required resources before retrying."
    )
    return RuntimeError(f"{hint}\nOriginal error: {exc}")


def _load_dependency(module_name: str, stage: str):
    """Import a heavy dependency lazily with a descriptive error if missing."""

    if importlib.util.find_spec(module_name) is None:
        raise _resource_error(stage, ModuleNotFoundError(f"No module named '{module_name}'"))
    return importlib.import_module(module_name)

def set_env_tokens(token: str):
    """
    Sets the Hugging Face authentication token as an environment variable.

    This is used by the diarization model to download necessary resources.

    Args:
        token: The Hugging Face API token.
    """
    if token:
        os.environ["HF_TOKEN"] = token
    else:
        print("[WARN] Hugging Face token is not set. Diarization may fail.")

def pick_device(device_cfg: str = "auto") -> str:
    """
    Selects the optimal computation device based on availability and configuration.

    It prioritizes CUDA if available and not explicitly disabled.

    Args:
        device_cfg: The desired device ('auto', 'cuda', or 'cpu').

    Returns:
        A string representing the selected device, either "cuda" or "cpu".
    """
    torch = _load_dependency("torch", "select computation device")

    if device_cfg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA specified but not available. Falling back to CPU.")
        return "cpu"
    if device_cfg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def base_of(path: Path) -> str:
    """
    Gets the base name of a file path, excluding the extension.

    Args:
        path: A Path object.

    Returns:
        The filename stem.
    """
    return path.stem

def _save_json(obj: dict, p: Path):
    """
    Saves a dictionary to a JSON file.

    Ensures the parent directory exists and writes the file with UTF-8 encoding
    and human-readable indentation.

    Args:
        obj: The dictionary to save.
        p: The destination Path object.
    """
    ensure_dirs(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_audio_native(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Loads an audio file into a 16kHz mono numpy array using torchaudio.

    This serves as a fallback when ffmpeg is not available or when we want
    to bypass subprocess calls. It mimics the output format of whisperx.load_audio.

    Args:
        file_path: Path to the input audio file.
        target_sr: Target sample rate (default: 16000).

    Returns:
        A numpy array containing the audio samples (float32).
    """
    torchaudio = _load_dependency("torchaudio", "load audio natively")
    torch = _load_dependency("torch", "load audio natively")

    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with torchaudio: {e}")

    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    # Mix to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Flatten to 1D array
    return waveform.squeeze().numpy()


def run_mock_mode(input_file: Path, mock_asr_json: Path, align_dir: Path):
    """Write a mocked ASR+diarization output for testing without heavy deps."""

    if not mock_asr_json.exists():
        raise FileNotFoundError(f"Mock ASR JSON not found: {mock_asr_json}")

    base = base_of(input_file)
    destination = align_dir / f"{base}.asr.visual.words.diar.json"
    print(f"[MOCK] Writing mock ASR output to: {destination}")
    payload = json.loads(mock_asr_json.read_text(encoding="utf-8"))
    _save_json(payload, destination)

# =========================
# Audio Processing
# =========================
def extract_and_convert_audio(video_path: Path, temp_dir: Path) -> Path:
    """
    Extracts audio from a media file and converts it to a standardized format.

    Uses ffmpeg to convert the audio from any supported video or audio file
    into a 16kHz mono WAV file. If ffmpeg is missing, falls back to a native
    Python implementation using torchaudio/soundfile (supports WAV/FLAC/MP3).

    Args:
        video_path: Path to the input media file.
        temp_dir: Directory to store the temporary WAV file.

    Returns:
        The path to the newly created WAV file.
    """
    output_wav_path = temp_dir / f"{video_path.stem}_16khz_mono.wav"
    print(f"[AUDIO] Extracting and converting audio to: {output_wav_path.name}")

    # Try ffmpeg first
    ffmpeg_success = False
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_wav_path), acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
        )
        ffmpeg_success = True
    except (ffmpeg.Error, FileNotFoundError):
        print("[WARN] ffmpeg failed or not found. Attempting fallback with torchaudio/soundfile...")

    if ffmpeg_success:
        if not output_wav_path.exists():
            raise IOError(f"ffmpeg ran but the output file was not created: {output_wav_path}")
        return output_wav_path

    # Fallback path
    try:
        sf = _load_dependency("soundfile", "save converted audio")
        audio_data = load_audio_native(str(video_path))
        sf.write(str(output_wav_path), audio_data, 16000)
        print(f"[FALLBACK] Successfully converted audio using torchaudio/soundfile.")
    except Exception as e:
        raise RuntimeError(f"Fallback audio conversion failed: {e}. Please install ffmpeg.") from e

    return output_wav_path

# =========================
# Main per-file pipeline
# =========================
def process_file(audio_path: Path, device: str, paths: Dict[str, Path], settings: Dict[str, Any]):
    """
    Runs the complete ASR and diarization pipeline for a single audio file.

    This function performs the following steps:
    1.  Extracts and converts the audio to a standard format using `extract_and_convert_audio`.
    2.  Transcribes the audio to text using a WhisperX model.
    3.  Aligns the transcription to get precise word-level timestamps.
    4.  Performs speaker diarization to assign a speaker label to each word.
    5.  Saves the final, flattened list of word objects to a JSON file.

    It manages GPU memory by loading and unloading models for each step.

    Args:
        audio_path: The path to the input media file.
        device: The computation device to use ("cuda" or "cpu").
        paths: A dictionary containing output directory paths.
        settings: A dictionary of operational settings for the pipeline.
    """
    base = base_of(audio_path)
    print(f"\n===== [{base}] =====")

    asr_raw_json = paths["asr_dir"] / f"{base}.asr.json"
    asr_final_json = paths["align_dir"] / f"{base}.asr.visual.words.diar.json"

    temp_audio_dir = paths["asr_dir"] / "_temp_audio"
    temp_audio_dir.mkdir(exist_ok=True)
    converted_audio_path = None

    try:
        converted_audio_path = extract_and_convert_audio(audio_path, temp_audio_dir)

        if settings.get("skip_if_asr_exists") and asr_final_json.exists():
            print(f"[SKIP] Final ASR file already exists: {asr_final_json.name}")
            return

        whisperx = _load_dependency("whisperx", "run WhisperX ASR and diarization")
        torch = _load_dependency("torch", "run WhisperX ASR and diarization")

        cache_dir_setting = settings.get("cache_dir") or str(Path.home() / ".cache" / "whisperx")
        cache_path = Path(cache_dir_setting)
        ensure_dirs(cache_path)
        settings["cache_dir"] = str(cache_path)
        print(f"[SETUP] Using WhisperX cache at: {cache_path}")

        print(f"[PIPELINE] Loading audio from: {converted_audio_path.name}")
        # Use native loader to avoid calling ffmpeg subprocess in whisperx.load_audio
        audio = load_audio_native(str(converted_audio_path))

        # 1. Transcribe
        print("[PIPELINE] 1/3: Transcribing...")
        try:
            model = whisperx.load_model(
                settings["whisper_model_id"],
                device,
                compute_type=settings["compute_type"],
                download_root=str(cache_path),
                language=settings.get("language"),
            )
        except Exception as exc:
            raise _resource_error("load the WhisperX transcription model", exc) from exc
        result = model.transcribe(audio, batch_size=settings["batch_size"])

        print("[PIPELINE] Unloading ASR model...")
        del model
        gc.collect(); torch.cuda.empty_cache()

        # 2. Align
        print("[PIPELINE] 2/3: Verifying and refining word timestamps...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=device,
                model_name=settings["align_model_id"],
                model_dir=str(cache_path),
            )
        except Exception as exc:
            raise _resource_error("load the alignment model", exc) from exc
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )

        print("[PIPELINE] Unloading alignment model...")
        del model_a
        gc.collect(); torch.cuda.empty_cache()

        # 3. Diarize and Assign Speakers
        final_result = result
        if settings.get("do_diarization", True):
            print("[PIPELINE] 3/3: Diarizing...")
            try:
                diarize_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=settings.get("hf_token"), device=device
                )
            except Exception as exc:
                raise _resource_error("initialise the diarization pipeline", exc) from exc
            diarize_segments = diarize_model(audio, min_speakers=settings.get("diar_min_spk"), max_speakers=settings.get("diar_max_spk"))
            final_result = whisperx.assign_word_speakers(diarize_segments, result)
        else:
            print("\n--- Diarization Disabled ---")

        _save_json(final_result, asr_raw_json)

        flat_words = []
        for seg in (final_result.get("segments") or []):
            for w in (seg.get("words") or []):
                if w.get("start") is None: continue
                flat_words.append({
                    "w": str(w.get("word", "")), "start": w.get("start"), "end": w.get("end"),
                    "speaker": w.get("speaker"), "score": w.get("score")
                })
        flat_words.sort(key=lambda d: (d["start"], d["end"]))

        _save_json({"words": flat_words}, asr_final_json)
        print(f"[OK] Wrote final ASR words to: {asr_final_json.name}")

    finally:
        if converted_audio_path and converted_audio_path.exists():
            print(f"[CLEANUP] Deleting temporary audio file: {converted_audio_path.name}")
            converted_audio_path.unlink()

def main():
    """
    Main entry point for the command-line interface.

    Parses command-line arguments, loads configuration from YAML files,
    initializes the environment (device, tokens), and calls the main
    `process_file` function to execute the ASR pipeline.
    """
    parser = argparse.ArgumentParser(description="Run ASR, diarization, and alignment on an audio/video file.")
    parser.add_argument("--input-file", required=True, type=Path, help="Path to the audio/video file to process.")
    parser.add_argument("--out-root", type=Path, help="Root directory for output artifacts.")
    parser.add_argument("--config-file", type=Path, help="Path to the pipeline_config.yaml file.")
    parser.add_argument(
        "--mock-asr-json",
        type=Path,
        help=(
            "Path to a precomputed ASR+diarization JSON. When provided, the script "
            "skips audio processing and copies the JSON to the expected align output."
        ),
    )
    args = parser.parse_args()

    config = load_pipeline_config(
        DEFAULT_SETTINGS, yaml_path=str(args.config_file) if args.config_file else "pipeline_config.yaml"
    )
    script_settings = config.get("align_make", {})
    out_root = Path(args.out_root) if args.out_root else Path(
        script_settings.get("out_root", Path(config.get("pipeline_root", ".")) / "_intermediate")
    )

    paths = {
        "asr_dir": out_root / "_asr",
        "align_dir": out_root / "_align",
    }
    ensure_dirs(paths["asr_dir"]); ensure_dirs(paths["align_dir"])

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if args.mock_asr_json:
        run_mock_mode(args.input_file, args.mock_asr_json, paths["align_dir"])
        return

    set_env_tokens(script_settings.get("hf_token"))
    device = pick_device()

    print(f"[INFO] Processing single specified file: {args.input_file.name}")
    print(f"[INFO] Outputting artifacts to: {out_root}")

    try:
        process_file(args.input_file, device, paths, script_settings)
    except Exception:
        print(f"[FAIL] {base_of(args.input_file)}: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()