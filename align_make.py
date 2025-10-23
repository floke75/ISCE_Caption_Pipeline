#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained audio processing and speech recognition engine.

This script is responsible for the first major stage of the ISCE pipeline.
It takes a single audio or video file as input and performs the following
steps:
1.  Extracts the audio stream and converts it to a standardized format
    (16kHz mono WAV) required by the speech recognition models.
2.  Uses a WhisperX model to transcribe the audio into text.
3.  Performs forced alignment to obtain precise word-level timestamps.
4.  Applies speaker diarization to identify and label different speakers.

The final output is a single JSON file containing a flat list of words,
each with its start time, end time, and assigned speaker label. This file
serves as the foundational timing reference for all subsequent steps in the
main ISCE pipeline.
"""

import os
import json
import traceback
import gc
from pathlib import Path
from typing import Any, Dict, List
import argparse
import ffmpeg
import warnings
import yaml

# =========================
# Dependency guards
# =========================
import whisperx
import torch

# =========================
# DEFAULT SETTINGS (Self-Contained)
# =========================
DEFAULT_SETTINGS: Dict[str, Any] = {
    "project_root": r"C:\dev\Captions_Formatter\Formatter_machine",
    "pipeline_root": r"T:\AI-Subtitles\Pipeline",
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

# =========================
# Configuration Helper Functions (Self-Contained)
# =========================
def _recursive_update(base: Dict, update: Dict) -> Dict:
    """
    Recursively updates a dictionary.

    Merges the 'update' dictionary into the 'base' dictionary. If a key
    exists in both dictionaries and its value is a dictionary in both,
    it recursively merges the nested dictionaries. Otherwise, the value
    from 'update' overwrites the value in 'base'.

    Args:
        base: The dictionary to be updated.
        update: The dictionary containing new values.

    Returns:
        The updated 'base' dictionary.
    """
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base

def _resolve_paths(config: Dict, context: Dict) -> Dict:
    """
    Resolves placeholder variables in configuration paths.

    Recursively iterates through a configuration dictionary and formats any
    string values that contain `{placeholder}` style variables using the
    provided context dictionary.

    Args:
        config: The configuration dictionary with unresolved path strings.
        context: A dictionary mapping placeholder keys to their values.

    Returns:
        The configuration dictionary with path placeholders resolved.
    """
    for k, v in config.items():
        if isinstance(v, str) and "{" in v and "}" in v:
            try:
                config[k] = v.format(**context)
            except KeyError:
                pass
        elif isinstance(v, dict):
            config[k] = _resolve_paths(v, context)
    return config

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

# =========================
# Audio Processing
# =========================
def extract_and_convert_audio(video_path: Path, temp_dir: Path) -> Path:
    """
    Extracts audio from a media file and converts it to a standardized format.

    Uses ffmpeg to convert the audio from any supported video or audio file
    into a 16kHz mono WAV file, which is the required format for the
    WhisperX ASR model.

    Args:
        video_path: Path to the input media file.
        temp_dir: Directory to store the temporary WAV file.

    Returns:
        The path to the newly created WAV file.

    Raises:
        ffmpeg.Error: If ffmpeg encounters an error during conversion.
        FileNotFoundError: If the `ffmpeg` executable is not found in the system's PATH.
        IOError: If ffmpeg runs but fails to create the output file.
    """
    output_wav_path = temp_dir / f"{video_path.stem}_16khz_mono.wav"
    print(f"[AUDIO] Extracting and converting audio to: {output_wav_path.name}")
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(output_wav_path), acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(cmd='ffmpeg', capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print("[ERROR] ffmpeg failed to convert the audio.")
        print("--- ffmpeg STDERR ---")
        print(e.stderr.decode(errors='ignore'))
        raise
    except FileNotFoundError:
        raise FileNotFoundError("ERROR: `ffmpeg` command not found. Please ensure it is in your system's PATH.")
    if not output_wav_path.exists():
        raise IOError(f"ffmpeg ran but the output file was not created: {output_wav_path}")
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

        print(f"[PIPELINE] Loading audio from: {converted_audio_path.name}")
        audio = whisperx.load_audio(str(converted_audio_path))

        # 1. Transcribe
        print("[PIPELINE] 1/3: Transcribing...")
        model = whisperx.load_model(
            settings["whisper_model_id"], device, compute_type=settings["compute_type"],
            download_root=settings.get("cache_dir"), language=settings.get("language")
        )
        result = model.transcribe(audio, batch_size=settings["batch_size"])

        print("[PIPELINE] Unloading ASR model...")
        del model
        gc.collect(); torch.cuda.empty_cache()

        # 2. Align
        print("[PIPELINE] 2/3: Verifying and refining word timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device,
            model_name=settings["align_model_id"], model_dir=settings.get("cache_dir")
        )
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
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=settings.get("hf_token"), device=device
            )
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
    parser.add_argument("--out-root", required=True, type=Path, help="Root directory for output artifacts.")
    parser.add_argument("--config-file", type=Path, help="Path to the pipeline_config.yaml file.")
    args = parser.parse_args()

    config = DEFAULT_SETTINGS.copy()
    if args.config_file and args.config_file.exists():
        with open(args.config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config: config = _recursive_update(config, yaml_config)

    path_context = {k: v for k, v in config.items() if isinstance(v, str)}
    config = _resolve_paths(config, path_context)
    script_settings = config.get("align_make", {})

    set_env_tokens(script_settings.get("hf_token"))
    device = pick_device()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    paths = {
        "asr_dir": args.out_root / "_asr",
        "align_dir": args.out_root / "_align",
    }
    ensure_dirs(paths["asr_dir"]); ensure_dirs(paths["align_dir"])

    print(f"[INFO] Processing single specified file: {args.input_file.name}")
    print(f"[INFO] Outputting artifacts to: {args.out_root}")

    try:
        process_file(args.input_file, device, paths, script_settings)
    except Exception:
        print(f"[FAIL] {base_of(args.input_file)}: {traceback.format_exc()}")

if __name__ == "__main__":
    main()