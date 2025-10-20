#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py - Master Orchestration Script for the ISCE Pipeline.
This is the refactored version with a clean separation between the
audio processing (align_make) and text processing (build_training_pair) steps.
"""

import time
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any

# =========================
# DEFAULT SETTINGS (Self-Contained)
# =========================
DEFAULT_SETTINGS: Dict[str, Any] = {
    "project_root": r"C:\dev\Captions_Formatter\Formatter_machine",
    "pipeline_root": r"T:\AI-Subtitles\Pipeline",
    
    "drop_folder_inference": "{pipeline_root}/1_DROP_FOLDER_INFERENCE",
    "drop_folder_training":  "{pipeline_root}/2_DROP_FOLDER_TRAINING",
    "srt_placement_folder":  "{pipeline_root}/3_MANUAL_SRT_PLACEMENT",
    "txt_placement_folder":  "{pipeline_root}/4_MANUAL_TXT_PLACEMENT",

    "processed_dir":    "{pipeline_root}/_processed",
    "intermediate_dir": "{pipeline_root}/_intermediate",
    "output_dir":       "{pipeline_root}/_output",
    
    "orchestrator": {
        "poll_interval_seconds": 10,
        "file_settle_delay_seconds": 5,
        "srt_wait_timeout_seconds": 300,
        "audio_exts": [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv"],
    }
}

# =========================
# Helper Functions
# =========================
def setup_directories(cfg: Dict):
    """Create all necessary directories if they don't exist."""
    print("--- Setting up pipeline directories ---")
    dir_keys = [
        "drop_folder_inference", "drop_folder_training", "srt_placement_folder",
        "txt_placement_folder", "intermediate_dir", "output_dir", "processed_dir"
    ]
    for key in dir_keys:
        path = Path(cfg[key])
        path.mkdir(parents=True, exist_ok=True)
        print(f"  - Ensuring directory exists: {path}")
    
    (Path(cfg["processed_dir"]) / "inference").mkdir(exist_ok=True)
    (Path(cfg["processed_dir"]) / "training").mkdir(exist_ok=True)
    (Path(cfg["processed_dir"]) / "srt").mkdir(exist_ok=True)
    (Path(cfg["processed_dir"]) / "txt").mkdir(exist_ok=True)

def get_project_path(cfg: Dict, script_name: str) -> Path:
    """Returns the full, absolute path to a script in the project."""
    return Path(cfg["project_root"]) / script_name

def run_command(command: list, cwd: Path):
    """Executes a command and streams its output to the console in real-time."""
    str_command = [str(c) for c in command]
    print(f"\n>>> RUNNING COMMAND: {' '.join(str_command)}")
    process = subprocess.Popen(
        str_command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='ignore', bufsize=1
    )
    print("--- [START SUBPROCESS OUTPUT] ---")
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
    process.wait()
    print("--- [END SUBPROCESS OUTPUT] ---")
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, str_command)
    print(f">>> SUCCESS: Command completed with exit code {process.returncode}.")

# =========================
# Refactored Workflow Implementations
# =========================
def process_inference_file(media_file: Path, cfg: Dict):
    """Run the full inference workflow for a media file.

    The orchestrator always regenerates the ASR reference, aligns it with the
    curated transcript when available, falls back to ASR-only mode when the TXT
    file is missing, and finally dispatches the segmenter to emit the `.srt`
    output.
    """
    print(f"\n--- STARTING INFERENCE WORKFLOW FOR: {media_file.name} ---")
    base_name = media_file.stem
    project_root = Path(cfg["project_root"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    
    asr_reference_path = intermediate_dir / "_align" / f"{base_name}.asr.visual.words.diar.json"
    txt_file_path = Path(cfg["txt_placement_folder"]) / f"{base_name}.txt"
    enriched_file_path = intermediate_dir / "_inference_input" / f"{base_name}.enriched.json"
    final_srt_path = Path(cfg["output_dir"]) / f"{base_name}.srt"
    config_file_path = get_project_path(cfg, "pipeline_config.yaml")

    # Step 1: Always run the Audio-to-ASR engine first.
    print("\n[Step 1/3] Generating time-stamped ASR reference...")
    run_command([
        sys.executable, get_project_path(cfg, "align_make.py"),
        "--input-file", media_file,
        "--out-root", intermediate_dir,
        "--config-file", config_file_path
    ], cwd=project_root)
    if not asr_reference_path.exists():
        raise FileNotFoundError(f"Audio processing did not produce ASR reference: {asr_reference_path}")

    # Step 2: Run the Text-to-Enrichment engine.
    print("\n[Step 2/3] Aligning and enriching words...")
    
    print("\n" + "="*50)
    print("[ORCHESTRATOR DIAGNOSTICS]")
    print(f"  -> Media file base name: {base_name}")
    print(f"  -> Checking for TXT file at this exact path: {txt_file_path.resolve()}")
    
    if txt_file_path.exists():
        print("  -> RESULT: TXT file FOUND. Setting primary input to TXT file.")
        primary_input_path = txt_file_path
    else:
        print("  -> RESULT: TXT file NOT FOUND. Setting primary input to ASR file (ASR-Only mode).")
        primary_input_path = asr_reference_path
    print("="*50 + "\n")

    run_command([
        sys.executable, get_project_path(cfg, "build_training_pair_standalone.py"),
        "--primary-input", primary_input_path,
        "--asr-reference", asr_reference_path,
        "--out-inference-dir", intermediate_dir / "_inference_input",
        "--config-file", config_file_path
    ], cwd=project_root)
    if not enriched_file_path.exists():
        raise FileNotFoundError(f"Enrichment did not produce expected output: {enriched_file_path}")

    # Step 3: Run the Segmentation engine.
    print("\n[Step 3/3] Segmenting and creating SRT...")
    run_command([
        sys.executable, get_project_path(cfg, "main.py"),
        "--input", enriched_file_path,
        "--output", final_srt_path,
        "--config", get_project_path(cfg, "config.yaml")
    ], cwd=project_root)

    print(f"\n--- WORKFLOW COMPLETE ---")
    print(f"Final SRT file created at: {final_srt_path}")
    if txt_file_path.exists():
        shutil.move(str(txt_file_path), str(Path(cfg["processed_dir"]) / "txt" / txt_file_path.name))
        print(f"Moved {txt_file_path.name} to processed folder.")

def process_training_file(media_file: Path, srt_file: Path, cfg: Dict):
    """Generate a labeled training pair from a media file and its reference SRT.

    The function rebuilds the ASR bridge, aligns the ground-truth subtitles,
    enriches the tokens, and writes the `.train.words.json` artifact expected by
    the training utilities.
    """
    print(f"\n--- STARTING TRAINING WORKFLOW FOR: {media_file.name} ---")
    base_name = media_file.stem
    project_root = Path(cfg["project_root"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    
    asr_reference_path = intermediate_dir / "_align" / f"{base_name}.asr.visual.words.diar.json"
    final_training_file = intermediate_dir / "_training" / f"{base_name}.train.words.json"
    config_file_path = get_project_path(cfg, "pipeline_config.yaml")

    # Step 1: Generate the time-stamped ASR reference.
    print("\n[Step 1/2] Generating time-stamped ASR reference...")
    run_command([
        sys.executable, get_project_path(cfg, "align_make.py"),
        "--input-file", media_file,
        "--out-root", intermediate_dir,
        "--config-file", config_file_path
    ], cwd=project_root)
    if not asr_reference_path.exists():
        raise FileNotFoundError(f"Audio processing did not produce ASR reference: {asr_reference_path}")

    # Step 2: Align the ground-truth SRT, enrich, and create the training pair.
    print("\n[Step 2/2] Creating training pair...")
    run_command([
        sys.executable, get_project_path(cfg, "build_training_pair_standalone.py"),
        "--primary-input", srt_file,
        "--asr-reference", asr_reference_path,
        "--out-training-dir", intermediate_dir / "_training",
        "--config-file", config_file_path
    ], cwd=project_root)
    if not final_training_file.exists():
        raise FileNotFoundError(f"Training pair creation did not produce expected output: {final_training_file}")

    print(f"\n--- WORKFLOW COMPLETE ---")
    print(f"New training file created: {final_training_file}")

# =========================
# Main Watch Folder Loop
# =========================
def main_loop(cfg: Dict):
    """The main continuous loop to monitor the hot folders."""
    print("--- Starting ISCE Pipeline Orchestrator ---")
    setup_directories(cfg)
    orch_settings = cfg.get("orchestrator", {})
    audio_exts = set(e.lower() for e in orch_settings.get("audio_exts", []))
    
    while True:
        try:
            # Inference Workflow
            drop_folder_inference = Path(cfg["drop_folder_inference"])
            inference_files = [p for p in drop_folder_inference.glob("*") if p.suffix.lower() in audio_exts]
            for media_file in inference_files:
                print(f"\n[ORCHESTRATOR] Detected new INFERENCE file: {media_file.name}")
                time.sleep(orch_settings['file_settle_delay_seconds'])
                try:
                    process_inference_file(media_file, cfg)
                    shutil.move(str(media_file), str(Path(cfg["processed_dir"]) / "inference" / media_file.name))
                    print(f"[ORCHESTRATOR] Moved {media_file.name} to processed folder.")
                except Exception as e:
                    print(f"[ORCHESTRATOR] ERROR processing {media_file.name}. Moving to 'failed'. Error: {e}")
                    failed_dir = Path(cfg["processed_dir"]) / "inference" / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    shutil.move(str(media_file), str(failed_dir / media_file.name))

            # Training Workflow
            drop_folder_training = Path(cfg["drop_folder_training"])
            training_files = [p for p in drop_folder_training.glob("*") if p.suffix.lower() in audio_exts]
            for media_file in training_files:
                print(f"\n[ORCHESTRATOR] Detected new TRAINING file: {media_file.name}")
                time.sleep(orch_settings['file_settle_delay_seconds'])
                srt_file = Path(cfg["srt_placement_folder"]) / f"{media_file.stem}.srt"
                
                print(f"Waiting for matching SRT file: {srt_file.name}...")
                wait_start = time.time()
                srt_found = False
                while time.time() - wait_start < orch_settings["srt_wait_timeout_seconds"]:
                    if srt_file.exists():
                        srt_found = True
                        break
                    time.sleep(2)
                
                if not srt_found:
                    print(f"[ORCHESTRATOR] ERROR: Timed out waiting for {srt_file.name}. Moving media file to 'failed'.")
                    (Path(cfg["processed_dir"]) / "training" / "failed").mkdir(exist_ok=True)
                    shutil.move(str(media_file), str(Path(cfg["processed_dir"]) / "training" / "failed" / media_file.name))
                    continue

                print("Found matching SRT file. Proceeding with training pipeline.")
                try:
                    process_training_file(media_file, srt_file, cfg)
                    shutil.move(str(media_file), str(Path(cfg["processed_dir"]) / "training" / media_file.name))
                    shutil.move(str(srt_file), str(Path(cfg["processed_dir"]) / "srt" / srt_file.name))
                    print(f"[ORCHESTRATOR] Moved {media_file.name} and {srt_file.name} to processed folders.")
                except Exception as e:
                    print(f"[ORCHESTRATOR] ERROR processing training pair for {media_file.name}. Moving files to 'failed'. Error: {e}")
                    (Path(cfg["processed_dir"]) / "training" / "failed").mkdir(exist_ok=True)
                    (Path(cfg["processed_dir"]) / "srt" / "failed").mkdir(exist_ok=True)
                    shutil.move(str(media_file), str(Path(cfg["processed_dir"]) / "training" / "failed" / media_file.name))
                    shutil.move(str(srt_file), str(Path(cfg["processed_dir"]) / "srt" / "failed" / srt_file.name))

            time.sleep(orch_settings["poll_interval_seconds"])
        except KeyboardInterrupt:
            print("\n[ORCHESTRATOR] Shutting down.")
            break
        except Exception as e:
            print(f"\n[ORCHESTRATOR] An unexpected error occurred in the main loop: {e}")
            print("Restarting poll in 30 seconds...")
            time.sleep(30)

if __name__ == "__main__":
    try:
        from pipeline_config import load_pipeline_config
    except ImportError:
        print("[FATAL ERROR] pipeline_config.py not found. The application cannot start.")
        sys.exit(1)
    
    CONFIG = load_pipeline_config(DEFAULT_SETTINGS)
    main_loop(CONFIG)