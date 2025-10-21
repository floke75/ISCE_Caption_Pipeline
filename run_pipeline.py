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
    """Creates all necessary pipeline directories if they do not already exist.

    This function reads a configuration dictionary to determine which folders
    are required for the pipeline to operate, ensuring the directory
    structure is in place before any processing begins.

    Args:
        cfg: The main configuration dictionary, which must contain the keys
             specified in the `dir_keys` list inside this function.
    """
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
    """Constructs the absolute path to a script within the project directory.

    Args:
        cfg: The main configuration dictionary, containing the 'project_root' key.
        script_name: The name of the script file (e.g., "main.py").

    Returns:
        A Path object representing the absolute path to the specified script.
    """
    return Path(cfg["project_root"]) / script_name

def run_command(command: list, cwd: Path):
    """Executes a command in a subprocess and streams its output in real-time.

    This function is a wrapper around `subprocess.Popen` that simplifies running
    external commands. It ensures that the output of the command (both stdout
    and stderr) is captured and printed to the console as it is generated,
    which is useful for monitoring long-running processes.

    Args:
        command: A list of strings representing the command and its arguments.
        cwd: The working directory from which to execute the command.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code,
                                       indicating an error.
    """
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
    """Orchestrates the end-to-end inference pipeline for a single media file.

    This function coordinates the three main stages of the inference process:
    1.  **Audio Processing**: Runs `align_make.py` to generate a time-stamped
        ASR (Automatic Speech Recognition) JSON file from the input media.
    2.  **Enrichment**: Runs `build_training_pair_standalone.py` to align the
        ASR data with a corresponding text file (if available) and engineer
        a rich set of features for segmentation.
    3.  **Segmentation**: Runs `main.py` to perform the final segmentation
        using the statistical model and generate the output SRT file.

    Args:
        media_file: The path to the input audio or video file.
        cfg: The main configuration dictionary.

    Raises:
        FileNotFoundError: If a critical intermediate file is not found after
                           a processing step, indicating a failure in that step.
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
    
    extra_args = []
    if txt_file_path.exists():
        print("  -> RESULT: TXT file FOUND. Setting primary input to TXT file.")
        primary_input_path = txt_file_path
    else:
        print("  -> RESULT: TXT file NOT FOUND. Entering ASR-only inference mode.")
        primary_input_path = asr_reference_path
        extra_args.extend(["--asr-only-mode", "--output-basename", base_name])
    print("="*50 + "\n")

    run_command([
        sys.executable, get_project_path(cfg, "build_training_pair_standalone.py"),
        "--primary-input", primary_input_path,
        "--asr-reference", asr_reference_path,
        "--out-inference-dir", intermediate_dir / "_inference_input",
        "--config-file", config_file_path,
        *extra_args,
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
    """Orchestrates the pipeline for preparing a single training data sample.

    This function coordinates the two main stages of the training data
    preparation process:
    1.  **Audio Processing**: Runs `align_make.py` to generate a time-stamped
        ASR JSON file from the input media. This serves as the timing reference.
    2.  **Label Generation**: Runs `build_training_pair_standalone.py` to align
        the ground-truth SRT file with the ASR data, generate break labels
        (`SB`, `LB`, `O`), and engineer a full set of features.

    Args:
        media_file: The path to the input audio or video file.
        srt_file: The path to the corresponding ground-truth SRT file.
        cfg: The main configuration dictionary.

    Raises:
        FileNotFoundError: If a critical intermediate file is not found after
                           a processing step, indicating a failure in that step.
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
    """The main orchestrator loop that monitors hot folders for new files.

    This function runs in an infinite loop, continuously scanning the
    `drop_folder_inference` and `drop_folder_training` directories.

    -   When a new media file is detected in the inference folder, it triggers
        the `process_inference_file` workflow.
    -   When a new media file is detected in the training folder, it waits for
        a corresponding `.srt` file to appear in the `srt_placement_folder`
        before triggering the `process_training_file` workflow.

    Processed files are moved to a `_processed` directory to prevent
    re-processing. The function includes error handling to gracefully manage
    failures in the sub-pipelines and can be shut down with a KeyboardInterrupt.

    Args:
        cfg: The main configuration dictionary.
    """
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