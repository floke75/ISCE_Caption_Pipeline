#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Master orchestration script for the ISCE Pipeline.

# === Legacy hot-folder orchestrator ===
# The canonical path for running ISCE is the Web Control Center
# (FastAPI backend + React frontend). This script remains supported for
# batch/air-gapped workflows but is considered secondary. See README for
# guidance.

This script monitors a set of "hot folders" for new media files and triggers
the appropriate processing pipeline (either inference or training data
preparation). It orchestrates the execution of several worker scripts in
sequence to perform tasks like audio extraction, speech recognition, text
alignment, and final subtitle segmentation.

The pipeline is designed to be robust, with error handling and file management
to ensure that processed files are archived and failed jobs are isolated.

It operates based on a configuration that can be defined in this file
(DEFAULT_SETTINGS) and overridden by a YAML file (e.g.,
`pipeline_config.yaml`) that is loaded and merged by
`pipeline_config.load_pipeline_config`. When run manually, provide
``--config-file`` to point at an alternative YAML.

Attributes:
    DEFAULT_SETTINGS (Dict[str, Any]): A dictionary containing the default
        configuration for the pipeline, including root paths, folder locations,
        and orchestrator settings. These defaults are merged with any YAML
        overrides provided to ``load_pipeline_config``.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

from pipeline_config import load_pipeline_config

# =========================
# DEFAULT SETTINGS (Self-Contained)
# =========================
DEFAULT_SETTINGS: Dict[str, Any] = {
    "project_root": ".",
    "pipeline_root": "{project_root}/pipeline_data",
    
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
def setup_directories(cfg: Dict[str, Any]):
    """Creates all necessary pipeline directories if they do not already exist.

    This function reads a configuration dictionary to determine which folders
    are required for the pipeline to operate. It ensures that the core hot
    folders, as well as archival and intermediate storage directories, are in
    place before any file processing begins.

    Args:
        cfg: The main configuration dictionary, which must contain paths for
             various pipeline stages (e.g., 'drop_folder_inference',
             'processed_dir').
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
    
    # Create specific subdirectories for archival purposes.
    (Path(cfg["processed_dir"]) / "inference").mkdir(exist_ok=True)
    (Path(cfg["processed_dir"]) / "training").mkdir(exist_ok=True)
    (Path(cfg["processed_dir"]) / "srt").mkdir(exist_ok=True)
    (Path(cfg["processed_dir"]) / "txt").mkdir(exist_ok=True)


def load_configuration(config_file: Path | None) -> Dict[str, Any]:
    """Load configuration defaults merged with optional YAML overrides."""

    yaml_path = str(config_file) if config_file else "pipeline_config.yaml"
    return load_pipeline_config(DEFAULT_SETTINGS, yaml_path=yaml_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for overriding config paths."""

    parser = argparse.ArgumentParser(description="Run the ISCE hot-folder orchestrator")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("pipeline_config.yaml"),
        help="Path to the pipeline_config YAML file used to override DEFAULT_SETTINGS.",
    )
    return parser.parse_args(argv)

def get_project_path(cfg: Dict[str, Any], script_name: str) -> Path:
    """Constructs the absolute path to a script within the project directory.

    This helper function is used to locate worker scripts (like `align_make.py`)
    relative to the project's root directory as defined in the configuration.

    Args:
        cfg: The main configuration dictionary, containing the 'project_root'
             key.
        script_name: The name of the script file (e.g., "main.py").

    Returns:
        A Path object representing the absolute path to the specified script.
    """
    return Path(cfg["project_root"]) / script_name

def run_command(command: list, cwd: Path):
    """Executes a command in a subprocess and streams its output in real-time.

    This function is a wrapper around `subprocess.Popen` that simplifies running
    external Python scripts or other shell commands. It ensures that the
    output of the command (both stdout and stderr) is captured and printed to
    the console as it is generated, which is crucial for monitoring the
-   progress of long-running pipeline stages.

    Args:
        command: A list of strings representing the command and its arguments
                 (e.g., `['python', 'main.py', '--input', 'file.json']`).
        cwd: The working directory from which to execute the command. This is
             typically the project root to ensure consistent relative path
             resolution.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit
                                       code, indicating that a pipeline stage
                                       has failed.
    """
    # Ensure all parts of the command are strings for Popen.
    str_command = [str(c) for c in command]
    print(f"\n>>> RUNNING COMMAND: {' '.join(str_command)}")

    # Start the subprocess.
    process = subprocess.Popen(
        str_command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore',
        bufsize=1  # Line-buffered
    )

    # Stream the output in real-time.
    print("--- [START SUBPROCESS OUTPUT] ---")
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
    process.wait()  # Wait for the subprocess to complete.
    print("--- [END SUBPROCESS OUTPUT] ---")

    # Check for errors.
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, str_command)

    print(f">>> SUCCESS: Command completed with exit code {process.returncode}.")

# =========================
# Refactored Workflow Implementations
# =========================
def process_inference_file(media_file: Path, cfg: Dict[str, Any]):
    """Orchestrates the end-to-end inference pipeline for a single media file.

    This function coordinates the three main stages of the inference process:
    1.  **Audio Processing**: Runs `align_make.py` to generate a time-stamped
        ASR (Automatic Speech Recognition) JSON file from the input media.
    2.  **Enrichment**: Runs `build_training_pair_standalone.py` to align the
        ASR data with a corresponding text file and engineer a rich set of
        features. If no text file is found, it runs in "ASR-only" mode.
    3.  **Segmentation**: Runs `main.py` to perform the final segmentation
        using the statistical model and generate the output SRT file.

    It handles file I/O, constructs command-line arguments for the worker
    scripts, and moves processed files to archival locations.

    Args:
        media_file: The path to the input audio or video file.
        cfg: The main configuration dictionary.

    Raises:
        FileNotFoundError: If a critical intermediate file (like the ASR
                           reference or enriched JSON) is not found after a
                           processing step, indicating a failure in that step.
    """
    print(f"\n--- STARTING INFERENCE WORKFLOW FOR: {media_file.name} ---")
    base_name = media_file.stem
    project_root = Path(cfg["project_root"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    
    # Define paths for all intermediate and final files.
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
    
    # Diagnostic print to clarify which mode is being used.
    print("\n" + "="*50)
    print("[ORCHESTRATOR DIAGNOSTICS]")
    print(f"  -> Media file base name: {base_name}")
    print(f"  -> Checking for TXT file at this exact path: {txt_file_path.resolve()}")
    
    extra_args = []
    if txt_file_path.exists():
        print("  -> RESULT: TXT file FOUND. Setting primary input to TXT file.")
        primary_input_path = txt_file_path
    else:
        # If no TXT file, use the ASR output as the primary source.
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
    # Archive the processed TXT file.
    if txt_file_path.exists():
        shutil.move(str(txt_file_path), str(Path(cfg["processed_dir"]) / "txt" / txt_file_path.name))
        print(f"Moved {txt_file_path.name} to processed folder.")

def process_training_file(media_file: Path, srt_file: Path, cfg: Dict[str, Any]):
    """Orchestrates the pipeline for preparing a single training data sample.

    This function coordinates the two main stages of the training data
    preparation process:
    1.  **Audio Processing**: Runs `align_make.py` to generate a time-stamped
        ASR JSON file from the input media. This serves as the timing
        reference.
    2.  **Label Generation**: Runs `build_training_pair_standalone.py` to align
        the ground-truth SRT file with the ASR data, generate break labels
        (`SB`, `LB`, `O`), and engineer a full set of features.

    The output is a `.train.words.json` file, ready to be consumed by the
    model training script.

    Args:
        media_file: The path to the input audio or video file.
        srt_file: The path to the corresponding ground-truth SRT file.
        cfg: The main configuration dictionary.

    Raises:
        FileNotFoundError: If a critical intermediate file (like the ASR
                           reference) is not found after a processing step.
    """
    print(f"\n--- STARTING TRAINING WORKFLOW FOR: {media_file.name} ---")
    base_name = media_file.stem
    project_root = Path(cfg["project_root"])
    intermediate_dir = Path(cfg["intermediate_dir"])
    
    # Define paths for intermediate and final files.
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
def main_loop(cfg: Dict[str, Any]):
    """The main orchestrator loop that monitors hot folders for new files.

    This function runs in an infinite loop, continuously scanning the
    `drop_folder_inference` and `drop_folder_training` directories defined in
    the configuration.

    -   When a new media file is detected in the inference folder, it triggers
        the `process_inference_file` workflow.
    -   When a new media file appears in the training folder, it waits for a
        corresponding `.srt` file to be placed in the `srt_placement_folder`
        before triggering the `process_training_file` workflow.

    The loop includes logic to handle file settling (to avoid reading
    partially copied files), timeouts for waiting on SRTs, and robust error
    handling that moves failed files to designated "failed" subdirectories
    to prevent blocking the pipeline.

    Args:
        cfg: The main configuration dictionary.
    """
    print("--- Starting ISCE Pipeline Orchestrator ---")
    setup_directories(cfg)
    orch_settings = cfg.get("orchestrator", {})
    audio_exts = set(e.lower() for e in orch_settings.get("audio_exts", []))
    
    while True:
        try:
            # --- Inference Workflow ---
            drop_folder_inference = Path(cfg["drop_folder_inference"])
            inference_files = [p for p in drop_folder_inference.glob("*") if p.suffix.lower() in audio_exts]

            for media_file in inference_files:
                print(f"\n[ORCHESTRATOR] Detected new INFERENCE file: {media_file.name}")
                time.sleep(orch_settings['file_settle_delay_seconds'])
                try:
                    process_inference_file(media_file, cfg)
                    # Archive the processed media file on success.
                    shutil.move(str(media_file), str(Path(cfg["processed_dir"]) / "inference" / media_file.name))
                    print(f"[ORCHESTRATOR] Moved {media_file.name} to processed folder.")
                except Exception as e:
                    print(f"[ORCHESTRATOR] ERROR processing {media_file.name}. Moving to 'failed'. Error: {e}")
                    failed_dir = Path(cfg["processed_dir"]) / "inference" / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    shutil.move(str(media_file), str(failed_dir / media_file.name))

            # --- Training Workflow ---
            drop_folder_training = Path(cfg["drop_folder_training"])
            training_files = [p for p in drop_folder_training.glob("*") if p.suffix.lower() in audio_exts]

            for media_file in training_files:
                print(f"\n[ORCHESTRATOR] Detected new TRAINING file: {media_file.name}")
                time.sleep(orch_settings['file_settle_delay_seconds'])
                srt_file = Path(cfg["srt_placement_folder"]) / f"{media_file.stem}.srt"
                
                # Wait for the corresponding SRT file to arrive.
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
                    failed_dir = Path(cfg["processed_dir"]) / "training" / "failed"
                    failed_dir.mkdir(exist_ok=True)
                    shutil.move(str(media_file), str(failed_dir / media_file.name))
                    continue

                print("Found matching SRT file. Proceeding with training pipeline.")
                try:
                    process_training_file(media_file, srt_file, cfg)
                    # Archive both media and SRT files on success.
                    shutil.move(str(media_file), str(Path(cfg["processed_dir"]) / "training" / media_file.name))
                    shutil.move(str(srt_file), str(Path(cfg["processed_dir"]) / "srt" / srt_file.name))
                    print(f"[ORCHESTRATOR] Moved {media_file.name} and {srt_file.name} to processed folders.")
                except Exception as e:
                    print(f"[ORCHESTRATOR] ERROR processing training pair for {media_file.name}. Moving files to 'failed'. Error: {e}")
                    failed_dir_training = Path(cfg["processed_dir"]) / "training" / "failed"
                    failed_dir_srt = Path(cfg["processed_dir"]) / "srt" / "failed"
                    failed_dir_training.mkdir(exist_ok=True)
                    failed_dir_srt.mkdir(exist_ok=True)
                    shutil.move(str(media_file), str(failed_dir_training / media_file.name))
                    shutil.move(str(srt_file), str(failed_dir_srt / srt_file.name))

            # Wait before polling again.
            time.sleep(orch_settings["poll_interval_seconds"])

        except KeyboardInterrupt:
            print("\n[ORCHESTRATOR] Shutting down.")
            break
        except Exception as e:
            # Catch-all for unexpected errors in the main loop itself.
            print(f"\n[ORCHESTRATOR] An unexpected error occurred in the main loop: {e}")
            print("Restarting poll in 30 seconds...")
            time.sleep(30)

def main(argv: list[str] | None = None) -> None:
    """Entry point that loads configuration and starts the orchestrator loop."""

    args = parse_args(argv)
    config = load_configuration(args.config_file)
    main_loop(config)


if __name__ == "__main__":
    main()