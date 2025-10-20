# AGENTS.md - A Guide for LLM Agents

This document provides a high-level overview of the ISCE Captioning Pipeline repository, designed to be parsed by Large Language Model (LLM) agents. It outlines the purpose of each key file, their primary interactions, and the overall data flow.

## 1. Project Mission

The primary goal of this project is to provide a robust, non-black-box, and highly controllable pipeline for segmenting transcribed text into broadcast-quality subtitle blocks (`.srt` files). It replaces an inefficient LLM-based segmentation step with a specialized statistical model trained on human captioning patterns.

**Core Task:** Transform a time-stamped transcript into a perfectly segmented `.srt` file.

## 2. Key Architectural Principles

1.  **Separation of Concerns:** The pipeline is split into distinct, logical stages. Audio processing is handled separately from text processing.
2.  **Single Source of Truth:** All feature engineering logic is consolidated into a single script (`build_training_pair_standalone.py`) to ensure perfect consistency between training and inference.
3.  **Orchestration:** A master script (`run_pipeline.py`) manages the execution flow using a "hot folder" system. Worker scripts are called as independent subprocesses.
4.  **Data Flow:** The primary data format passed between steps is a JSON file containing a list of word-level token dictionaries.

## 3. Component Map & Agent Instructions

### 3.1. The Orchestrator (Entry Point)

*   **File:** `run_pipeline.py`
*   **Role:** The **Project Manager**. This is the main entry point for all automated workflows.
*   **Agent Instructions:**
    *   To understand the full end-to-end process, start your analysis here.
    *   Pay close attention to the `process_inference_file` and `process_training_file` functions. They define the sequence of script calls and the file paths passed between them.
    *   This script is responsible for detecting files in the "hot folders" and launching the appropriate worker scripts.

### 3.2. The Audio Specialist

*   **File:** `align_make.py`
*   **Role:** A pure **Audio-to-Timed-ASR Engine**.
*   **Inputs:** A raw media file (e.g., `.mp4`).
*   **Process:**
    1.  Uses `ffmpeg` to extract and convert audio.
    2.  Runs the full WhisperX pipeline:
        *   Transcription (`KBLab/kb-whisper-large`) to get words.
        *   Alignment (`KBLab/wav2vec2-large-voxrex-swedish`) to get accurate word-level timestamps.
        *   Diarization (`pyannote.audio`) to get speaker labels.
    3.  Manages GPU memory by explicitly loading and unloading each model.
*   **Output:** A single JSON file (`...asr.visual.words.diar.json`) containing a list of word tokens with precise timestamps and speaker labels. This file serves as the "timing reference" or "bridge" for the next step.
*   **Agent Instructions:** This script is highly specialized. Focus on the `process_file` function to understand the WhisperX workflow. Note that it has no knowledge of `.txt` or `.srt` files.

### 3.3. The Data Specialist (Core Logic)

*   **File:** `build_training_pair_standalone.py`
*   **Role:** The **Heart of the Pipeline**. A monolithic "Text-to-Enriched-Data Engine."
*   **Inputs:**
    1.  `--primary-input`: A text file (`.txt` for inference, `.srt` for training).
    2.  `--asr-reference`: The JSON file produced by `align_make.py`.
*   **Process:**
    1.  **Text-to-ASR Alignment:** Calls `align_text_to_asr` to transfer the precise timestamps from the ASR reference onto the words from the primary input file. This is the core alignment logic.
    2.  **Speaker Correction:** Runs the `correct_speaker_labels` "Sole Winner" algorithm to clean up diarization errors at the sentence level.
    3.  **Feature Engineering:** Calls the consolidated `engineer_features` function to add all linguistic and heuristic features (SpaCy, pauses, speaker change guardrails, LLM hints, sentence boundaries, etc.).
    4.  **Label Generation (Training Mode Only):** If the primary input is an `.srt` file, it calls `generate_labels_from_cues` to create the ground-truth `break_type` labels.
*   **Output:** A single, fully prepared JSON file (`.enriched.json` for inference, `.train.words.json` for training).
*   **Agent Instructions:** This is the most important script for understanding the project's unique logic. To modify any feature, start your analysis in the `engineer_features` function. To modify the text alignment, analyze `align_text_to_asr`.

### 3.4. The Segmentation Engine

*   **File:** `main.py`
*   **Role:** The **Decision Maker**.
*   **Input:** An `.enriched.json` file.
*   **Process:**
    1.  Loads the enriched tokens into typed `Token` objects (defined in `isce/types.py`).
    2.  Loads the trained statistical model (`model_weights.json`) and constraints (`constraints.json`).
    3.  Initializes the `Scorer` (from `isce/scorer.py`).
    4.  Calls the `segment` function (from `isce/beam_search.py`), which runs the beam search algorithm to find the optimal sequence of `SB` and `LB` breaks.
    5.  Uses `isce/srt_writer.py` to format the final output.
*   **Output:** A final `.srt` file.
*   **Agent Instructions:** To understand the final decision-making process, analyze `isce/beam_search.py`. To understand how decisions are scored, analyze `isce/scorer.py`.

### 3.5. The Training Utilities

*   **File:** `scripts/train_model.py`
*   **Role:** The **Model Trainer**. This is a standalone script run manually by the user.
*   **Input:** A directory of `.train.words.json` files.
*   **Process:**
    1.  Loads all training data.
    2.  Calls `derive_constraints` (from `isce/model_builder.py`) to analyze the corpus.
    3.  Runs an iterative reweighting loop:
        *   Calls `build_weights` (from `isce/model_builder.py`) to learn the statistical patterns.
        *   Uses the `Scorer` to find hard examples and adjust their weights for the next iteration.
*   **Output:** The two core model files: `model_weights.json` and `constraints.json`.
*   **Agent Instructions:** To understand how the statistical model is created, analyze this script and its primary dependency, `isce/model_builder.py`.

## 4. Data Flow Summary (Inference)

1.  `MyVideo.mp4` + `MyVideo.txt` are placed in hot folders.
2.  `run_pipeline.py` calls `align_make.py` with `MyVideo.mp4`.
3.  `align_make.py` -> `MyVideo.asr.visual.words.diar.json`.
4.  `run_pipeline.py` calls `build_training_pair_standalone.py` with `MyVideo.txt` and the ASR JSON.
5.  `build_training_pair_standalone.py` -> `MyVideo.enriched.json`.
6.  `run_pipeline.py` calls `main.py` with `MyVideo.enriched.json`.
7.  `main.py` -> `MyVideo.srt`.

This guide should provide a comprehensive starting point for any LLM agent tasked with analyzing or modifying this repository.