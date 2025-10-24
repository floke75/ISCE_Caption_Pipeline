# ISCE: Interpretable Statistical Captioning Engine

ISCE is a complete, end-to-end pipeline for transforming raw audio/video and a corrected transcript into professionally segmented, broadcast-quality subtitles. It is designed as a "glass box" alternative to opaque machine learning models, combining a data-driven statistical model with explicit, common-sense rules for maximum control and interpretability.

This pipeline is intended to replace the inefficient and error-prone step of using a generic LLM for subtitle block formatting.

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
- [Web Control Center](#web-control-center)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Usage](#usage)
- [Command-Line Entry Points](#command-line-entry-points)
- [How to Train a New Model](#how-to-train-a-new-model)
- [Intermediate Artifacts](#intermediate-artifacts--data-contracts)
- [Operational Tips](#operational-tips)

## Features

*   **Hybrid Model:** Combines a statistical model trained on human captioning patterns with robust, rule-based guardrails.
*   **Two-Stage Alignment:** Uses a sophisticated process to transfer hyper-accurate word-level timestamps from an ASR transcript onto a perfect, human- or LLM-edited text.
*   **Advanced Feature Engineering:** Enriches each word with prosodic (pauses), linguistic (SpaCy), and heuristic features to inform segmentation decisions.
*   **Robust Speaker Correction:** Implements a two-stage strategy ("Sole Winner" + "Guardrail") to correct and handle common speaker diarization errors from the ASR.
*   **LLM Hint Integration:** Recognizes newlines in the input text file as a strong hint from an upstream LLM to insert a structural break.
*   **Automated Workflow:** A master orchestrator script (`run_pipeline.py`) manages the entire process using a "hot folder" system.
*   **Web UI:** A full-stack control center for running jobs, managing configuration, and monitoring progress.

## Architecture Overview

The pipeline is a linear, multi-stage process. For a standard inference run, the data flows as follows:

```
[Media File] + [TXT File]
|
v
[1. run_pipeline.py] -> (Detects files)
|
+---> [2. align_make.py] -> (Creates timed ASR "bridge" file)
|
+---> [3. build_training_pair_standalone.py] -> (Aligns TXT to ASR, enriches data)
|
+---> [4. main.py] -> (Loads model, performs segmentation)
|
v
[Final .srt File]
```

## Getting Started

This section provides a step-by-step guide to get the ISCE pipeline up and running.

### 1. Prerequisites

*   **Python:** 3.11 or higher.
*   **ffmpeg:** Must be installed and accessible in your system's PATH.
*   **Node.js and npm:** Required for the web UI.
*   **GPU (Recommended):** A CUDA-enabled GPU dramatically accelerates WhisperX; CPU-only runs are supported but slower on long-form audio.
*   **Hugging Face Token:** Required for speaker diarization. Provide it via the `HF_TOKEN` environment variable or the `hf_token` field in `pipeline_config.yaml`.
*   **First-Run Network Access:** Allow outbound access the first time you install or run the pipeline so WhisperX, PyAnnote, and the SpaCy Swedish model can download required assets.

### 2. Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd ISCE_Caption_Pipeline
    ```

2.  **Run the Installer:** This creates a `.venv` folder, installs the latest
    Python dependencies, and downloads the required Swedish SpaCy model.

    *CPU-only (any platform)*
    ```bash
    python scripts/install_pipeline.py
    ```

    *CUDA-enabled SpaCy (Windows, optional)*
    ```powershell
    python scripts/install_pipeline.py --gpu
    ```

    The installer can be re-run safely. Use `--force-recreate` to rebuild the
    virtual environment, `--no-venv` to install into the active interpreter, or
    `--skip-model-download` if you already have `sv_core_news_lg` cached.

3.  **Activate the Environment:**
    ```bash
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

### 3. Configuration

1.  **Update `pipeline_config.yaml`:** This file lives in the repository root. Replace placeholder paths for `project_root` and `pipeline_root` with locations on your system and set an `hf_token` (or rely on the `HF_TOKEN` environment variable).

2.  **Update `config.yaml`:** Also in the repository root. Confirm the `paths` section references the trained model files in `models/`, and adjust the beam search `sliders` or `constraints` if you need to tune segmentation behavior.

## Web Control Center

In addition to the CLI orchestrator, the repository now ships with a full-stack control surface that exposes the most common
operations (inference, training data generation, model training, and configuration management) through a browser-based
interface.

### Backend API

The API is implemented with FastAPI in `ui/backend`. It wraps the existing pipeline scripts, launches each run inside an
isolated workspace under `ui_data/jobs/<job-id>/`, and streams stdout/stderr into per-job log files so long jobs can be
monitored safely. Configuration updates are persisted in `ui_data/config/pipeline_overrides.yaml` and merged with the base
`pipeline_config.yaml` at runtime.

Start the API with:

```bash
uvicorn ui.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend SPA

The React + TypeScript single-page application lives in `ui/frontend`. It provides:

* Tabbed forms for inference, training pair generation, and model training, each with per-run override editors.
* A structured configuration editor backed by typed field metadata plus a raw YAML override view.
* A live job monitor with status pills, progress bars, detailed parameters/results panes, and log viewers with clipboard
  shortcuts.

To run the development server:

```bash
cd ui/frontend
npm install
npm run dev
```

By default the frontend expects the FastAPI backend on `http://localhost:8000` and proxies API calls via the `/api` prefix.

## How It Works

The ISCE pipeline is a multi-stage process designed to create high-quality subtitles.

1.  **Audio Processing:** The `align_make.py` script takes a media file, extracts the audio, and uses a speech recognition model (WhisperX) to generate an initial transcript with highly accurate word-level timestamps. It can also perform speaker diarization to identify who is speaking. The output is a time-stamped JSON file that serves as a "timing reference."

2.  **Text Alignment & Enrichment:** The `build_training_pair_standalone.py` script is the core of the pipeline. It takes the timing reference from the previous step and a corrected text file (e.g., a `.txt` or `.srt` file). It intelligently aligns the corrected text to the timed words from the ASR output, effectively transferring the accurate timestamps onto the correct words. It then enriches this data with a wide range of features, including:
    *   **Linguistic Features:** Part-of-speech tags, lemmas, and syntactic dependencies from SpaCy.
    *   **Prosodic Features:** The duration of pauses between words.
    *   **Heuristic Features:** Speaker changes, sentence boundaries, and other structural patterns.

3.  **Segmentation:** The `main.py` script takes the enriched data and feeds it into the segmentation engine. This engine uses a beam search algorithm guided by a statistical model to find the optimal placement of subtitle breaks (`SB` for a block break, `LB` for a line break). The model scores different possibilities based on the features, and the beam search efficiently explores the most promising options.

4.  **SRT Generation:** The final output is a standard `.srt` subtitle file, formatted with correct timings and line breaks.

## Configuration

ISCE uses two main configuration files stored in the repository root. The UI backend persists editable overrides under `ui_data/config/`.

*   **`pipeline_config.yaml`:** This file controls the overall pipeline workflow and the behavior of the worker scripts. Key settings include:
    *   `project_root` and `pipeline_root`: The base directories for the project and the pipeline hot folders.
    *   `align_make`: Settings for the audio processing stage, including the Whisper model to use, whether to perform diarization, and your Hugging Face token.
    *   `build_pair`: Settings for the data enrichment stage, including language, alignment tolerances, and whether to use SpaCy for linguistic features.

*   **`config.yaml`:** This file configures the final segmentation engine. Key settings include:
    *   `beam_width`: The width of the beam search algorithm. A larger number may yield better results but will be slower.
    *   `sliders`: User-adjustable weights to fine-tune the importance of different features in the scoring model.
    *   `paths`: The paths to your trained `model_weights.json` and `constraints.json` files.

## Usage

The pipeline is designed to be run continuously using the `run_pipeline.py` orchestrator, which monitors a set of "hot folders."

1.  **Start the Orchestrator:**
    ```bash
    python run_pipeline.py
    ```

2.  **Inference Workflow:**
    *   Place your media file (e.g., `video.mp4`) into the `1_DROP_FOLDER_INFERENCE`.
    *   Place the corresponding corrected transcript (e.g., `video.txt`) into the `4_MANUAL_TXT_PLACEMENT` folder.
    *   The orchestrator will detect the files, run the full pipeline, and place the final `video.srt` in the `_output` directory.

3.  **Training Data Workflow:**
    *   Place your media file (e.g., `training_video.mp4`) into the `2_DROP_FOLDER_TRAINING`.
    *   Place the corresponding human-captioned ground-truth subtitle file (e.g., `training_video.srt`) into the `3_MANUAL_SRT_PLACEMENT` folder.
    *   The orchestrator will process the pair and create a `training_video.train.words.json` file in the `_intermediate/_training` directory, ready to be used for model training.

## Command-Line Entry Points

Each major stage can be executed independently for testing or debugging.

| Script | Typical Use | Key Arguments |
| --- | --- | --- |
| `run_pipeline.py` | Watches the hot folders, invokes workers, and archives processed files. | `python run_pipeline.py` merges embedded defaults with overrides from `pipeline_config.yaml` (if present in the repo root). |
| `align_make.py` | Extracts audio, runs WhisperX, and optionally diarizes speakers to create `*.asr.visual.words.diar.json`. | `python align_make.py --input-file path/to/media.mp4 --out-root path/to/_intermediate --config-file pipeline_config.yaml` |
| `build_training_pair_standalone.py` | Aligns TXT/SRT text with the ASR reference, enriches tokens, and labels training examples. | `python build_training_pair_standalone.py --primary-input Transcript.txt --asr-reference MyClip.asr.visual.words.diar.json --config-file pipeline_config.yaml` plus optional `--out-training-dir`/`--out-inference-dir`. |
| `main.py` | Runs the beam-search segmenter to produce the final `.srt` (and optional labeled JSON). | `python main.py --input MyClip.enriched.json --output MyClip.srt --config config.yaml` |
| `scripts/train_model.py` | Rebuilds the statistical model weights and constraints from `.train.words.json` files. | `python scripts/train_model.py --corpus path/to/_training --constraints models/v2/constraints.json --weights models/v2/model_weights.json --iterations 5` |

All commands accept `-h/--help` for an exhaustive argument list.

## How to Train a New Model

Training is a manual step performed after you have prepared a sufficient amount of training data.

1.  **Prepare Data:** Use the training workflow described above to generate a corpus of `.train.words.json` files in your intermediate training directory (e.g., `T:\AI-Subtitles\Pipeline\_intermediate\_training`).

2.  **Run the Trainer:** Execute the `train_model.py` script, pointing it to your corpus and desired output paths for the new model.
    ```powershell
    python scripts/train_model.py --corpus "T:\AI-Subtitles\Pipeline\_intermediate\_training" --constraints "models/v2/constraints.json" --weights "models/v2/model_weights.json" --iterations 5
    ```

3.  **Update Configuration:** After training, remember to update your `config.yaml` file to point to your new `v2` model files.


## Intermediate Artifacts & Data Contracts

Understanding the on-disk artifacts makes it easier to integrate ISCE into adjacent systems or to add instrumentation.

### 1. ASR Reference (`*.asr.visual.words.diar.json`)

Produced by `align_make.py`, this JSON file contains a sorted list of words with timestamps and diarization metadata. A minimal example:

```json
{
  "words": [
    {
      "w": "hej",
      "start": 12.34,
      "end": 12.98,
      "speaker": "SPEAKER_00",
      "score": 0.98
    }
  ]
}
```

Only tokens with valid timestamps are retained. This artifact is the bridge between audio timing and text alignment.

### 2. Enriched Tokens (`*.enriched.json`)

Created by `build_training_pair_standalone.py` during inference runs. Each entry mirrors the immutable `Token` dataclass in `isce/types.py`.

| Field | Description |
| --- | --- |
| `w` | Surface form of the token. |
| `start` / `end` | Word-level timestamps in seconds (rounded per `round_seconds`). |
| `speaker` | Speaker label propagated from the ASR or corrected by the Sole Winner algorithm. |
| `cue_id` | Identifier linking a token back to its originating cue (training) or `null` during inference. |
| `is_sentence_initial` / `is_sentence_final` | Sentence boundary flags emitted by the enrichment pipeline. |
| `pause_before_ms` / `pause_after_ms` / `pause_z` | Prosodic pause metrics. |
| `pos`, `lemma`, `tag`, `morph`, `dep`, `head_idx` | SpaCy linguistic annotations when `spacy_enable: true`. |
| `starts_with_dialogue_dash`, `speaker_change`, `num_unit_glue`, `is_llm_structural_break` | Hand-crafted heuristic flags consumed by the scorer. |
| `break_type` | Assigned only after segmentation (`null` in raw enriched files). |

### 3. Training Tokens (`*.train.words.json`)

Structurally identical to the enriched tokens, but `break_type` is pre-populated by `generate_labels_from_cues()` so the trainer can learn from human-edited SRT boundaries.

### 4. Final Deliverables

`main.py` writes the broadcast-ready `.srt` file and, when invoked with `--save-labeled-json`, emits a labeled JSON copy that mirrors the token schema with `break_type` filled in.

## Operational Tips

* The orchestrator enforces a short "settle" delay before reading new files. Increase `file_settle_delay_seconds` in `pipeline_config.yaml` if you routinely upload large files that take longer to finish copying.
* Set `skip_if_asr_exists: true` in the `align_make` section when re-running downstream stages on previously aligned audio. This keeps debugging iterations fast by reusing cached ASR output.
* The SpaCy Swedish model wheel is referenced directly in `requirements.txt`. For offline installations, download the wheel ahead of time and point `pip` to the saved file.
