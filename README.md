# ISCE: Interpretable Statistical Captioning Engine

ISCE is a complete, end-to-end pipeline for transforming raw audio/video and a corrected transcript into professionally segmented, broadcast-quality subtitles. It is designed as a "glass box" alternative to opaque machine learning models, combining a data-driven statistical model with explicit, common-sense rules for maximum control and interpretability.

This pipeline is intended to replace the inefficient and error-prone step of using a generic LLM for subtitle block formatting.

## Features

*   **Hybrid Model:** Combines a statistical model trained on human captioning patterns with robust, rule-based guardrails.
*   **Two-Stage Alignment:** Uses a sophisticated process to transfer hyper-accurate word-level timestamps from an ASR transcript onto a perfect, human- or LLM-edited text.
*   **Advanced Feature Engineering:** Enriches each word with prosodic (pauses), linguistic (SpaCy), and heuristic features to inform segmentation decisions.
*   **Robust Speaker Correction:** Implements a two-stage strategy ("Sole Winner" + "Guardrail") to correct and handle common speaker diarization errors from the ASR.
*   **LLM Hint Integration:** Recognizes newlines in the input text file as a strong hint from an upstream LLM to insert a structural break.
*   **Automated Workflow:** A master orchestrator script (`run_pipeline.py`) manages the entire process using a "hot folder" system.

## Architecture Overview

The pipeline is a linear, multi-stage process managed by the orchestrator. For a standard inference run, the data flows as follows:


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

## File & Folder Structure

The repository is organized for clarity and maintainability, requiring no code changes to the scripts.

/
├── run_pipeline.py # The main orchestrator you run
├── align_make.py # Worker: Audio -> Timed ASR
├── build_training_pair_standalone.py # Worker: Text + ASR -> Enriched Data
├── main.py # Worker: Enriched Data -> SRT
├── pipeline_config.py # Utility helpers for reading pipeline_config.yaml
├── pipeline_config.yaml # Central hot-folder and worker configuration
├── config.yaml # Beam-search and scorer configuration
├── requirements.txt # All Python dependencies
├── README.md # This file
├── AGENTS.md # A guide for LLM agents
│
├── isce/ # Core ISCE toolkit (scorer, model builder, etc.)
└── scripts/ # Standalone tools (train_model.py, etc.)

User-specific artifacts such as trained models (e.g., `models/`) and local hot folders
are intentionally not committed. Create those directories locally before running the
pipeline.

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.11 or higher.
    *   `ffmpeg` must be installed on your system and accessible in your PATH.
    *   A CUDA-enabled GPU is strongly recommended for WhisperX. CPU-only runs are supported but dramatically slower on long-form audio.
    *   A Hugging Face access token with permission to download diarization models when `do_diarization: true`. Provide it in `pipeline_config.yaml` or via the `HF_TOKEN` environment variable.
    *   Ensure outbound network access the first time you execute each stage so WhisperX, PyAnnote, and the SpaCy Swedish model can be fetched and cached.

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd ISCE_Caption_Pipeline
    ```

3.  **Create Virtual Environment:**
    ```powershell
    # For Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

4.  **Install Dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

5.  **Configure the Pipeline:**
    *   Both `pipeline_config.yaml` and `config.yaml` live in the repository root and double as living templates. Copy them to a safe location before making local edits if you plan to keep untracked overrides.
    *   **Edit `pipeline_config.yaml`:** Update `project_root` to your checkout path and set `pipeline_root` (add the key if it is missing) so every hot-folder entry resolves correctly. Provide your Hugging Face token in `align_make.hf_token` when diarization is enabled.
    *   **Edit `config.yaml`:** Adjust the `paths` section so it points at the `model_weights.json` and `constraints.json` files you trained or received. Create the referenced directories (for example, `models/v1/`) before running the pipeline.
    *   Create the hot folders declared in `pipeline_config.yaml` (`_intermediate`, `_output`, `_processed`, etc.) on your filesystem so the orchestrator can place artifacts where expected.

## Configuration Files Explained

ISCE draws configuration values from two YAML files. The orchestrator can fall back to the embedded defaults, but supplying explicit files keeps multi-user deployments predictable and portable.

| File | Purpose | Notable Sections |
| --- | --- | --- |
| `pipeline_config.yaml` | Central settings for the hot-folder workflow and worker scripts. | `project_root`/`pipeline_root` establish the base directories. The nested `align_make` block controls WhisperX model IDs, diarization toggles, cache directories, and Hugging Face token usage. The `build_pair` block governs language, alignment tolerance, SpaCy features, and output targets. |
| `config.yaml` | Parameters for the segmentation engine (`main.py`). | `beam_width` sets the beam-search breadth; `constraints` provides fallback guardrails if learned constraints are absent; `sliders` adjusts per-feature multipliers; and `paths` points to the trained `model_weights.json` and `constraints.json`. |

Any path-like entry in `pipeline_config.yaml` can reference previously defined keys using Python-style placeholders. For example, `{pipeline_root}` expands to the value of `pipeline_root` declared in the same file.

## Usage

The entire pipeline is managed via the `run_pipeline.py` orchestrator and its "hot folder" system.

1.  **Start the Orchestrator:**
    ```powershell
    # Ensure your virtual environment is active
    python run_pipeline.py
    ```

2.  **Run Inference:**
    *   Place your media file (e.g., `MyVideo.mp4`) into the folder specified by `drop_folder_inference` in your config (e.g., `T:\AI-Subtitles\Pipeline\1_DROP_FOLDER_INFERENCE`).
    *   Place your corresponding corrected text file (e.g., `MyVideo.txt`) into the folder specified by `txt_placement_folder` (e.g., `T:\AI-Subtitles\Pipeline\4_MANUAL_TXT_PLACEMENT`).
    *   The orchestrator will automatically process the files. The final `.srt` file will appear in your `output_dir`.

3.  **Prepare Training Data:**
    *   Place your media file into the `drop_folder_training`.
    *   Place its corresponding ground-truth `.srt` file into the `srt_placement_folder`.
    *   The orchestrator will process the pair and create a `.train.words.json` file in your intermediate training directory.

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
| `cue_id` | Identifier linking a token back to its originating SRT cue when available. Tokens without a matching cue use `null` (inference) or `-1` (training fallback). |
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


