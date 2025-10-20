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
├── pipeline_config.py # Utility for loading YAML configs
├── requirements.txt # All Python dependencies
├── README.md # This file
├── AGENTS.md # A guide for LLM agents
│
├── isce/ # Core ISCE toolkit (scorer, model builder, etc.)
├── scripts/ # Standalone tools (train_model.py, etc.)
├── configs/ # User-editable configuration files
└── models/ # Your trained model artifacts

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.11 or higher.
    *   `ffmpeg` must be installed on your system and accessible in your PATH.

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
    *   Navigate to the `configs/` directory.
    *   Make a copy of `pipeline_config.sample.yaml` and rename it to `pipeline_config.yaml`.
    *   Make a copy of `config.sample.yaml` and rename it to `config.yaml`.
    *   **Edit `pipeline_config.yaml`:** Update the `project_root` and `pipeline_root` paths to match your local system. Add your Hugging Face token to the `hf_token` field if you plan to use speaker diarization.
    *   **Edit `config.yaml`:** Update the `paths` to point to the correct location of your trained model files within the `models/` directory.

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

## How to Train a New Model

Training is a manual step performed after you have prepared a sufficient amount of training data.

1.  **Prepare Data:** Use the training workflow described above to generate a corpus of `.train.words.json` files in your intermediate training directory (e.g., `T:\AI-Subtitles\Pipeline\_intermediate\_training`).

2.  **Run the Trainer:** Execute the `train_model.py` script, pointing it to your corpus and desired output paths for the new model.
    ```powershell
    python scripts/train_model.py --corpus "T:\AI-Subtitles\Pipeline\_intermediate\_training" --constraints "models/v2/constraints.json" --weights "models/v2/model_weights.json" --iterations 5
    ```

3.  **Update Configuration:** After training, remember to update your `config.yaml` file to point to your new `v2` model files.


