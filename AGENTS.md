# AGENTS.md – Orientation for LLM Agents

This guide summarizes the ISCE Caption Pipeline so automated agents can quickly understand the codebase, discover key entry points, and find deeper documentation.

## Repository Mission
Transform a media file plus an edited transcript into a broadcast-ready `.srt`. Statistical scoring, guardrail rules, and curated heuristics replace brittle LLM-only segmentation so operators retain control and auditability.

## Where to Read Next
- **`README.md`** – Complete walkthrough of features, prerequisites, installation, CLI entry points, and the full-stack web control center.
- **`docs/build_training_pair_comparison.md`** – Deep dive into the production alignment script vs. the alternate prototype.
- **`docs/alt_build_training_pair_standalone.py`** – Self-contained alternative implementation referenced in the comparison doc.

## Architecture Overview

### Orchestrators & CLI Stages
- **`run_pipeline.py`** – Hot-folder supervisor that sequences the CLI scripts for inference/training, handles archival moves, and logs lifecycle events (`process_inference_file` / `process_training_file`).
- **`align_make.py`** – Extracts audio, invokes WhisperX + diarization, and emits timestamped ASR JSON (`process_file`).
- **`build_training_pair_standalone.py`** – Aligns human text/SRT to ASR tokens, enriches linguistic/prosodic features, and emits `.enriched.json` / `.train.words.json`. Core logic: `align_text_to_asr`, `_apply_spacy`, `_apply_guardrails`.
- **`main.py`** – Loads enriched tokens and statistical weights, runs the ISCE beam search (`isce/beam_search.py`), scores transitions via `isce/scorer.py`, and writes `.srt` output.

### Core ISCE Library (`isce/`)
- `beam_search.py` – Constrained search over break decisions using learned weights, hard limits, and heuristic boosts.
- `scorer.py` – Combines learned weights, guardrail boosts, and UI slider overrides; also provides diagnostics consumed by the UI.
- `model_builder.py`, `features.py`, and friends – Token schemas, constraint derivation, and training utilities.

### Training Utilities (`scripts/`)
- `train_model.py` aggregates corpora, recomputes constraints, and emits updated `model_weights.json` / `constraints.json`.
- `install.py` provisions the virtual environment, installs SpaCy assets, and bootstraps frontend dependencies (referenced in `README.md`).

### Web Control Center (`ui/`)
- **Backend (`ui/backend/`)** – FastAPI service exposing health, configuration, and job lifecycle APIs. `app.py` wires routes, dependency injection, and SSE log streaming. `pipelines.py` stages inputs, launches the CLI pipeline per job, and records artifacts. `services/config_service.py` materializes editable config metadata consumed by the SPA. `api/routes/files.py` powers the filesystem allowlist endpoints.
- **Frontend (`ui/frontend/`)** – Vite/React SPA with tabbed workflows (inference, training pair generation, model training, configuration editing) and a live job monitor. Components such as `ConfigPanel`, `OverrideEditor`, `JobBoard`, and `FilePathPicker` orchestrate API interactions.
- **Integration surface** – REST endpoints for job creation (`/api/jobs`), status (`/api/jobs/{id}`), and configuration (`/api/config`), plus SSE streaming (`/api/jobs/{id}/logs/stream`) for real-time logs. Overrides are persisted under `ui_data/config/pipeline_overrides.yaml`.
- **Assets & outputs** – Job artifacts, cached configs, and uploads live under `ui_data/`.

### Configuration Surface
- **`pipeline_config.yaml`** – Declares hot-folder roots, WhisperX/diarization toggles, and defaults consumed by CLI scripts and the UI backend.
- **`config.yaml`** – Holds ISCE beam-search settings, slider defaults, and model paths.
- **UI overrides** – Persisted under `ui_data/config/pipeline_overrides.yaml` and merged by `ConfigService`.

## Data Flow

### Inference Path (Hot-folder & UI share the same backbone)
1. Operator drops `MyVideo.mp4` + `MyVideo.txt` into configured watch folders **or** submits a job via the UI.
2. `run_pipeline.py` (or `ui/backend/pipelines.py`) invokes `align_make.py` to produce `MyVideo.asr.visual.words.diar.json`.
3. `build_training_pair_standalone.py` aligns/enriches the transcript and writes `MyVideo.enriched.json`.
4. `main.py` loads enriched tokens plus `config.yaml` weights to emit `MyVideo.srt` and derived diagnostics for UI download.
5. Results and intermediate artifacts are archived under the run’s output directory (mirrored to `ui_data/jobs/<id>` for UI-triggered runs).

### Training Pair Generation
1. Operator supplies human-aligned captions (SRT or text) plus media.
2. `align_make.py` produces ASR tokens; `build_training_pair_standalone.py` aligns them and emits `.train.words.json`.
3. Generated corpora are staged for `scripts/train_model.py`.

## Known Gaps & Notes
- Frontend validation for output directories is stricter than the backend, which creates missing folders—align behavior when touching UI forms.
- UI-exposed `project_root` / `pipeline_root` sliders are currently overridden by the backend; adjust messaging or respect overrides if modifying the UI.
- Some CLI scripts assume WhisperX resources have already been downloaded—follow the installation guidance in `README.md` before running alignment locally.

This document should orient any agent before deeper changes—consult the referenced README and docs for operational details and historical context.
