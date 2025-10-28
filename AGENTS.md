# AGENTS.md – Orientation for LLM Agents

This guide summarizes the ISCE Caption Pipeline so automated agents can quickly understand the codebase, discover key entry points, and locate the extended documentation that explains nuanced behaviours.

## Repository Mission
Transform a media file plus an edited transcript into a broadcast-ready `.srt`. Statistical scoring, guardrail rules, and curated heuristics replace brittle LLM-only segmentation so operators retain control and auditability.

## Primary References
- **`README.md`** – Comprehensive walkthrough of features, prerequisites, installation, CLI entry points, and the full-stack web control center. The **Segmentation safeguards at a glance** section links to deeper search documentation.
- **`docs/beam_search_walkthrough.md`** – Stepwise explanation of how `_token_to_row_dict`, `_get_lookahead_slice`, `Segmenter.run`, `_score_segmentation`, `_reconcile_bidirectional_breaks`, and `refine_blocks` cooperate. Pair it with the docstrings in `isce/beam_search.py` for parameter details.
- **`docs/spacy_feature_impact.md`** – Background on dependency-aware features and why stable `token_index` values matter inside scorer payloads.
- **`docs/build_training_pair_comparison.md`** – Rationale behind the production alignment script vs. the experimental standalone variant.
- **`docs/alt_build_training_pair_standalone.py`** – Fully documented alternate alignment implementation referenced in the comparison note.
- **`tests/test_beam_search.py`** – A literate test suite highlighting edge cases for the segmenter, reconciliation, and refinement helpers. Follow the class and helper docstrings to see expected scorer payloads.

## Architecture Overview

### Orchestrators & CLI Stages
- **`run_pipeline.py`** – Hot-folder supervisor that sequences the CLI scripts for inference/training, handles archival moves, and logs lifecycle events (`process_inference_file` / `process_training_file`).
- **`align_make.py`** – Extracts audio, invokes WhisperX + diarization, and emits timestamped ASR JSON (`process_file`).
- **`build_training_pair_standalone.py`** – Aligns human text/SRT to ASR tokens, enriches linguistic/prosodic features, and emits `.enriched.json` / `.train.words.json`. Core logic: `align_text_to_asr`, `_apply_spacy`, `_apply_guardrails`. For a self-contained, tutorial-style version see `docs/alt_build_training_pair_standalone.py`.
- **`main.py`** – Loads enriched tokens and statistical weights, runs the ISCE beam search (`isce/beam_search.py`), scores transitions via `isce/scorer.py`, and writes `.srt` output. The `main.segment_file` docstring links back to both orchestrators.

### Core ISCE Library (`isce/`)
- **`beam_search.py`** – Constrained search over break decisions using learned weights, hard limits, and heuristic boosts. Key docstrings: `_token_to_row_dict`, `Segmenter._build_transition_context`, `_run_forward_breaks`, `_reconcile_bidirectional_breaks`, `refine_blocks`, and `segment` (entry point describing configuration toggles).
- **`scorer.py`** – Combines learned weights, guardrail boosts, UI slider overrides, and exposes `ScoringContext`/`BlockDiagnostics` data structures. See the module docstring plus `Scorer.score_transition`/`Scorer.score_block` for integration guidance.
- **`postprocess.py`** – Describes the reflow safeguards that merge or rebalance short cues after segmentation; consult `_tokens_to_dicts` for how block payloads are reconstituted.
- **`model_builder.py`, `features.py`, `data_validation.py`** – Training utilities for token schemas, constraint derivation, and validation. Each module’s top-level docstring outlines expected input artifacts.

### Training Utilities (`scripts/`)
- **`scripts/train_model.py`** – Aggregates corpora, recomputes constraints, and emits updated `model_weights.json` / `constraints.json`. Function-level docstrings detail the CLI arguments and staging expectations.
- **`scripts/install.py`** – Provisions the virtual environment, installs SpaCy assets, and bootstraps frontend dependencies. The `Installer` class docstring lists supported flags.

### Web Control Center (`ui/`)
- **Backend (`ui/backend/`)** – FastAPI service exposing health, configuration, and job lifecycle APIs. `app.py` wires routes, dependency injection, and SSE log streaming. `pipelines.py` stages inputs, launches the CLI pipeline per job, and records artifacts. `services/config_service.py` materializes editable config metadata consumed by the SPA. `api/routes/files.py` powers the filesystem allowlist endpoints. See each router/service docstring for endpoint-level behaviour.
- **Frontend (`ui/frontend/`)** – Vite/React SPA with tabbed workflows (inference, training pair generation, model training, configuration editing) and a live job monitor. Components such as `ConfigPanel`, `OverrideEditor`, `JobBoard`, and `FilePathPicker` orchestrate API interactions. Component-level comments document prop contracts.
- **Integration surface** – REST endpoints for job creation (`/api/jobs`), status (`/api/jobs/{id}`), and configuration (`/api/config`), plus SSE streaming (`/api/jobs/{id}/logs/stream`) for real-time logs. Overrides persist under `ui_data/config/pipeline_overrides.yaml`.
- **Assets & outputs** – Job artifacts, cached configs, and uploads live under `ui_data/`.

### Configuration Surface
- **`pipeline_config.yaml`** – Declares hot-folder roots, WhisperX/diarization toggles, and defaults consumed by CLI scripts and the UI backend. Inline comments call out required fields.
- **`config.yaml`** – Holds ISCE beam-search settings, slider defaults, and model paths. The YAML comments reference the matching slider IDs used by `ConfigService`.
- **UI overrides** – Persisted under `ui_data/config/pipeline_overrides.yaml` and merged by `ConfigService.merge_overrides`. See the method docstring for precedence rules.

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
- Frontend validation for output directories is stricter than the backend, which creates missing folders—align behaviour when touching UI forms.
- UI-exposed `project_root` / `pipeline_root` sliders are currently overridden by the backend; adjust messaging or respect overrides if modifying the UI.
- Some CLI scripts assume WhisperX resources have already been downloaded—follow the installation guidance in `README.md` before running alignment locally.
- The integration tests rely on large models and audio assets that are not bundled. Use the targeted unit tests (e.g., `tests/test_beam_search.py`) when working in constrained environments.

## Recent Enhancements & Where to Learn More
- **Lookahead-aware beam search** – `docs/beam_search_walkthrough.md` and the `_get_lookahead_slice` docstring explain index propagation. Review `tests/test_beam_search.py::test_token_index_propagates_through_all_scoring_paths` for expectations.
- **Guardrail penalties for short/imbalanced cues** – Read `isce/scorer.py` (`Scorer.score_transition`, `Scorer._apply_guardrails`) alongside the README section “Segmentation safeguards at a glance” to understand slider interplay.
- **Reflow safeguards and block profiling** – `isce/postprocess.py` documents the merge heuristics; the same README section covers when to enable reflow and how it cooperates with speaker boundaries.
- **Backend config surface updates** – UI metadata lives under `ui/backend/services/config_service.py`. Cross-check with `config.yaml` and the README defaults to keep slider descriptions consistent when changing operator-facing labels.

## Break Markers Cheat Sheet (`is_llm_structural_break`, `LB`, `SB`)
- **Placement rule:** All three markers describe the **word immediately before** the visual break that follows. `isce/srt_writer` expects this convention, and the scorer boosts/penalizes transitions based on the current token’s metadata.
- **Training data (`*.train.words.json`):** `generate_labels_from_cues()` sets `LB` on the first-line closing word and `SB` on the last word in each cue. `is_llm_structural_break` is present only as carry-over metadata from `tokenize_srt_cues()` so the newline location survives alignment—it is *not* used as a feature when fitting models.
- **Inference data (`*.enriched.json`):** `is_llm_structural_break` is populated from LLM-refined plaintext newlines to hint that the next break should be an `SB`. The decoder reads the hint inside `Scorer.score_transition()` but still chooses among `O`/`LB`/`SB` using the learned weights and guardrails.
- **Why both:** `LB`/`SB` are the supervised outcomes the trainer learns from human SRTs. `is_llm_structural_break` is a reusable hint channel so inference can respect human-edit-style nudges without leaking answers into training.

## Testing Environment

The test suite is run using `pytest`. Due to issues with the `requirements.txt` file and the agent environment, it is recommended to install dependencies in batches.

1.  **Install Core Dependencies**:
    ```bash
    pip install pyyaml pandas numpy rapidfuzz tqdm pysrt
    ```
2.  **Install Speech Recognition Dependencies (excluding whisperx)**:
    ```bash
    pip install pyannote.audio torch ffmpeg-python
    ```
3.  **Install NLP Dependencies**:
    ```bash
    pip install "spacy>=3.7,<4.0" && python -m spacy download sv_core_news_lg
    ```
4.  **Install Web and Test Dependencies**:
    ```bash
    pip install fastapi pydantic "uvicorn[standard]" pytest httpx
    ```
5.  **Run Tests**:
    ```bash
    pytest
    ```

This document should orient any agent before deeper changes—consult the referenced README and docs for operational details, and lean on the cited docstrings/tests when editing specific subsystems.
