# CODEX — ISCE: Interpretable Statistical Captioning Engine

This is the **fact-checked field guide** for the ISCE Caption Pipeline. It is written for coding agents who need a fast, accurate understanding of what the system does, how it is wired, and where to look when changing or debugging it.

---

## 0) Mission (one sentence)

Transform a **media file** plus an **edited transcript** into a **broadcast-ready `.srt`**, using an interpretable hybrid of **statistical scoring** and **explicit guardrail rules** instead of a black-box, LLM-only formatter.

> Baseline snapshot for this cleanup lives at `docs/BASELINE.md`.

---

---

## 0.5) Environment Setup (Agent)

For agents operating in a persistent containerized environment (like `/app`), a setup script is provided to install system and python dependencies.

- **Setup script:** `setup.sh` (installs `ffmpeg`, `pip` dependencies, `spacy` model).
- **Execution:** `./setup.sh` (may require `sudo` for system packages, which is handled inside the script if available).

## 1) Quick orientation for an agent

See `docs/ENTRYPOINTS.md` for the canonical entrypoints and exact CLI flags. The FastAPI job runner (`ui/backend/pipelines.py`) is the primary orchestrator; `run_pipeline.py` remains as the legacy hot-folder path.

1. **Inference chain is linear:** `align_make.py` → `build_training_pair_standalone.py` → `main.py` → `.srt` (this is the exact order invoked by both orchestrators).【F:run_pipeline.py†L156-L241】【F:ui/backend/pipelines.py†L56-L189】
2. **Two orchestrators run the same chain:**
   - `run_pipeline.py`: hot-folder supervisor (CLI).【F:run_pipeline.py†L1-L340】
   - `ui/backend/pipelines.py`: FastAPI job runner (UI backend).【F:ui/backend/pipelines.py†L21-L190】
3. **Hot-folder layout (CLI):** `1_DROP_FOLDER_INFERENCE` for media, `4_MANUAL_TXT_PLACEMENT` for transcripts, `2_DROP_FOLDER_TRAINING` + `3_MANUAL_SRT_PLACEMENT` for training pairs; `_processed` keeps archives.【F:run_pipeline.py†L25-L170】【F:pipeline_config.yaml†L10-L23】
4. **Configs split by responsibility:**
   - `pipeline_config.yaml` (plus `pipeline_config.py` loader) → paths, hot folders, WhisperX/diarization/alignment toggles, UI override merge.【F:pipeline_config.yaml†L1-L65】【F:pipeline_config.py†L1-L114】【F:ui/backend/config_service.py†L96-L205】
   - `config.yaml` → beam search, scorer sliders/penalties, model weight/constraint paths (Stage 3).【F:config.yaml†L4-L53】【F:README.md†L93-L187】
5. **Artifacts to debug:** `_intermediate/_align/*.asr.visual.words.diar.json`, `_intermediate/_inference_input/*.enriched.json`, `_intermediate/_training/*.train.words.json`, `_output/*.srt` (or UI mirror `ui_data/jobs/<id>/artifacts`).【F:run_pipeline.py†L186-L246】【F:run_pipeline.py†L274-L306】【F:ui/backend/pipelines.py†L117-L190】
6. **Fast checks:** `pytest`, especially `tests/test_beam_search.py` for segmentation behaviour.【F:tests/test_beam_search.py†L1-L188】

---

## 2) What ISCE is (and isn’t)

ISCE is a **“glass box”** alternative to LLM-only subtitle formatting. It **learns segmentation patterns**, **applies rule-based guardrails**, and **respects optional LLM newline hints** without leaking labels into training.【F:README.md†L3-L62】

It **does not** replace ASR (WhisperX provides timestamps) or claim a single “true” segmentation—it exposes sliders and constraints for operator control.【F:README.md†L9-L31】【F:README.md†L87-L115】

---

## 3) Feature summary (aligned to code)

- **Hybrid scorer:** Learned weights + guardrails + optional UI slider overrides (`Scorer.score_transition` / `score_block`).【F:isce/scorer.py†L1-L119】【F:isce/scorer.py†L223-L360】
- **Two-stage alignment:** Global alignment of edited tokens onto ASR words with timestamp interpolation (`align_text_to_asr`).【F:build_training_pair_standalone.py†L242-L341】
- **Needleman–Wunsch core:** `_global_align` runs a Needleman–Wunsch matrix with fuzzy token scoring (`txt_match_close` / `txt_match_weak`) and gap penalties, then `align_text_to_asr` backfills timestamps for insertions so every edited token stays temporally anchored to the ASR.【F:build_training_pair_standalone.py†L187-L330】
- **Feature engineering:** `engineer_features` adds pause metrics, optional SpaCy tags/deps, and structural heuristics (speaker_change, dialogue dash, dangling EOS) that feed the scorer’s learned weights and guardrails.【F:build_training_pair_standalone.py†L549-L619】【F:isce/scorer.py†L150-L214】
- **Speaker correction:** A sliding-window "sole winner" relabels diarization before scoring so structural boosts respond to stable speaker_change flags; diarization itself comes straight from WhisperX.【F:build_training_pair_standalone.py†L500-L544】【F:align_make.py†L315-L368】
- **LLM structural hints (inference only):** `is_llm_structural_break` preserved in enriched tokens and read by the scorer as a hint, not a label.【F:isce/scorer.py†L167-L218】【F:build_training_pair_standalone.py†L812-L874】
- **Post-segmentation cleanup:** Optional reflow merges or rebalances short cues without crossing speaker changes (`postprocess`).【F:isce/postprocess.py†L1-L175】【F:main.py†L103-L118】
- **Automation surfaces:**
  - **Hot folder** via `run_pipeline.py` (polling, settle delays, archive handling).【F:run_pipeline.py†L1-L340】
  - **Web UI** via FastAPI/React (`ui/backend`, `ui/frontend`).【F:ui/backend/app.py†L1-L84】【F:README.md†L117-L197】

---

## 4) Repository map (fact-checked)

### Top-level scripts
- **`run_pipeline.py`** – Hot-folder orchestrator: sets up folders, calls Stage 1–3 in order, archives outputs, and handles failed runs.【F:run_pipeline.py†L1-L340】
- **`align_make.py`** – Stage 1: extract audio, run WhisperX + diarization, emit `{name}.asr.visual.words.diar.json` under `_intermediate/_align/`. Configured by `pipeline_config.yaml:align_make` keys.【F:align_make.py†L315-L410】【F:pipeline_config.yaml†L23-L52】
- **`build_training_pair_standalone.py`** – Stage 2: align edited TXT/SRT to ASR, engineer features, emit `.enriched.json` (inference) or `.train.words.json` (training).【F:build_training_pair_standalone.py†L431-L614】【F:build_training_pair_standalone.py†L873-L1042】
- **`main.py`** – Stage 3: load enriched tokens + `config.yaml`, run beam search, postprocess, and write `.srt`. See `segment_file` docstring for flags.【F:main.py†L15-L119】
- **`scripts/train_model.py`** – Train/refresh `model_weights.json` and `constraints.json` from `_training` corpus.【F:scripts/train_model.py†L1-L137】
- **`scripts/install.py`** – Installer for Python deps, SpaCy model, and optional UI frontend deps.【F:scripts/install.py†L1-L168】

### Core library (`isce/`)
- **`beam_search.py`** – Segmentation search (lookahead, optional bidirectional reconciliation, refinement).【F:isce/beam_search.py†L1-L189】
- **`scorer.py`** – Transition/block scoring with learned weights, guardrails, and overrides.【F:isce/scorer.py†L16-L360】
- **`postprocess.py`** – Reflow heuristics that merge or rebalance short cues while respecting speaker changes.【F:isce/postprocess.py†L1-L175】
- **`srt_writer.py`** – Converts segmented tokens into `.srt` blocks using `SB`/`LB` markers.【F:isce/srt_writer.py†L1-L139】
- **Training helpers:** `model_builder.py` and `data_validation.py` describe token schemas and constraint derivation (learned weight layout, constraint defaults, validation gates).【F:isce/model_builder.py†L1-L150】【F:isce/data_validation.py†L1-L168】

### Web control center (`ui/`)
- **Backend (`ui/backend/`)** – FastAPI app wiring routes and dependency injection (`app.py`), job runner that stages inputs and calls CLI scripts (`pipelines.py`), config metadata/override merge service (`config_service.py`), and allowlist endpoints (`api/routes/files.py`).【F:ui/backend/app.py†L1-L84】【F:ui/backend/pipelines.py†L1-L205】【F:ui/backend/config_service.py†L96-L219】【F:ui/backend/api/routes/files.py†L1-L215】
- **Frontend (`ui/frontend/`)** – Vite/React SPA with workflow tabs, job monitor, config editors; see component docs and README for setup commands.【F:README.md†L117-L197】
- **UI job workspaces** – Each backend job copies inputs into `ui_data/jobs/<id>/inputs`, stages artifacts under `ui_data/jobs/<id>/artifacts`, and streams logs via `JobContext`.【F:ui/backend/pipelines.py†L72-L189】

### Documentation / tests
- `docs/beam_search_walkthrough.md`, `docs/spacy_feature_impact.md`, `docs/build_training_pair_comparison.md`, `experiments/alt_build_training_pair_standalone.py`.
- `tests/test_beam_search.py` – literate tests for the segmenter and reconciliation helpers.【F:tests/test_beam_search.py†L1-L188】

---

## 5) Inference and training flows (ground truth)

### Inference (hot folder or UI)
1. **Stage 1 – Audio → ASR:** `align_make.py --input-file <media> --out-root <intermediate> --config-file pipeline_config.yaml` writes `_align/<name>.asr.visual.words.diar.json`. Diarization toggled by `do_diarization`; HF token pulled from `hf_token` or `HF_TOKEN`.【F:run_pipeline.py†L186-L200】【F:align_make.py†L315-L343】【F:pipeline_config.yaml†L29-L52】
2. **Stage 2 – Align & enrich:** `build_training_pair_standalone.py --primary-input <txt> --asr-reference <asr> --out-inference-dir <intermediate>/_inference_input --config-file pipeline_config.yaml`. Without a TXT file, Stage 2 runs ASR-only mode to keep timestamps contiguous; TXT/SRT paths propagate `is_llm_structural_break` hints and mark edited tokens while applying speaker correction and feature engineering before serialization.【F:run_pipeline.py†L220-L233】【F:build_training_pair_standalone.py†L421-L520】【F:build_training_pair_standalone.py†L820-L912】
3. **Stage 3 – Segment:** `main.py --input <name>.enriched.json --output <name>.srt --config config.yaml` runs beam search, optional bidirectional/refinement passes, optional reflow, then writes `.srt` (and `--save-labeled-json` if requested).【F:run_pipeline.py†L234-L241】【F:main.py†L30-L127】
4. **Archival/outputs:** `run_pipeline.py` moves completed inputs to `_processed` and writes SRTs under `_output`; UI mirrors artifacts under `ui_data/jobs/<id>/`.【F:run_pipeline.py†L166-L246】【F:ui/backend/pipelines.py†L142-L198】

### Training pair generation
1. Place media in `2_DROP_FOLDER_TRAINING` and matching `.srt` in `3_MANUAL_SRT_PLACEMENT` (hot folder) or submit via UI.
2. `align_make.py` produces `_align/<name>.asr.visual.words.diar.json` (same command as inference).【F:run_pipeline.py†L250-L297】
3. `build_training_pair_standalone.py --primary-input <srt> --asr-reference <asr> --out-training-dir <intermediate>/_training` emits `<name>.train.words.json` with SB/LB/O labels plus optional `.train.raw.words.json` copy when `emit_asr_style_training_copy` is true.【F:run_pipeline.py†L298-L340】【F:build_training_pair_standalone.py†L873-L937】【F:pipeline_config.yaml†L54-L65】
4. `scripts/train_model.py` consumes `_training` to refresh `constraints.json` and `model_weights.json` referenced by `config.yaml`.【F:scripts/train_model.py†L1-L137】【F:README.md†L199-L234】

---

## 6) Configuration surface

- **`pipeline_config.yaml`** – Centralized paths and Stage 1/2 settings; placeholders resolved by `pipeline_config.py::load_pipeline_config` (project/pipeline roots, intermediate/output folders, diarization options, SpaCy toggles). UI overrides are merged from `ui_data/config/pipeline_overrides.yaml` via `ConfigService`.【F:pipeline_config.yaml†L1-L65】【F:pipeline_config.py†L1-L114】【F:ui/backend/config_service.py†L96-L205】
- **`config.yaml`** – Segmentation engine settings (beam widths, lookahead, bidirectional/refinement toggles, guardrail thresholds/penalties, postprocessing). Referenced by CLI (`main.py`) and UI config metadata.【F:config.yaml†L4-L53】【F:main.py†L15-L119】【F:ui/backend/config_service.py†L96-L205】
- **Environment/Secrets:** Hugging Face diarization token should be provided via `HF_TOKEN` instead of committing real tokens to `pipeline_config.yaml`.【F:pipeline_config.yaml†L33-L52】

---

## 7) CLI entry points (confirmed)

| Script | Typical use | Example |
| --- | --- | --- |
| `run_pipeline.py` | Hot-folder supervisor | `python run_pipeline.py`【F:run_pipeline.py†L1-L340】 |
| `align_make.py` | Media → ASR reference | `python align_make.py --input-file media.mp4 --out-root _intermediate --config-file pipeline_config.yaml`【F:align_make.py†L315-L410】 |
| `build_training_pair_standalone.py` | Align/enrich | `python build_training_pair_standalone.py --primary-input Transcript.txt --asr-reference Clip.asr.visual.words.diar.json --out-inference-dir _intermediate/_inference_input --config-file pipeline_config.yaml`【F:build_training_pair_standalone.py†L431-L614】 |
| `main.py` | Segment → `.srt` | `python main.py --input Clip.enriched.json --output Clip.srt --config config.yaml`【F:main.py†L15-L119】 |
| `scripts/train_model.py` | Train weights/constraints | `python scripts/train_model.py --corpus _intermediate/_training --constraints models/v2/constraints.json --weights models/v2/model_weights.json --iterations 5`【F:scripts/train_model.py†L1-L137】 |

---

## 8) Testing

- Primary unit coverage: `tests/test_beam_search.py` for segmentation, reconciliation, and refinement edge cases.【F:tests/test_beam_search.py†L1-L188】
- Install dependencies via `python scripts/install.py` (provisions `.venv`, installs `requirements.txt`, downloads the Swedish SpaCy model, and optionally npm deps) or manually with `pip install -r requirements.txt` before running `pytest`.【F:scripts/install.py†L1-L178】【F:README.md†L64-L188】
- External prerequisites still apply when exercising the full pipeline (e.g., `ffmpeg`, Node/npm for the UI, HF token, first-run model downloads, GPU optional but recommended for WhisperX).【F:README.md†L55-L63】
- **Testing environment (batched installs):** When `requirements.txt` fails in this sandbox, install dependencies in batches before `pytest`: core (`pip install pyyaml pandas numpy rapidfuzz tqdm pysrt`), speech (`pip install pyannote.audio torch ffmpeg-python`), NLP (`pip install "spacy>=3.7,<4.0" && python -m spacy download sv_core_news_lg`), and web/test (`pip install fastapi pydantic "uvicorn[standard]" pytest httpx`).
- Some CLI scripts assume WhisperX assets are already downloaded—follow the installation guidance in `README.md` before running alignment locally.【F:README.md†L55-L63】【F:align_make.py†L300-L343】
- Integration tests rely on large, non-bundled audio/models; in constrained environments prefer the targeted unit tests (e.g., `tests/test_beam_search.py`).【F:tests/test_beam_search.py†L1-L188】

---

## 9) Intermediate artifacts & data contracts

- **ASR reference (`*.asr.visual.words.diar.json`):** Ordered words with timestamps and optional diarization produced by `align_make.py` before downstream alignment.【F:align_make.py†L315-L359】
- **Enriched tokens (`*.enriched.json`):** Mirrors the `Token` dataclass, carrying timing, speaker, linguistic, prosodic, heuristic flags, and a final `break_type` once segmentation completes.【F:isce/types.py†L20-L101】【F:build_training_pair_standalone.py†L431-L614】【F:main.py†L15-L119】
- **Training tokens (`*.train.words.json`):** Same schema as enriched tokens but labeled by `generate_labels_from_cues()` so the trainer learns human cue/line boundaries.【F:isce/types.py†L20-L101】【F:build_training_pair_standalone.py†L816-L938】
- **Training tokens (simulated ASR copy, optional):** When `emit_asr_style_training_copy` is true, a normalized lowercased copy (`*.train.raw.words.json`) reuses labels to reduce training/serving skew.【F:build_training_pair_standalone.py†L915-L937】【F:pipeline_config.yaml†L54-L65】
- **Structural hint bookkeeping:** `tokenize_srt_cues()` attaches `is_llm_structural_break` to the last word before each human newline; the hint survives alignment and later biases inference without entering training labels.【F:build_training_pair_standalone.py†L338-L367】【F:build_training_pair_standalone.py†L812-L938】

### Break markers (concise)
- **Placement rule:** `SB`, `LB`, and `is_llm_structural_break` always annotate the **word immediately before** the upcoming visual break; SRT emission reads the current token’s `break_type` to place cue/line endings.【F:build_training_pair_standalone.py†L754-L813】【F:isce/srt_writer.py†L35-L96】
- **Training data conventions:** `generate_labels_from_cues()` marks the last word in each cue as `SB` and the first-line end as `LB`, while `is_llm_structural_break` is only preserved as carry-over metadata from `tokenize_srt_cues()` so training labels stay clean.【F:build_training_pair_standalone.py†L338-L367】【F:build_training_pair_standalone.py†L754-L813】
- **Inference behavior:** `is_llm_structural_break` hints that the next break should be an `SB`, but the scorer still weighs features and guardrails to pick among `O`/`LB`/`SB`.【F:build_training_pair_standalone.py†L700-L718】【F:isce/scorer.py†L172-L219】
- **Why keep both:** `LB`/`SB` remain the supervised targets the model learns from edited SRTs, while `is_llm_structural_break` is a reusable hint channel for inference-time nudges without leaking answers into training.【F:build_training_pair_standalone.py†L754-L813】【F:isce/scorer.py†L172-L219】

## 10) Operational tips / known gaps

- Increase `file_settle_delay_seconds` if large uploads race the hot-folder poller (`run_pipeline.py` or `ui/backend/pipelines.py`).【F:run_pipeline.py†L1-L118】【F:ui/backend/pipelines.py†L31-L83】
- Set `skip_if_asr_exists: true` in `pipeline_config.yaml:align_make` when iterating downstream stages to avoid re-running WhisperX.【F:pipeline_config.yaml†L45-L52】
- **Discrepancy noted:** `pipeline_config.yaml` ships with a placeholder `hf_token` string. The code reads it literally unless the `HF_TOKEN` env var is set, so avoid committing real secrets and prefer environment-based injection.【F:pipeline_config.yaml†L33-L52】【F:align_make.py†L315-L362】
- **Known gaps:** UI file pickers validate paths against allowlisted roots and warn when directories do not yet exist, while the backend job runner will still create intermediate/output/txt folders before invoking the scripts—keep the allowlist + auto-create pairing in mind when adjusting UI messaging or behaviors.【F:ui/frontend/src/components/FilePathPicker.tsx†L149-L199】【F:ui/backend/api/routes/files.py†L134-L215】【F:ui/backend/pipelines.py†L113-L161】

---

## 11) Deep references

- `docs/beam_search_walkthrough.md` – lookahead, reconciliation, refinement payloads.
- `docs/spacy_feature_impact.md` – dependency feature rationale.
- `docs/build_training_pair_comparison.md` – rationale for Stage 2 implementation.
- `experiments/alt_build_training_pair_standalone.py` – tutorial version of Stage 2.
- `tests/test_beam_search.py` – executable examples of beam search behavior.【F:tests/test_beam_search.py†L1-L188】
