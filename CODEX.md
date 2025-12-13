# CODEX — ISCE: Interpretable Statistical Captioning Engine

This document is the **definitive, high-signal guide** to the ISCE Caption Pipeline.  
It is written so an autonomous coding agent can fork the repo and quickly understand:

- what the project does and why it exists,
- the end-to-end architecture (inference + training),
- where the important code lives (entry points, core library, UI),
- how configuration and data contracts work,
- how to run, test, and safely modify the system.

---

## 0) One-sentence mission

Transform a **media file** plus an **edited transcript** into a **broadcast-ready `.srt`**, using an interpretable, controllable hybrid of **statistical scoring** and **explicit guardrail rules** (instead of an opaque, LLM-only formatter).

---

## 1) TL;DR for a coding agent

1. **Inference pipeline is linear**: `align_make.py` → `build_training_pair_standalone.py` → `main.py` → `.srt`
2. There are **two orchestrators**:
   - `run_pipeline.py`: a **hot-folder** supervisor for continuous operation
   - `ui/backend/*`: a **web control center** that runs the same scripts inside per-job workspaces
3. Two root configs matter:
   - `pipeline_config.yaml`: paths + worker-script behavior (ASR/enrichment)
   - `config.yaml`: beam search + scoring sliders + model paths (segmentation)
4. Primary debugging surface:
   - data artifacts under `_intermediate/` (or `ui_data/jobs/<job-id>/`)
5. Fast sanity tests:
   - `pytest` (especially `tests/test_beam_search.py`)

---

## 2) What ISCE is (and isn’t)

ISCE is designed as a **“glass box”** alternative to subtitle segmentation using a generic LLM.  
It aims to replace the brittle/expensive step of “ask an LLM to format subtitles into blocks/lines”.

ISCE **does**:
- learn segmentation patterns from human captioning corpora (statistical model),
- enforce operator-tunable **guardrails** (minimum lengths, balance penalties, etc),
- integrate inference-time **LLM structural hints** (newlines) without training leakage,
- run end-to-end as a continuous pipeline or via a web UI.

ISCE **does not**:
- attempt to be a general-purpose ASR system (it uses WhisperX as a component),
- assume that segmentation is “one true answer” (it exposes sliders and constraints).

---

## 3) Feature summary

- **Hybrid model**: learned statistical scoring + rule-based guardrails.
- **Two-stage alignment**: transfer word-level timestamps from ASR onto a corrected transcript.
- **Feature engineering**: prosody (pauses), linguistic spaCy features, and hand-built heuristics.
- **Speaker correction**: two-stage strategy (“Sole Winner” + “Guardrail”) to mitigate diarization errors.
- **LLM hint integration (inference only)**: newlines in LLM-refined text become a “strong suggestion”.
- **Automation**:
  - hot-folder orchestrator (`run_pipeline.py`)
  - web UI (FastAPI + React) for jobs, config, and monitoring

---

## 4) Repository map (what lives where)

### Top-level entry points
- `run_pipeline.py`  
  Hot-folder supervisor. Detects new files, sequences pipeline stages, archives/moves outputs.
- `align_make.py`  
  Stage 1 (audio → ASR words): extract audio, run WhisperX transcription + forced alignment, optional diarization.
- `build_training_pair_standalone.py`  
  Stage 2 (text ↔ ASR → enriched tokens): align corrected text/SRT to ASR words; compute features; label training data.
- `main.py`  
  Stage 3 (enriched tokens → segmentation → `.srt`): loads weights + constraints, runs beam search, writes SRT.
- `scripts/train_model.py`  
  Rebuilds statistical weights + constraints from training artifacts.
- `scripts/install.py`  
  Installer: venv, dependencies, spaCy Swedish model, optional UI deps.

### Core library (`isce/`)
- `isce/beam_search.py`  
  Beam search segmentation logic (lookahead, optional bidirectional reconciliation, refinement).
- `isce/scorer.py`  
  Learnt weights + guardrails + slider overrides; scores transitions and blocks.
- `isce/postprocess.py`  
  Reflow safeguards: merge/rebalance short cues after segmentation (respect speaker boundaries).
- `isce/srt_writer.py`  
  Converts segmented tokens into `.srt` content using `SB`/`LB` break types.
- `isce/types.py`  
  Token schema (dataclass) mirrored by the JSON artifacts.
- `isce/model_builder.py`, `isce/features.py`, `isce/data_validation.py`  
  Training utilities and schema/constraint logic.

### Web control center (`ui/`)
- `ui/backend/` (FastAPI)
  - `app.py`: route wiring, DI, SSE log streaming.
  - `pipelines.py`: job runner that stages inputs and invokes CLI scripts in sequence per job.
  - `services/config_service.py`: config metadata + merge logic for UI overrides.
  - `api/routes/files.py`: filesystem allowlist endpoints.
- `ui/frontend/` (Vite/React/TS)
  - Tabbed workflows: inference, training pair gen, model training, config editing
  - Live job monitor: status pills, progress bars, params/results panes, log viewer
  - Notable components: `ConfigPanel`, `OverrideEditor`, `JobBoard`, `FilePathPicker`

### Documentation / tests
- `docs/beam_search_walkthrough.md` (deep dive on segmentation/search internals)
- `docs/spacy_feature_impact.md` (why dependency features matter; token_index stability)
- `docs/build_training_pair_comparison.md` (alignment script rationale)
- `docs/alt_build_training_pair_standalone.py` (tutorial-style alternate builder)
- `tests/test_beam_search.py` (literate unit tests for segmenter and helpers)

---

## 5) Architecture overview

### 5.1 Standard inference flow (conceptual)
```
[Media File] + [Corrected TXT]
        |
        v
(1) align_make.py
        |
        v
(2) build_training_pair_standalone.py
        |
        v
(3) main.py  ->  (4) isce/srt_writer.py
        |
        v
[Final .srt]
```

### 5.2 Orchestrators (two ways to run the same stages)

#### A) Hot-folder supervisor (continuous mode)
- `run_pipeline.py` watches configured folders (drop zones + manual transcript placement).
- It triggers stage scripts and moves/archives outputs.

#### B) Web UI (job mode)
- Backend runs each job in an isolated workspace:
  - `ui_data/jobs/<job-id>/`
- It merges UI overrides from:
  - `ui_data/config/pipeline_overrides.yaml`
  with the base `pipeline_config.yaml`, then orchestrates the same scripts.

---

## 6) Getting started (requirements + install)

### 6.1 Prerequisites
- **Python**: 3.11+
- **ffmpeg**: installed and in PATH
- **Node.js + npm**: required for web UI
- **GPU (recommended)**: CUDA speeds up WhisperX; CPU works but is slower
- **Hugging Face token**: required for speaker diarization
  - via `HF_TOKEN` env var **or** `hf_token` in `pipeline_config.yaml`
- **First-run network access**: needed to download WhisperX, PyAnnote, and Swedish spaCy assets

### 6.2 Installation (recommended)
```bash
python scripts/install.py
```

What `scripts/install.py` does:
- creates `.venv`,
- upgrades `pip/setuptools/wheel`,
- installs Python deps,
- downloads `sv_core_news_lg` (spaCy Swedish model),
- runs `npm install` for UI if Node.js is present.

Useful flags:
- `--gpu` installs spaCy with CUDA 12.x support on Windows
- `--recreate-venv` replaces an existing venv
- `--skip-frontend` skips React dependencies

### 6.3 Installation (manual fallback)
- create venv
- `pip install -r requirements.txt`
- `python -m spacy download sv_core_news_lg`

Offline tip:
- the Swedish spaCy model wheel is referenced in `requirements.txt`; download it ahead of time and point `pip` at it.

---

## 7) Configuration (two files, two layers)

### 7.1 `pipeline_config.yaml` (pipeline/workers)
Controls:
- base paths: `project_root`, `pipeline_root`
- `align_make` settings: Whisper model + diarization + HF token + `skip_if_asr_exists`
- `build_pair` settings: alignment tolerances + spaCy enable/model + speaker correction + skew mitigation
- orchestrator operational parameters: `file_settle_delay_seconds`

**UI overrides layer:**  
`ui_data/config/pipeline_overrides.yaml` is merged with the base pipeline config at runtime.

### 7.2 `config.yaml` (segmentation engine)
Controls:
- `paths` to `model_weights.json` and `constraints.json`
- search strategy: `beam_width`, `lookahead_width`
- safeguards toggles: `enable_bidirectional_pass`, `enable_refinement_pass`, `enable_reflow`
- line constraints: `min_chars_for_single_word_block`
- guardrail thresholds and penalties:
  - thresholds like `min_total_chars_per_block`, `min_last_line_chars`
  - penalties like `single_word_line_penalty`, `short_block_penalty`, `short_line_penalty`,
    `extreme_balance_penalty`, `fragment_penalty`, with `fragment_char_threshold`

---

## 8) How it works (stage-by-stage)

### 8.1 Stage 1 — `align_make.py` (audio → timed ASR words)
Outputs:
- `*.asr.visual.words.diar.json` (ASR timing “bridge”)

Pipeline:
1. extract + convert audio to 16 kHz mono WAV (ffmpeg)
2. WhisperX transcription
3. forced alignment for word-level timestamps
4. optional diarization for speaker labels

### 8.2 Stage 2 — `build_training_pair_standalone.py` (text ↔ ASR → enriched tokens)

This is the “make the model’s input data” stage. It has two distinct modes:

- **Inference mode**: produce `*.enriched.json` (same schema as training tokens, but without labels) from
  - ASR-only (`--asr-only-mode`, primary input is the ASR JSON), or
  - ASR + a human transcript (`.txt`) that must be aligned to the ASR word stream.
- **Training mode**: produce `*.train.words.json` by aligning a ground-truth `.srt` against the ASR word stream and deriving the “correct” `break_type` labels.

Core responsibilities (what must work, end-to-end):
1) **Tokenization & normalization** (SRT/TXT → token list; ASR JSON → token list)
2) **Needleman–Wunsch global alignment** of transcript tokens to ASR tokens
3) **Timestamp + speaker reconstruction** for transcript tokens (including tokens inserted vs ASR)
4) **Feature engineering** (prosody, punctuation, structure, optional spaCy syntax/dependencies, etc.)
5) **Training-label derivation** (only in training mode): convert the ground-truth SRT cue structure into per-token `break_type` (`O`, `LB`, `SB`)
6) **Optional training-serving skew mitigation**: emit a simulated “ASR-style” copy of training data (`emit_asr_style_training_copy`) so training more closely resembles inference.

#### 8.2.1 Needleman–Wunsch is not “a detail”, it’s the hinge pin

The whole pipeline assumes **everything downstream operates on a single token stream** where each token has:
- a word string (`w`)
- start/end timestamps (`start`, `end`)
- speaker identity (`speaker`) and “speaker_change” flags
- engineered features used by the segmenter/scorer

If you skip / break alignment, you don’t just get slightly worse captions—you change the meaning of every engineered feature and every scoring decision. Treat this part like a cryptographic primitive: tiny changes = big downstream shifts.

#### 8.2.2 How alignment scoring works (actual code behavior)

Alignment is implemented directly in `build_training_pair_standalone.py` via:
- `_match_score(a: str, b: str, close_threshold: float, weak_threshold: float) -> int`
- `_global_align(...) -> list[tuple[int|None, int|None]]`
- `align_text_to_asr(text_tokens: list[str], asr_words: list[dict], ...) -> list[dict]`

**Match scoring (`_match_score`)**
- Both strings are normalized using **Unicode NFKC + casefold**, and then stripped of edge punctuation before comparing.
- The similarity metric is `rapidfuzz.fuzz.ratio(...)`.
- Scores are discrete:
  - `4` for exact match
  - `2` for close match (ratio ≥ `txt_match_close`)
  - `0` for weak match (ratio ≥ `txt_match_weak`)
  - `-3` otherwise

**Global alignment (`_global_align`)**
- Classic Needleman–Wunsch dynamic programming:
  - Uses a score matrix `S` and backpointer matrix `B`.
  - Gap penalty is constant (default `-3`).
  - Maximizes total score over match/mismatch + insertions + deletions.
- The returned “path” is a list of pairs:
  - `(i, j)` = transcript token `i` matched to ASR token `j`
  - `(i, None)` = transcript token inserted (not found in ASR)
  - `(None, j)` = ASR token deleted (not used to create output transcript tokens)

#### 8.2.3 Timestamp reconstruction for inserted transcript tokens (actual code behavior)

`align_text_to_asr(...)` reconstructs a *new* token list in transcript order:

- For `(i, j)` matches:
  - token `i` inherits `start/end/speaker` from ASR word `j`
  - timestamps are forced monotonic: `start >= last_end`
- For `(i, None)` insertions:
  - inserted tokens are *batched into runs*.
  - each run is assigned a time span between the nearest matched left/right neighbors.
  - `_safe_interval_split(left_end, right_start, k)` divides the span into `k` sub-intervals.
  - each inserted token gets one sub-interval.
  - speaker is inherited from the left neighbor (or right if left is missing).
- `(None, j)` deletions are skipped (they don’t create output tokens), but they influence the spacing of neighboring insertions.

Finally, there’s a small pass to ensure **no overlaps** (a token cannot start before the previous token’s end).

This is why alignment quality matters twice:
1) it chooses which ASR word donates timing/speaker metadata
2) it determines where inserted words land in time (and who “says” them)

#### 8.2.4 Structural hints from SRT formatting

When the input is an `.srt`, Stage 2 can extract *structure signals* that don’t exist in ASR:
- explicit line breaks inside a cue become token-level flags such as `is_llm_structural_break`.

Downstream, `Scorer.score_transition(...)` treats structural hints like speaker changes and dialogue dashes as high-impact signals: a structural hint can add a strong `structure_boost` to `SB` and penalize `O`. This is the “glass floor” that stops the beam from ignoring obvious subtitle structure cues.

#### 8.2.5 Outputs (what Stage 3 expects)

Whether training or inference, the output must be a JSON list of tokens where each token has, at minimum:
- `w` (string), `start` (float seconds), `end` (float seconds)
- engineered features consumed by the scorer (e.g., `pause_z`, punctuation class inputs, speaker change flags, dependency-derived keys if enabled)
- `break_type`:
  - **in inference**: typically null/absent until Stage 3 fills it in
  - **in training**: set per-token to `O`, `LB`, or `SB`

See **§9 Data contracts & artifacts** for the precise schema expectations.


### 8.3 Stage 3 — `main.py` (enriched tokens → segmentation → `.srt`)
- scores transitions and blocks via `isce/scorer.py`
- beam search in `isce/beam_search.py` chooses `O` / `LB` / `SB`
- optional passes: lookahead, bidirectional reconciliation, refinement, reflow
- final `.srt` formatting is performed by `isce/srt_writer.py`

---

## 9) Data contracts & artifacts

### 9.1 ASR reference (`*.asr.visual.words.diar.json`)
Minimal example:
```json
{
  "words": [
    {"w": "hej", "start": 12.34, "end": 12.98, "speaker": "SPEAKER_00", "score": 0.98}
  ]
}
```

### 9.2 Enriched tokens (`*.enriched.json`)
The **single canonical token stream** used by the captioning engine. Stage 3 (`main.py`) consumes this.

Produced by: `build_training_pair_standalone.py` (inference mode).

Each entry is a JSON object describing a token (word) with timing, speaker metadata, and engineered features.
Downstream code treats missing fields defensively, but quality depends on having the full feature set.

Minimum fields required for Stage 3:
- `w`: word text (string)
- `start`, `end`: timestamps in seconds (float), monotonic within the file
- `speaker`: speaker label (string or int-like); may be null if diarization disabled
- `speaker_change`: boolean flag marking a speaker transition at/after this token
- `break_type`: typically null until segmentation; will be filled with `O`/`LB`/`SB` by the segmenter

Highly important “signals” that materially affect segmentation quality:
- Prosody:
  - `pause_before_ms`, `pause_after_ms` (ms), and a normalized `pause_z` (used heavily by the scorer)
- Punctuation / text-shape:
  - sentence-final / comma-like punctuation flags (the scorer classifies punctuation from token text)
  - `is_sentence_initial`, `is_sentence_final` (if engineered)
- Structure:
  - `starts_with_dialogue_dash` (dash at start of a speaker turn)
  - `is_llm_structural_break` (structure inferred from SRT line breaks or other upstream hints)
- Token identity / stability:
  - `token_index` (int) — used to keep dependency-aware feature keys stable across lookahead/refinement/postprocess passes

Where these come from:
- Timing + speaker labels come from `align_make.py`’s ASR+alignment+diarization output.
- Transcript alignment to ASR uses Needleman–Wunsch in `build_training_pair_standalone.py` (see §8.2.2–8.2.3).
- Optional syntax/dependency features are added via spaCy when enabled in `pipeline_config.yaml` (`build_pair.spacy_*`).


### 9.3 Training tokens (`*.train.words.json`)
Same schema, but `break_type` is pre-populated by SRT cue structure.

#### Capturing human newline intent
- `tokenize_srt_cues()` attaches newline markers to the **last word before the break**
- `generate_labels_from_cues()` assigns:
  - `SB` to final word of each cue
  - `LB` to first-line closing word
  - `O` elsewhere

#### Relationship between `is_llm_structural_break`, `LB`, `SB`
| Marker | When set | Meaning | Placement |
| --- | --- | --- | --- |
| `is_llm_structural_break` | Training: metadata only; Inference: derived from LLM plaintext newlines | Hint toward starting a new block | Last word before suggested break |
| `LB` | Training label | Human editor’s line break | Last word on first line |
| `SB` | Training label + decoder-enforced | Cue/block ends | Last word of cue |

---

## 10) Running the system

### 10.1 Hot-folder mode
```bash
python run_pipeline.py
```

Inference:
- media → `1_DROP_FOLDER_INFERENCE`
- corrected TXT → `4_MANUAL_TXT_PLACEMENT`
- output `.srt` → `_output`

Training pairs:
- media → `2_DROP_FOLDER_TRAINING`
- SRT → `3_MANUAL_SRT_PLACEMENT`
- output `.train.words.json` → `_intermediate/_training`

### 10.2 Web Control Center

Backend:
```bash
uvicorn ui.backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:
```bash
cd ui/frontend
npm install
npm run dev
```

Defaults:
- backend: `http://localhost:8000`
- frontend proxies `/api`
- job artifacts: `ui_data/jobs/<job-id>/`
- pipeline overrides: `ui_data/config/pipeline_overrides.yaml`

Key endpoints:
- `/api/jobs`, `/api/jobs/{id}`, `/api/config`, `/api/jobs/{id}/logs/stream`

---

## 11) CLI entry points

| Script | Typical use | Example |
| --- | --- | --- |
| `run_pipeline.py` | hot-folder supervisor | `python run_pipeline.py` |
| `align_make.py` | media → ASR bridge | `python align_make.py --input-file media.mp4 --out-root _intermediate --config-file pipeline_config.yaml` |
| `build_training_pair_standalone.py` | align/enrich | `python build_training_pair_standalone.py --primary-input Transcript.txt --asr-reference Clip.asr.visual.words.diar.json --config-file pipeline_config.yaml` |
| `main.py` | segment → `.srt` | `python main.py --input Clip.enriched.json --output Clip.srt --config config.yaml` |
| `scripts/train_model.py` | train weights/constraints | `python scripts/train_model.py --corpus path/to/_training --constraints models/v2/constraints.json --weights models/v2/model_weights.json --iterations 5` |

---

## 12) Training a new model
1. Generate many `.train.words.json` files
2. Run trainer:
```powershell
python scripts/train_model.py --corpus "T:\AI-Subtitles\Pipeline\_intermediate\_training" --constraints "models/v2/constraints.json" --weights "models/v2/model_weights.json" --iterations 5
```
3. Update `config.yaml` to point to new model files

---

## 13) Testing
Run:
```bash
pytest
```

Notes:
- Integration tests can require large models/audio assets that are not bundled.
- Prefer targeted tests, especially `tests/test_beam_search.py`.

If installation is flaky in an agent environment, install in batches:
1) `pip install pyyaml pandas numpy rapidfuzz tqdm pysrt`  
2) `pip install pyannote.audio torch ffmpeg-python`  
3) `pip install "spacy>=3.7,<4.0" && python -m spacy download sv_core_news_lg`  
4) `pip install fastapi pydantic "uvicorn[standard]" pytest httpx`  
Then run `pytest`.

---

## 14) Operational tips
- Increase `file_settle_delay_seconds` if large uploads race detection.
- Use `skip_if_asr_exists: true` to reuse ASR outputs when iterating downstream.
- Ensure first-run network access for model downloads.

---

## 15) Known gaps / gotchas
- `pipeline_config.yaml` currently contains an `hf_token` field for diarization downloads. Do **not** keep real tokens in version control—prefer an env var (e.g. `HF_TOKEN`) and have `align_make.py` read it at runtime.
- Frontend directory validation may be stricter than backend (backend creates missing folders).
- UI `project_root` / `pipeline_root` fields may be overridden by backend behavior.
- Some CLI scripts assume WhisperX assets are already downloaded (run install first).
- Use unit tests when full integration assets aren’t available.

---

## 16) Deep references
- `docs/beam_search_walkthrough.md`
- `docs/spacy_feature_impact.md`
- `docs/build_training_pair_comparison.md`
- `docs/alt_build_training_pair_standalone.py`
- `tests/test_beam_search.py`

---

## 17) Fact-check notes on AGENTS.md

AGENTS.md is *directionally* right, but a few specifics are easy to misread months later:

- **Stage boundaries are correct**: the practical inference chain is `align_make.py` → `build_training_pair_standalone.py` → `main.py`.
  - You can see this explicitly in the orchestrators (`run_pipeline.py` and `ui/backend/pipelines.py`), both of which call those three scripts in that order.
- **Alignment is real Needleman–Wunsch**: Stage 2 implements a full global alignment with a discrete match score, gap penalty, and a reconstructed token stream (including timestamp interpolation for inserted words).
- **Some function names in AGENTS.md are “concept labels”, not literal symbols**:
  - AGENTS mentions `_apply_spacy` / `_apply_guardrails`; in the current repo these behaviors live inside the Stage 2 feature-engineering and guardrail routines, but the exact function names may differ.
  - The most stable anchors are the public CLI scripts and the JSON contracts they emit, not private helper names.
- **Config split is important**:
  - `pipeline_config.yaml` (loaded by `pipeline_config.py`) controls Stage 1/2/Hot-folder orchestration settings.
  - `config.yaml` (loaded by `isce/config.py`) controls the captioning engine (Stage 3) settings + model paths + sliders.

Practical rule for autonomous agents: treat AGENTS.md as a roadmap, then verify by tracing the orchestrators + the emitted artifacts.

