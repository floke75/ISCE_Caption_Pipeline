# ISCE Caption Pipeline Frontend Update Plan — Work-in-Progress

**Plan ID:** `isce_caption_pipeline_frontend_update_v1`
**Repository:** `floke75/ISCE_Caption_Pipeline`
**Plan created (UTC):** `2025-07-05T00:00:00Z`

**Step status summary:** 0/16 passed, 0 failed, 0 in progress (initial draft)

---

## Purpose

Create a structured, auditable plan to stabilize and improve the user interface for training and inference, emphasizing clarity for non-experts, embedded guidance, and visual validation of artifacts.

---

## Agent operating rules (strict workflow)

- Obey repository-level instructions (see `/workspace/ISCE_Caption_Pipeline/AGENTS.md`).
- Execute steps strictly in order; do not mark a step as passed until its verification test succeeds exactly as written.
- After each step passes, commit with the step id in the commit message (e.g., `S02: ...`).
- If a step fails, do not proceed—fix within the same step until tests pass.
- Keep changes offline-friendly; avoid WhisperX/model downloads unless explicitly required by the step.
- Document all deviations or skipped items in the step's Notes.
- **Tests are integral for code changes:**
  - Every code-affecting step must specify at least one verification command (unit test, lint, type check, or script) OR justify why no executable test is applicable.
  - Record actual command outputs in Notes when executed; passing tests are required before marking a step as ✅.
- **Screenshots are mandatory for UI-affecting steps:**
  - Capture at least one screenshot of the relevant UI after completing the step for both LLM-agent verification and human review.
  - Store screenshots under `docs/screenshots/<step_id>/` with descriptive filenames.
  - Reference the screenshot path in the step's Notes once captured.

---

## Assumptions

- **Node/npm:** available for frontend builds (or installable via project scripts).
- **Python >= 3.10:** available as `python` for backend and tooling.
- **ffmpeg/GPU:** not required for plan steps unless explicitly called out; avoid triggering heavy downloads.
- **git:** available as `git`.

---

## Conventions

- **New/standardized directories for this plan:**
  - `docs/screenshots/<step_id>/` — required screenshot storage per UI-affecting step.
  - `docs/notes/` — optional scratch notes or observations captured during steps.
- **Fixture references (if needed):** reuse existing fixtures under `tests/fixtures` for smoke checks.
- **Status markers:** ⬜ Not started | 🔄 In progress | ✅ Passed | ⚠️ Blocked | ❌ Failed

---

## Steps

### Index

- [S00 — Establish current frontend state and data collection protocol](#s00-establish-current-frontend-state-and-data-collection-protocol) — ⬜ Not started
- [S01 — Baseline UI capture and navigation audit](#s01-baseline-ui-capture-and-navigation-audit) — ⬜ Not started
- [S02 — Document existing user guidance and gaps](#s02-document-existing-user-guidance-and-gaps) — ⬜ Not started
- [S03 — Frontend build, lint, and dependency baseline](#s03-frontend-build-lint-and-dependency-baseline) — ⬜ Not started
- [S04 — Training flow input clarity and validation](#s04-training-flow-input-clarity-and-validation) — ⬜ Not started
- [S05 — Inference flow guidance and presets](#s05-inference-flow-guidance-and-presets) — ⬜ Not started
- [S06 — Submission feedback, validation errors, and recovery](#s06-submission-feedback-validation-errors-and-recovery) — ⬜ Not started
- [S07 — Job monitoring baseline and navigation](#s07-job-monitoring-baseline-and-navigation) — ⬜ Not started
- [S08 — Artifact visibility and preview widgets](#s08-artifact-visibility-and-preview-widgets) — ⬜ Not started
- [S09 — Training alignment visualization design (SRT ↔ ASR/NW)](#s09-training-alignment-visualization-design-srt--asrnw) — ⬜ Not started
- [S10 — Inference alignment visualization design (LLM ↔ ASR/NW)](#s10-inference-alignment-visualization-design-llm--asrnw) — ⬜ Not started
- [S11 — System health signals and observability hooks](#s11-system-health-signals-and-observability-hooks) — ⬜ Not started
- [S12 — Insight verification and test harness planning](#s12-insight-verification-and-test-harness-planning) — ⬜ Not started
- [S13 — Guided job templates and presets](#s13-guided-job-templates-and-presets) — ⬜ Not started
- [S14 — Data quality dashboard and cue diagnostics](#s14-data-quality-dashboard-and-cue-diagnostics) — ⬜ Not started
- [S15 — Embedded help center and onboarding tours](#s15-embedded-help-center-and-onboarding-tours) — ⬜ Not started

---

## S00 — Establish current frontend state and data collection protocol

**Status:** ⬜ Not started

**Objective:** Confirm environment readiness for frontend review and define how screenshots and artifacts will be captured and stored throughout the plan.

**Actions to perform:**
- Verify frontend install/build commands in `FRONTEND.md` and ensure npm/yarn availability.
- Create `docs/screenshots/S00/` and `docs/notes/` directories if missing.
- Record the baseline commands for running the frontend locally (dev mode) and for building static assets.
- Outline the standard screenshot capture process (naming, resolution, where to store) and note any tool limitations in this environment.

**Code/doc pointers:** `FRONTEND.md` (dev/build commands), `ui/frontend/README.md` (component overview if present), `scripts/install.py` (optional npm install helper), top-level `README.md` (UI run instructions).

**Deliverables:**
- `docs/notes/frontend_readiness.md` summarizing available tooling, commands, and any constraints.
- Directories `docs/screenshots/S00/` and `docs/notes/` present in the repo.

**Verification test:**
- **Name:** Frontend readiness documented
- **Commands:**

```text
test -d docs/screenshots/S00
test -d docs/notes
test -f docs/notes/frontend_readiness.md
grep -q "frontend dev" docs/notes/frontend_readiness.md
```
- **Expected results:**
  - Required directories exist
  - Readiness note exists and mentions the frontend dev command(s)
- **Pass criteria:** All commands exit with code 0 AND expected contents are present.

**Notes:**
- No UI screenshots required for this setup step.

---

## S01 — Baseline UI capture and navigation audit

**Status:** ⬜ Not started

**Objective:** Run the current frontend, capture baseline screenshots of key views, and map navigation flows to anchor future comparisons.

**Actions to perform:**
- Start the frontend in dev mode using the commands validated in S00.
- Capture screenshots of primary views (home/landing, training form, inference form, job board/monitor, config editor if present). Save to `docs/screenshots/S01/` with descriptive filenames (e.g., `training-form.png`).
- Note any console errors, missing assets, or navigation dead ends encountered during browsing.
- Record the versions of frontend dependencies if easily accessible (package.json lock info or dev server banner).

**Code/doc pointers:** `ui/frontend/src/App.tsx` (routing/tabs), `ui/frontend/src/components/NavigationTabs.tsx` (if present), `ui/frontend/src/components` directory for page-level views, `ui/frontend/package.json` and `ui/frontend/package-lock.json` for dependency versions.

**Deliverables:**
- Screenshots under `docs/screenshots/S01/` covering the primary views.
- `docs/notes/ui_baseline.md` summarizing navigation paths, errors, and environment details (port, dev server command used).

**Verification test:**
- **Name:** Baseline UI captured
- **Commands:**

```text
test -d docs/screenshots/S01
test -f docs/notes/ui_baseline.md
find docs/screenshots/S01 -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Screenshot directory exists and contains at least one file
  - Baseline notes file exists
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Include links or paths to captured screenshots in the notes.

---

## S02 — Document existing user guidance and gaps

**Status:** ⬜ Not started

**Objective:** Inventory the current inline help, placeholders, and documentation touchpoints in the UI to identify where non-experts may struggle.

**Actions to perform:**
- Review visible help text/tooltips/placeholders across forms (training, inference, file pickers, config inputs) using the running app.
- Capture focused screenshots of representative help text (or the lack thereof) and store under `docs/screenshots/S02/`.
- Catalogue existing links to documentation within the UI and note where they lead.
- Summarize identified gaps where additional guidance or visual affordances are needed.

**Deliverables:**
- Screenshots under `docs/screenshots/S02/` illustrating current guidance (or missing guidance).
- `docs/notes/ui_guidance_gaps.md` listing observed help text, links, and prioritized gaps.

**Verification test:**
- **Name:** Guidance inventory recorded
- **Commands:**

```text
test -d docs/screenshots/S02
test -f docs/notes/ui_guidance_gaps.md
find docs/screenshots/S02 -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Guidance screenshot directory exists and has at least one file
  - Guidance gaps note exists
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Highlight any areas where screenshots could not be captured due to environment limitations; propose alternatives if needed.

---

## S03 — Frontend build, lint, and dependency baseline

**Status:** ⬜ Not started

**Objective:** Establish the current reliability baseline by ensuring the frontend builds, passes lint/tests (where available), and documenting any blockers or flaky behavior.

**Actions to perform:**
- Install frontend dependencies following `FRONTEND.md` (prefer `npm ci` if lockfile present).
- Run the standard build (`npm run build`) and available lint/test commands (e.g., `npm run lint`, `npm test` or `npm run test:unit`).
- Capture any build errors, warnings, or known flaky tests with remediation notes.
- Note Node/npm versions used for reproducibility.

**Code/doc pointers:** `ui/frontend/package.json` (scripts), `ui/frontend/package-lock.json` (locked deps), `FRONTEND.md` (expected commands), `scripts/install.py` (npm bootstrap step).

**Deliverables:**
- `docs/notes/frontend_reliability.md` capturing commands run, outcomes, and any issues.
- Confirmed command list for future steps (build, lint, tests) with observed durations and caveats.

**Verification test:**
- **Name:** Frontend reliability baseline
- **Commands:**

```text
cd ui/frontend && npm install
cd ui/frontend && npm run build
cd ui/frontend && npm run lint
```
- **Expected results:**
  - Dependency install succeeds
  - Build completes without errors
  - Lint passes (or explicitly documents lint gaps if command is unavailable)
- **Pass criteria:** All commands exit with code 0 OR documented blockers with proposed mitigation in `docs/notes/frontend_reliability.md`.

**Notes:**
- Record actual command outputs and durations in the notes; if certain commands are unavailable, justify and adjust the verification accordingly.

---

## S04 — Training flow input clarity and validation

**Status:** ⬜ Not started

**Objective:** Evaluate end-to-end usability of the training flow specifically, focusing on clarity of required inputs, validation feedback, and inline guidance for non-experts.

**Actions to perform:**
- Using the running frontend (from S01), walk through submitting a representative training job with sample inputs (real or mock paths as allowed).
- Capture screenshots of each critical step in the training flow (file selection, parameter tuning, submission confirmation, and immediate post-submit state). Save under `docs/screenshots/S04/` with descriptive names.
- Note friction points: unclear placeholders, missing validation, confusing error messages, or non-intuitive parameter labels for training inputs.
- Propose concrete UX improvements for training (tooltips, helper text, presets) in the notes.

**Code/doc pointers:** `ui/frontend/src/components/TrainingPairForm.tsx` and `ModelTrainingForm.tsx` (training UI), `ui/frontend/src/components/FilePathPicker.tsx` (path validation UX), `ui/backend/pipelines.py` (training submission API), `ui/backend/api/routes/files.py` (allowlisted paths), `FRONTEND.md` (training instructions).

**Deliverables:**
- Screenshots under `docs/screenshots/S04/` covering the training interaction flow.
- `docs/notes/training_flow.md` summarizing observations and recommended UX adjustments for training.

**Verification test:**
- **Name:** Training flow documented
- **Commands:**

```text
test -d docs/screenshots/S04
test -f docs/notes/training_flow.md
find docs/screenshots/S04 -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Screenshot directory exists and contains at least one file
  - Training flow notes exist
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Include any console/network errors encountered while submitting training jobs and suggested mitigations.

---

## S05 — Inference flow guidance and presets

**Status:** ⬜ Not started

**Objective:** Evaluate the inference flow for clarity of required inputs, preset usefulness, and non-expert guidance.

**Actions to perform:**
- Using the running frontend (from S01), walk through submitting a representative inference job with sample inputs.
- Capture screenshots of each critical step in the inference flow (file selection, parameter overrides, submission confirmation, immediate post-submit state). Save under `docs/screenshots/S05/` with descriptive names.
- Assess the clarity and defaults of inference parameters (e.g., diarization toggles, slider presets) and note where presets or explanations would help.
- Propose UX improvements tailored to inference (tooltips, presets, warnings for unsafe combinations).

**Code/doc pointers:** `ui/frontend/src/components/InferenceForm.tsx` (inference UI), `ui/frontend/src/components/FilePathPicker.tsx` (path validation), `ui/backend/pipelines.py` (inference submission API), `ui/backend/config_service.py` (config overrides), `FRONTEND.md` (inference usage notes).

**Deliverables:**
- Screenshots under `docs/screenshots/S05/` covering the inference interaction flow.
- `docs/notes/inference_flow.md` summarizing observations and recommended UX adjustments for inference.

**Verification test:**
- **Name:** Inference flow documented
- **Commands:**

```text
test -d docs/screenshots/S05
test -f docs/notes/inference_flow.md
find docs/screenshots/S05 -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Screenshot directory exists and has at least one file
  - Inference flow notes exist
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Capture any console/network errors specific to inference submission and proposed mitigations.

---

## S06 — Submission feedback, validation errors, and recovery

**Status:** ⬜ Not started

**Objective:** Assess how the UI communicates submission success/failure and validation issues, ensuring users can recover quickly.

**Actions to perform:**
- Trigger both successful and intentionally invalid submissions (for training and inference) to observe validation and error states.
- Capture screenshots of validation errors, toasts/banners, and retry/recovery affordances. Save under `docs/screenshots/S06/`.
- Map which fields lack inline validation or have unclear error text, and propose concrete fixes.
- Note whether submissions indicate backend processing status clearly (e.g., spinner vs. silent fail).

**Code/doc pointers:** `ui/frontend/src/components/JobBoard.tsx` (submission feedback and job status), `ui/frontend/src/components/common` (toast/alert components, if present), `ui/backend/pipelines.py` (error surfaces from job creation), `ui/backend/app.py` (FastAPI error handling), `ui/backend/api/routes/files.py` (validation responses).

**Deliverables:**
- Screenshots under `docs/screenshots/S06/` illustrating validation and error handling states.
- `docs/notes/submission_feedback.md` documenting current feedback patterns and recommended improvements.

**Verification test:**
- **Name:** Submission feedback documented

**Commands:**

```text
test -d docs/screenshots/S06
test -f docs/notes/submission_feedback.md
find docs/screenshots/S06 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot directory exists and contains at least one file
  - Submission feedback notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Highlight any missing retry paths or unhelpful error text and suggest concrete wording or UX patterns.

---

## S07 — Job monitoring baseline and navigation

**Status:** ⬜ Not started

**Objective:** Review the job board/monitor for usability, navigation, and clarity of statuses across both training and inference jobs.

**Actions to perform:**
- Navigate to job listings and detail views for at least one training and one inference job (mock or real where possible).
- Capture screenshots of list and detail views, including status chips, timestamps, and navigation controls. Save under `docs/screenshots/S07/`.
- Note any confusing labels, missing timestamps, or pagination/filter gaps.
- Propose improvements for distinguishing job types and surfacing progress.

**Code/doc pointers:** `ui/frontend/src/components/JobBoard.tsx` (job list/detail UI), `ui/backend/pipelines.py` (job status payloads), `ui/backend/config_service.py` (override status merging), `ui_data/jobs/<id>/` (job workspace layout), `README.md` (UI job flow overview).

**Deliverables:**
- Screenshots under `docs/screenshots/S07/` covering job list and detail navigation.
- `docs/notes/job_monitoring.md` summarizing navigation flow, status clarity, and proposed UX adjustments.

**Verification test:**
- **Name:** Job monitoring documented

**Commands:**

```text
test -d docs/screenshots/S07
test -f docs/notes/job_monitoring.md
find docs/screenshots/S07 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot directory exists and contains at least one file
  - Job monitoring notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Record any missing differentiation between training vs. inference jobs and how filters/sorting could help.

---

## S08 — Artifact visibility and preview widgets

**Status:** ⬜ Not started

**Objective:** Audit how artifacts (SRT, enriched JSON, logs, waveform cues) are exposed and previewed in the UI, and identify quick wins for visual validation.

**Actions to perform:**
- Inspect job detail views for artifact links/previews and note what file types are surfaced versus hidden.
- Capture screenshots of current artifact presentation (or absence) under `docs/screenshots/S08/`.
- Identify missing preview widgets (cue tables, waveform snippets, alignment diffs) and map them to available data sources.
- Propose a prioritized list of preview components to implement, with data availability notes.

**Code/doc pointers:** `ui/frontend/src/components/JobBoard.tsx` (artifact display), `ui_data/jobs/<id>/artifacts/` (artifact layout), `ui/backend/pipelines.py` (job result payload), `README.md` (artifact descriptions), `align_make.py` / `build_training_pair_standalone.py` (artifact generation paths).

**Deliverables:**
- Screenshots under `docs/screenshots/S08/` showcasing artifact visibility.
- `docs/notes/artifact_visibility.md` detailing current coverage, gaps, and proposed preview widgets.

**Verification test:**
- **Name:** Artifact visibility baseline captured

**Commands:**

```text
test -d docs/screenshots/S08
test -f docs/notes/artifact_visibility.md
find docs/screenshots/S08 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot directory exists and contains at least one file
  - Artifact visibility notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Call out which artifacts already exist in `ui_data/jobs/<id>/artifacts` and how they could be previewed safely in-browser.

---

## S09 — Training alignment visualization design (SRT ↔ ASR/NW)

**Status:** ⬜ Not started

**Objective:** Define a synchronized, side-by-side visualization that compares the human-edited SRT cues to WhisperX ASR word-level timestamps aligned via Needleman–Wunsch during training, highlighting matches, insertions, and timing deltas.

**Actions to perform:**
- Inventory available training artifacts for alignment: `.train.words.json`, `.asr.visual.words.diar.json`, and any intermediate alignment matrices produced during Stage 2.
- Specify the UI layout for side-by-side cue vs. word-level timelines (e.g., dual columns with synchronized scrolling, match/gap coloring, hover tooltips for timestamps and diffs).
- Identify interaction affordances (scrub/playback hooks if audio is available, filtering by cue, toggling diarization/speaker labels).
- Capture wireframe or mock screenshot(s) illustrating the proposed training alignment view and save under `docs/screenshots/S09/`.
- Document data-loading strategy (which endpoint or artifact path) and performance considerations for large files.

**Code/doc pointers:** `build_training_pair_standalone.py` (Needleman–Wunsch alignment and training artifacts), `ui/backend/pipelines.py` (artifact copy into `ui_data/jobs/<id>`), `ui/frontend/src/components/JobBoard.tsx` (job details hook), `docs/build_training_pair_comparison.md` (Stage 2 rationale), `README.md` (artifact descriptions).

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S09/` showing the proposed training alignment visualization.
- `docs/notes/training_alignment_design.md` describing layout, interactions, data sources, and fallbacks.

**Verification test:**
- **Name:** Training alignment design recorded

**Commands:**

```text
test -d docs/screenshots/S09
test -f docs/notes/training_alignment_design.md
find docs/screenshots/S09 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot/mockup directory exists and contains at least one file
  - Training alignment design notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot/mockup file is present.

**Notes:**
- Explicitly call out how Needleman–Wunsch alignment output (matches, insertions, deletions) will be visualized and kept synchronized with cue timing.
- Prefer designs that reuse existing artifact formats to avoid new backend dependencies unless justified.

---

## S10 — Inference alignment visualization design (LLM ↔ ASR/NW)

**Status:** ⬜ Not started

**Objective:** Plan a synchronized visualization for inference that compares the LLM-edited/refined transcript to WhisperX ASR word-level timestamps after Needleman–Wunsch alignment, emphasizing timing quality and structural cues.

**Actions to perform:**
- Catalog inference-time artifacts that hold aligned word-level timestamps (e.g., enriched JSON with `is_llm_structural_break`, ASR reference files, alignment outputs) and note their locations per job.
- Define the UI for side-by-side comparison (LLM text vs. ASR words) with synchronized scrolling, highlighting of alignment matches/mismatches, and visual indicators for timing gaps or uncertain alignments.
- Specify how to surface cue-level timing deltas, confidence/score overlays if available, and toggles for showing LLM structural break hints.
- Capture wireframe or mock screenshot(s) of the proposed inference alignment view and save under `docs/screenshots/S10/`.
- Note any shared components with the training alignment view to maximize reuse and ensure consistent UX.

**Code/doc pointers:** `align_make.py` (ASR output), `build_training_pair_standalone.py` (inference alignment logic and enriched JSON fields), `ui/backend/pipelines.py` (job artifact handling), `ui/frontend/src/components/JobBoard.tsx` and planned preview widgets (artifact consumption), `docs/beam_search_walkthrough.md` (timing context), `README.md` (inference flow overview).

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S10/` for the inference alignment visualization.
- `docs/notes/inference_alignment_design.md` detailing layout, interactions, data sources, and reuse strategy relative to training.

**Verification test:**
- **Name:** Inference alignment design recorded

**Commands:**

```text
test -d docs/screenshots/S10
test -f docs/notes/inference_alignment_design.md
find docs/screenshots/S10 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot/mockup directory exists and contains at least one file
  - Inference alignment design notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot/mockup file is present.

**Notes:**
- Document how Needleman–Wunsch alignment results will be kept in sync across text panes and any fallback behavior if alignment data is incomplete.
- Consider visual consistency with training alignment while accounting for LLM-specific metadata (e.g., `is_llm_structural_break`).

---

## S11 — System health signals and observability hooks

**Status:** ⬜ Not started

**Objective:** Assess how the UI and backend expose operational health (statuses, backend errors, resource warnings) and define the signals needed for trustworthy usage.

**Actions to perform:**
- Review UI components for health indicators (status chips, banners, spinners) and note when they appear or fail to appear.
- Capture screenshots of health/alert states under `docs/screenshots/S11/`.
- Inspect available backend endpoints/log surfaces (FastAPI) that could provide health data and note gaps.
- Propose specific observability hooks (heartbeat endpoint checks, queue depth indicators, error surfacing) to add.

**Code/doc pointers:** `ui/frontend/src/components/JobBoard.tsx` (job status rendering), `ui/backend/app.py` (FastAPI setup), `ui/backend/pipelines.py` (job lifecycle statuses), `ui/backend/api/routes/files.py` (file validation errors), `README.md` (operations notes).

**Deliverables:**
- Screenshots under `docs/screenshots/S11/` covering current health signals.
- `docs/notes/system_health.md` detailing observed signals, gaps, and recommended observability hooks.

**Verification test:**
- **Name:** System health baseline captured

**Commands:**

```text
test -d docs/screenshots/S11
test -f docs/notes/system_health.md
find docs/screenshots/S11 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot directory exists and has at least one file
  - System health notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Tie proposed hooks to concrete endpoints or UI surfaces for later implementation and testing.

---

## S12 — Insight verification and test harness planning

**Status:** ⬜ Not started

**Objective:** Define the test strategy and harnesses needed to verify artifact previews and health signals once implemented.

**Actions to perform:**
- Enumerate automated and manual tests required for the planned preview widgets and health indicators (e.g., component tests, API contract checks, e2e smoke tests).
- Identify sample data or fixtures needed to exercise previews (SRT, enriched JSON, alignment outputs) and where to store them.
- Outline commands to be used later (frontend unit tests, backend API tests, screenshot-based visual checks) and how to record evidence.
- Capture a summary screenshot (e.g., checklist or doc snippet) under `docs/screenshots/S12/` showing the planned test matrix or harness layout.

**Code/doc pointers:** `tests/` (existing Python tests), `ui/frontend/package.json` (test scripts), `ui/frontend/src` (components targeted for unit tests), `ui/backend/pipelines.py` and `ui/backend/config_service.py` (API contract surfaces), `docs/ENTRYPOINTS.md` (CLI references for fixtures).

**Deliverables:**
- Screenshots under `docs/screenshots/S12/` illustrating the planned test matrix or harness notes.
- `docs/notes/insight_verification_plan.md` describing test coverage, fixtures, and tooling to be used in later steps.

**Verification test:**
- **Name:** Insight verification plan documented

**Commands:**

```text
test -d docs/screenshots/S12
test -f docs/notes/insight_verification_plan.md
find docs/screenshots/S12 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot directory exists and contains at least one file
  - Insight verification plan notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Explicitly call out which tests are required versus optional, and how results will be captured for future steps.

---

## S13 — Guided job templates and presets

**Status:** ⬜ Not started

**Objective:** Define a template-driven flow that lets users pick common scenarios (e.g., short clip inference, long-form training refresh, alignment QA) and auto-populates safe defaults, with the ability to save/load custom presets.

**Actions to perform:**
- Review training and inference forms to identify fields suitable for templating (file pickers, diarization toggles, scoring overrides, reflow options) and map them to preset values.
- Draft UX for selecting a template at form entry (dropdown or wizard landing) and for saving/updating user presets; capture mockups under `docs/screenshots/S13/`.
- Specify validation and guardrails for templates (e.g., warn when presets reference missing paths or incompatible flags) and how to surface them inline.
- Outline how templates will be persisted (local storage vs. backend) and synchronized across sessions, noting any security constraints.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S13/` showing the proposed template/preset selector and save/update flows.
- `docs/notes/job_templates_design.md` describing template schemas, default bundles, validation rules, and persistence approach.

**Verification test:**
- **Name:** Job templates design documented
- **Commands:**

```text
test -d docs/screenshots/S13
test -f docs/notes/job_templates_design.md
find docs/screenshots/S13 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot/mockup directory exists and contains at least one file
  - Job templates design notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot/mockup file is present.

**Notes:**
- Include copy/wording guidance for presets so non-experts understand when to choose each template and how to revert to defaults.

---

## S14 — Data quality dashboard and cue diagnostics

**Status:** ⬜ Not started

**Objective:** Plan a data quality view that surfaces interpretable metrics from training and inference artifacts (e.g., diarization consistency, pause distributions, alignment gaps, cue length outliers) to help non-experts judge readiness and spot issues quickly.

**Actions to perform:**
- Catalog available metrics/features in emitted artifacts (e.g., pause_ms, speaker_change flags, alignment gap markers, structural hints) and decide which to visualize for quality assessment.
- Design dashboard panels (charts/tables) for per-job summaries and per-cue drilldowns; capture mockups under `docs/screenshots/S14/`.
- Define thresholds and highlighting rules for common problems (overlong cues, dense speaker changes, large alignment gaps) and how to present remediation tips inline.
- Outline data retrieval strategy (which endpoints/artifact files) and performance considerations for large jobs.

**Code/doc pointers:** `build_training_pair_standalone.py` (pause_ms, speaker_change features), `isce/scorer.py` (structural heuristics), `ui/backend/pipelines.py` (artifact delivery), `ui/frontend/src/components/JobBoard.tsx` and planned dashboard components (data consumption), `docs/beam_search_walkthrough.md` (scoring context), `README.md` (artifact descriptions).

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S14/` showing the proposed data quality dashboard and cue diagnostics views.
- `docs/notes/data_quality_dashboard.md` detailing metrics, thresholds, visualization types, and data loading approach.

**Verification test:**
- **Name:** Data quality dashboard design recorded

- **Commands:**

```text
test -d docs/screenshots/S14
test -f docs/notes/data_quality_dashboard.md
find docs/screenshots/S14 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot/mockup directory exists and contains at least one file
  - Data quality dashboard notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot/mockup file is present.

**Notes:**
- Emphasize plain-language labels and legends so average users can interpret metrics without domain knowledge; include suggestions for tooltips and “what this means” text.

---

## S15 — Embedded help center and onboarding tours

**Status:** ⬜ Not started

**Objective:** Define an in-app help center with glossary, quickstart checklists, and guided tours that orient new users across training, inference, and monitoring views without leaving the UI.

**Actions to perform:**
- Identify the top questions/confusions from prior steps (training/inference inputs, alignment visuals, artifact previews) and draft concise answers/glossary entries.
- Design entry points for help (e.g., persistent “Help” button, contextual “?” icons) and guided tour steps; capture mockups under `docs/screenshots/S15/`.
- Map help content to existing docs (README, FRONTEND.md) and propose inline anchors or embedded markdown rendering.
- Specify how tours will be triggered/dismissed, saved per user/session, and localized if needed.

**Code/doc pointers:** `FRONTEND.md` and `README.md` (user-facing docs for inline linking), `ui/frontend/src/App.tsx` / `Navigation` components (insertion points for help/tours), `ui/frontend/src/components` (modal/overlay primitives if present), `ui/frontend/src/hooks` (state management hooks, if available) for storing tour state.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S15/` showing the help center, glossary, and guided tour overlays.
- `docs/notes/help_center_plan.md` describing content structure, triggering logic, tour steps, and documentation links.

**Verification test:**
- **Name:** Help center plan documented

- **Commands:**

```text
test -d docs/screenshots/S15
test -f docs/notes/help_center_plan.md
find docs/screenshots/S15 -maxdepth 1 -type f | head -n 1
```

**Expected results:**
  - Screenshot/mockup directory exists and contains at least one file
  - Help center plan notes exist

**Pass criteria:** All commands exit with code 0 AND at least one screenshot/mockup file is present.

**Notes:**
- Include accessibility considerations (keyboard navigation, ARIA labels) for all help/tour entry points and overlays.
