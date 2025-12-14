# ISCE Caption Pipeline Frontend Update Plan — Work-in-Progress

**Plan ID:** `isce_caption_pipeline_frontend_update_v1`
**Repository:** `floke75/ISCE_Caption_Pipeline`
**Plan created (UTC):** `2025-07-05T00:00:00Z`

**Step status summary:** 0/13 passed, 0 failed, 0 in progress (initial draft)

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

---

## S00 — Establish current frontend state and data collection protocol

**Status:** ⬜ Not started

**Objective:** Confirm environment readiness for frontend review and define how screenshots and artifacts will be captured and stored throughout the plan.

**Actions to perform:**
- Verify frontend install/build commands in `FRONTEND.md` and ensure npm/yarn availability.
- Create `docs/screenshots/S00/` and `docs/notes/` directories if missing.
- Record the baseline commands for running the frontend locally (dev mode) and for building static assets.
- Outline the standard screenshot capture process (naming, resolution, where to store) and note any tool limitations in this environment.

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
