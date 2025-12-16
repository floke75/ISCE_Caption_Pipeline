# ISCE Caption Pipeline Frontend Update Plan — Work-in-Progress

**Plan ID:** `isce_caption_pipeline_frontend_update_v1`
**Repository:** `floke75/ISCE_Caption_Pipeline`
**Plan created (UTC):** `2025-07-05T00:00:00Z`

**Step status summary:** 17/26 passed, 0 failed, 0 in progress

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
- Update the status markers (in the index and per-step header) immediately after each step, and include the executed command
  outputs in Notes to preserve traceability.
- **Tests are integral for code changes:**
  - Every code-affecting step must specify at least one verification command (unit test, lint, type check, or script) OR justify why no executable test is applicable.
  - Record actual command outputs in Notes when executed; passing tests are required before marking a step as ✅.
- **Screenshots are mandatory for UI-affecting steps:**
  - Capture at least one screenshot of the relevant UI after completing the step for both LLM-agent verification and human review.
  - Store screenshots under `docs/screenshots/<step_id>/` with descriptive filenames.
  - Reference the screenshot path in the step's Notes once captured.
- **Milestone gates:** Treat S03 (build/lint baseline) and S08 (artifact visibility baseline) as go/no-go checkpoints before
  shipping UX changes or visualization work; unresolved blockers at these steps must be documented with mitigation plans.

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

### Phase ordering (do not reorder)

- **Baseline & readiness:** S00–S03 (must be green before altering UX components).
- **Core flows & feedback:** S04–S08 (captures training/inference UX, error handling, monitoring, and artifact surfacing).
- **Alignment & insights:** S09–S12 (alignment visuals, health signals, test harness definitions).
- **Guidance & acceleration:** S13–S15 (templates/presets, quality dashboards, embedded help).

---

## Steps

### Index

- [S00 — Establish current frontend state and data collection protocol](#s00-establish-current-frontend-state-and-data-collection-protocol) — ✅ Passed
- [S01 — Baseline UI capture and navigation audit](#s01-baseline-ui-capture-and-navigation-audit) — ✅ Passed
- [S02 — Document existing user guidance and gaps](#s02-document-existing-user-guidance-and-gaps) — ✅ Passed
- [S03 — Frontend build, lint, and dependency baseline](#s03-frontend-build-lint-and-dependency-baseline) — ✅ Passed
- [S04 — Training flow input clarity and validation](#s04-training-flow-input-clarity-and-validation) — ✅ Passed
- [S05 — Inference flow guidance and presets](#s05-inference-flow-guidance-and-presets) — ✅ Passed
- [S05b — Implement inference flow improvements](#s05b-implement-inference-flow-improvements) — ✅ Passed
- [S06 — Submission feedback analysis and design](#s06-submission-feedback-analysis-and-design) — ✅ Passed
- [S06b — Implement submission feedback improvements](#s06b-implement-submission-feedback-improvements) — ✅ Passed
- [S07 — Job monitoring analysis and design](#s07-job-monitoring-analysis-and-design) — ✅ Passed
- [S07b — Implement job monitoring improvements](#s07b-implement-job-monitoring-improvements) — ✅ Passed
- [S08 — Artifact visibility analysis and design](#s08-artifact-visibility-analysis-and-design) — ✅ Passed
- [S08b — Implement artifact visibility improvements](#s08b-implement-artifact-visibility-improvements) — ✅ Passed
- [S09 — Training alignment visualization design](#s09-training-alignment-visualization-design) — ✅ Passed
- [S09b — Implement training alignment visualization](#s09b-implement-training-alignment-visualization) — ✅ Passed
- [S10 — Inference alignment visualization design](#s10-inference-alignment-visualization-design) — ✅ Passed
- [S10b — Implement inference alignment visualization](#s10b-implement-inference-alignment-visualization) — ✅ Passed
- [S11 — System health analysis and design](#s11-system-health-analysis-and-design) — ✅ Passed
- [S11b — Implement system health signals](#s11b-implement-system-health-signals) — ✅ Passed
- [S12 — Insight verification infrastructure](#s12-insight-verification-infrastructure) — ⬜ Not started
- [S13 — Guided job templates design](#s13-guided-job-templates-design) — ⬜ Not started
- [S13b — Implement job templates](#s13b-implement-job-templates) — ⬜ Not started
- [S14 — Data quality dashboard design](#s14-data-quality-dashboard-design) — ⬜ Not started
- [S14b — Implement data quality dashboard](#s14b-implement-data-quality-dashboard) — ⬜ Not started
- [S15 — Embedded help center design](#s15-embedded-help-center-design) — ⬜ Not started
- [S15b — Implement help center](#s15b-implement-help-center) — ⬜ Not started

---

## S00 — Establish current frontend state and data collection protocol

**Status:** ✅ Passed

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
- Environment commands executed: `node -v` → `v20.19.5`; `npm -v` → `11.4.2` (with warning about unknown env config `http-proxy`).
- Directories `docs/notes/` and `docs/screenshots/S00/` created for plan artifacts.

---

## S01 — Baseline UI capture and navigation audit

**Status:** ✅ Passed

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
- Screenshots captured: `inference-form.png`, `training-pair-form.png`, `model-training-form.png`, `config-panel.png` in `docs/screenshots/S01/`.
- `docs/notes/ui_baseline.md` created with navigation details.
- Environment: Node v20.19.5, Vite v5.4.21, Backend running on port 8000.
- Playwright script `capture_baseline.py` used for capture and then removed.

---

## S02 — Document existing user guidance and gaps

**Status:** ✅ Passed

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
- Screenshots captured: `inference_guidance.png`, `training_pair_guidance.png`, `model_training_guidance.png`, `config_panel_guidance.png`.
- `docs/notes/ui_guidance_gaps.md` created listing prioritized gaps.
- Playwright script `capture_guidance.py` used for capture and removed.
- Verification passed: 4 screenshots found and gap analysis document exists.

---

## S03 — Frontend build, lint, and dependency baseline

**Status:** ✅ Passed

**Objective:** Establish a robust reliability baseline by installing standard tooling (`eslint`, `vitest`) and **fixing all existing lint errors** to ensure a clean slate.

**Actions to perform:**
- Install `eslint` and `vitest` (already done).
- Run `npm run lint` to confirm error list.
- Fix `no-unused-vars` and `no-explicit-any` issues across components.
- Fix `react-hooks/set-state-in-effect` and `exhaustive-deps` issues in `ConfigPanel`, `OverrideEditor`, and `JobBoard`.
- Fix `react-hooks/immutability` / declaration hoisting in `useEventStream.ts`.
- Verify `npm run lint` passes with 0 warnings/errors.
- Verify `npm test` and `npm run build` still pass.
- Update `docs/notes/frontend_reliability.md` to reflect a clean baseline.

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
cd ui/frontend && npm test -- --runInBand || npm run test:unit
```
- **Expected results:**
  - Dependency install succeeds
  - Build completes without errors
  - Lint passes (or explicitly documents lint gaps if command is unavailable)
  - At least one automated test command runs; if no test script exists, note the absence explicitly in the reliability notes
- **Pass criteria:** All commands exit with code 0 OR documented blockers with proposed mitigation in `docs/notes/frontend_reliability.md` (including rationale when tests are unavailable or intentionally skipped).

**Notes:**
- Record actual command outputs and durations in the notes; if certain commands are unavailable, justify and adjust the verification accordingly.

---

## S04 — Training flow input clarity and validation

**Status:** ✅ Passed

**Objective:** Finalize the help text and placeholders in the training forms to be fully accurate and relevant, removing any generic or placeholder content.

**Actions to perform:**
- Research the actual meaning/impact of `Iterations` and `Error boost factor` in the codebase.
- Research the purpose of "Operator notes".
- Update `ModelTrainingForm.tsx`:
  - Replace generic help text with precise technical explanations (e.g., explaining EM-style reweighting or penalty multipliers).
  - Ensure placeholders reflect realistic defaults.
- Update `TrainingPairForm.tsx`:
  - Clarify "Operator notes" usage.
- Verify changes with `scripts/verify_s04_improved.py` (capture new screenshots).
- Update `docs/notes/training_flow.md` with the final text used.

**Code/doc pointers:** `ui/frontend/src/components/TrainingPairForm.tsx` and `ModelTrainingForm.tsx` (training UI), `ui/frontend/src/components/FilePathPicker.tsx` (path validation UX), `ui/backend/pipelines.py` (training submission API), `ui/backend/api/routes/files.py` (allowlisted paths), `FRONTEND.md` (training instructions).

**Deliverables:**
- Improved component code.
- Updated screenshots under `docs/screenshots/S04/`.
- `docs/notes/training_flow.md` updated with implementation details.

**Verification test:**
- **Name:** Training flow improvements verified
- **Commands:**

```text
test -d docs/screenshots/S04
test -f docs/notes/training_flow.md
grep "Implemented" docs/notes/training_flow.md
```
- **Expected results:**
  - Screenshot directory exists and contains at least one file
  - Training flow notes exist
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Include any console/network errors encountered while submitting training jobs and suggested mitigations.

---

## S05 — Inference flow guidance and presets

**Status:** ✅ Passed

**Objective:** Evaluate the inference flow for clarity of required inputs, preset usefulness, and non-expert guidance.

**Actions to perform:**
- Using the running frontend (from S01), walk through submitting a representative inference job with sample inputs.
- Capture screenshots of each critical step in the inference flow (file selection, parameter overrides, submission confirmation, immediate post-submit state). Save under `docs/screenshots/S05/` with descriptive names.
- Assess the clarity and defaults of inference parameters (e.g., diarization toggles, slider presets) and note where presets or explanations would help.
- Propose UX improvements tailored to inference (tooltips, presets, warnings for unsafe combinations).

**Code/doc pointers:** `ui/frontend/src/components/InferenceForm.tsx` (inference UI), `ui/frontend/src/components/FilePathPicker.tsx` (path validation), `ui/backend/pipelines.py` (inference submission API), `ui/backend/config_service.py` (config overrides), `FRONTEND.md` (inference usage notes).

**Deliverables:**
- Screenshots under `docs/screenshots/S05/` covering the inference interaction flow.
- `docs/notes/inference_flow.md` summarizing observations and recommended UX adjustments for inference, including any job IDs or
  mock submissions used during the walkthrough.

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
- Screenshots captured: `inference_empty.png`, `inference_filled.png`, `inference_submitted.png`.
- `docs/notes/inference_flow.md` documents pain points (hidden overrides, lack of presets) and improvement plan.
- Playwright script used network interception to mock job submission and file validation.

---

## S05b — Implement inference flow improvements

**Status:** ✅ Passed

**Objective:** Implement the high-priority UX improvements identified in S05: surface critical toggles (diarization, beam width), add basic quality presets, and improve field guidance.

**Actions to perform:**
- Update `InferenceForm.tsx` to include a "Preset" selector (Standard, High Precision, Fast Draft) that auto-populates overrides.
- Expose `do_diarization` (checkbox) and `num_beams` (number/slider) as top-level controls, syncing them with the override state.
- Refine tooltips/helper text for "Transcript" and "Model config" to clarify their purpose.
- Ensure "Output directory" validation is friendly.

**Code/doc pointers:** `ui/frontend/src/components/InferenceForm.tsx`, `ui/frontend/src/types.ts` (if new types needed).

**Deliverables:**
- Updated `InferenceForm.tsx`.
- Screenshots under `docs/screenshots/S05b/` showing the new form elements and preset interactions.

**Verification test:**
- **Name:** Inference improvements verified
- **Commands:**

```text
test -d docs/screenshots/S05b
find docs/screenshots/S05b -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Screenshot directory exists and has at least one file
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Refactored `OverrideEditor` to support controlled mode (props-driven edits).
- Added Preset, Beam Width, and Diarization controls to `InferenceForm` that sync bidirectionally with overrides.
- Verified interactions and state sync with Playwright.

---

## S06 — Submission feedback analysis and design

**Status:** ✅ Passed

**Objective:** Assess how the UI communicates submission success/failure and validation issues, ensuring users can recover quickly.

**Actions to perform:**
- Trigger both successful and intentionally invalid submissions (for training and inference) to observe validation and error states.
- Capture screenshots of validation errors, toasts/banners, and retry/recovery affordances. Save under `docs/screenshots/S06/`.
- Map which fields lack inline validation or have unclear error text, and propose concrete fixes.
- Note whether submissions indicate backend processing status clearly (e.g., spinner vs. silent fail).

**Deliverables:**
- Screenshots under `docs/screenshots/S06/` illustrating validation and error handling states.
- `docs/notes/submission_feedback.md` documenting current feedback patterns, recommended improvements, and any job IDs or
  request payloads used to trigger success/failure states.

**Verification test:**
- **Name:** Submission feedback documented
- **Commands:**

```text
test -d docs/screenshots/S06
test -f docs/notes/submission_feedback.md
find docs/screenshots/S06 -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Execution: Ran `scripts/verify_s06_submission.py` to capture behavior using Playwright network interception.
- Findings: `InferenceForm` relies on click-to-validate (toast), while `TrainingPairForm` disables the submit button until valid. Design document proposes unifying on the "Disabled Button" pattern.
- Screenshots captured in `docs/screenshots/S06/`: `inference_invalid_toast.png`, `training_invalid_disabled.png`, `inference_backend_error.png`, `training_backend_error.png`.

---

## S06b — Implement submission feedback improvements

**Status:** ✅ Passed

**Objective:** Implement the recommended improvements for submission feedback, including better error messages, visual cues for invalid fields, and retry affordances.

**Actions to perform:**
- Update `JobBoard` and form components to display structured error messages from the backend.
- Enhance `FilePathPicker` and other inputs to show error states more clearly (e.g., red borders, inline text).
- Implement loading spinners or progress bars during submission.
- Ensure toast notifications provide actionable details on failure.

**Deliverables:**
- Updated components (`JobBoard`, forms, `FilePathPicker`).
- Screenshots under `docs/screenshots/S06b/` showing improved error states and submission feedback.

**Verification test:**
- **Name:** Feedback improvements verified
- **Commands:**

```text
test -d docs/screenshots/S06b
find docs/screenshots/S06b -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Implemented "Disabled Button" pattern in `InferenceForm.tsx` to match other forms.
- Verified that `InferenceForm` now disables the submit button when valid and enables it when inputs are valid (mocked).
- Confirmed that `FilePathPicker` styling for `.invalid` uses distinct red color (`#f87171`).
- Captured screenshots in `docs/screenshots/S06b/`.
- `JobBoard` already supports displaying error details from the backend.

---

## S07 — Job monitoring analysis and design

**Status:** ✅ Passed

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
- **Commands:**

```text
test -d docs/screenshots/S07
test -f docs/notes/job_monitoring.md
find docs/screenshots/S07 -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Execution: Ran `scripts/verify_s07_monitoring.py` to inject diverse mock jobs (Running, Failed, Success).
- Findings: Job list is functional but lacks visual distinction between job types (text only). Timestamps lack absolute tooltips. Error display is good.
- Design: Propose adding Type Icons (Film, Database, Brain) and Status Icons (Check, X, Clock) in S07b.
- Screenshots captured in `docs/screenshots/S07/`.

---

## S07b — Implement job monitoring improvements

**Status:** ✅ Passed

**Objective:** Implement improvements to the job monitor, such as distinct icons for job types, relative timestamps, and status filters.

**Actions to perform:**
- Update `JobBoard` to show distinct icons/badges for Training vs Inference jobs.
- Format timestamps (created/completed) to be relative (e.g., "5 mins ago") with absolute tooltip.
- Add basic filtering by status (Pending, Running, Completed, Failed).
- Improve status chip styling for better readability.

**Deliverables:**
- Updated `JobBoard` component.
- Screenshots under `docs/screenshots/S07b/` showing the enhanced job list.

**Verification test:**
- **Name:** Job monitoring improvements verified
- **Commands:**

```text
test -d docs/screenshots/S07b
find docs/screenshots/S07b -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Implemented `JobTypeIcon` (visual distinction for Inference vs Training) and `StatusIcon` (Check/X/Clock) in `JobBoard.tsx`.
- Added a status filter dropdown (`All`, `Pending`, `Running`, `Succeeded`, `Failed`) to the header.
- Added absolute timestamp tooltips to relative time displays using the `title` attribute.
- Verified changes with `scripts/verify_s07b_improvements.py`, which mocked backend jobs to test filtering and icon rendering.
- Screenshots captured in `docs/screenshots/S07b/`: `job_list_icons.png`, `job_list_filtered_failed.png`.

---

## S08 — Artifact visibility analysis and design

**Status:** ✅ Passed

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
- **Commands:**

```text
test -d docs/screenshots/S08
test -f docs/notes/artifact_visibility.md
find docs/screenshots/S08 -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Documented the existing artifact surface gaps and proposed `/files/download` + `/files/content` endpoints in `docs/notes/artifact_visibility.md`.
- Captured the baseline Job Details artifact presentation in `docs/screenshots/S08/baseline_job_details.png`.
- Verification checklist satisfied using the commands above.

---

## S08b — Implement artifact visibility improvements

**Status:** ✅ Passed

**Objective:** Implement links and basic previews for key artifacts (SRT, logs) directly in the job details view.

**Actions to perform:**
- Update `JobBoard` detail view to list all generated artifacts with download links.
- Add a text/code viewer component for `.srt` and `.json` files.
- Ensure log streams are visible and auto-scroll correctly.

**Deliverables:**
- Updated `JobBoard` and new artifact viewer components.
- Screenshots under `docs/screenshots/S08b/` showing artifact links and previews.

**Verification test:**
- **Name:** Artifact visibility improvements verified
- **Commands:**

```text
test -d docs/screenshots/S08b
find docs/screenshots/S08b -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Added Artifact Viewer route and linkified artifact paths in the Job Board; backend `/files/content` and `/files/download` endpoints enable previews and downloads.
- Screenshots captured in `docs/screenshots/S08b/` (`artifact_viewer.png`, `job_details_with_link.png`, and error coverage).
- Verified presence of screenshots with the listed commands (dev session captured in `frontend_s08b.log`).

---

## S09 — Training alignment visualization design

**Status:** ✅ Passed

**Objective:** Define a synchronized, side-by-side visualization that compares the human-edited SRT cues to WhisperX ASR word-level timestamps aligned via Needleman–Wunsch during training.

**Actions to perform:**
- Inventory available training artifacts for alignment: `.train.words.json`, `.asr.visual.words.diar.json`.
- Specify the UI layout for side-by-side cue vs. word-level timelines.
- Identify interaction affordances (scrub/playback hooks if audio is available, filtering by cue).
- Capture wireframe or mock screenshot(s) illustrating the proposed training alignment view.
- Document data-loading strategy.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S09/` showing the proposed training alignment visualization.
- `docs/notes/training_alignment_design.md` describing layout, interactions, data sources, and fallbacks.

**Verification test:**
- **Name:** Training alignment design recorded
- **Commands:**

```text
test -d docs/screenshots/S09
test -f docs/notes/training_alignment_design.md
find docs/screenshots/S09 -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Defined the teleprompter-style two-column layout and interaction model in `docs/notes/training_alignment_design.md` (revised v2).
- Captured mockups and error states in `docs/screenshots/S09/`.
- Verified deliverables with the checklist commands above.

---

## S09b — Implement training alignment visualization

**Status:** ✅ Passed

**Objective:** Implement the training alignment visualization, rendering synchronized views of SRT cues and ASR words.

**Actions to perform:**
- Create a new visualization component (e.g., `AlignmentViewer`).
- Implement logic to load `.train.words.json` and `.asr.visual.words.diar.json`.
- Render side-by-side columns with alignment connections or color-coding.
- Integrate into the Job Details view for training jobs.

**Deliverables:**
- `AlignmentViewer` component.
- Screenshots under `docs/screenshots/S09b/` showing the visualization with sample data.

**Verification test:**
- **Name:** Training alignment visualization verified
- **Commands:**

```text
test -d docs/screenshots/S09b
find docs/screenshots/S09b -maxdepth 1 -type f | head -n 1
```

**Notes:**
- Implemented the teleprompter-style `AlignmentViewer` with timestamp-based positioning for cues and ASR words, linked from the Job Board.
- Recorded the implementation summary in `docs/notes/alignment_viewer_implementation.md` and captured the live view in `docs/screenshots/S09b/alignment_viewer_implementation.png`.
- Verification checklist satisfied using the commands above.

---

## S10 — Inference alignment visualization design

**Status:** ✅ Passed

**Objective:** Plan a synchronized visualization for inference that compares the LLM-edited/refined transcript to WhisperX ASR word-level timestamps after Needleman–Wunsch alignment.

**Actions to perform:**
- Catalog inference-time artifacts that hold aligned word-level timestamps (enriched JSON).
- Define the UI for side-by-side comparison (LLM text vs. ASR words) with synchronized scrolling.
- Specify how to surface cue-level timing deltas and structural hints.
- Capture wireframe or mock screenshot(s).
- Note any shared components with the training alignment view.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S10/` for the inference alignment visualization.
- `docs/notes/inference_alignment_design.md` detailing layout, interactions, data sources, and reuse strategy.

**Verification test:**
- **Name:** Inference alignment design recorded
- **Commands:**

```text
test -d docs/screenshots/S10
test -f docs/notes/inference_alignment_design.md
find docs/screenshots/S10 -maxdepth 1 -type f | head -n 1
```

---

## S10b — Implement inference alignment visualization

**Status:** ✅ Passed

**Objective:** Implement the inference alignment visualization, adapting the `AlignmentViewer` to handle inference artifacts and metadata.

**Actions to perform:**
- Update `AlignmentViewer` to support inference mode (consuming enriched JSON).
- Visualize LLM structural breaks and confidence scores if available.
- Integrate into the Job Details view for inference jobs.

**Deliverables:**
- Updated `AlignmentViewer`.
- Screenshots under `docs/screenshots/S10b/` showing the inference alignment view.

**Verification test:**
- **Name:** Inference alignment visualization verified
- **Commands:**

```text
test -d docs/screenshots/S10b
find docs/screenshots/S10b -maxdepth 1 -type f | head -n 1
```

---

## S11 — System health analysis and design

**Status:** ✅ Passed

**Objective:** Assess how the UI and backend expose operational health and define the signals needed for trustworthy usage.

**Actions to perform:**
- Review UI components for health indicators (status chips, banners, spinners).
- Capture screenshots of health/alert states under `docs/screenshots/S11/`.
- Inspect available backend endpoints/log surfaces.
- Propose specific observability hooks (heartbeat endpoint checks, queue depth indicators).

**Deliverables:**
- Screenshots under `docs/screenshots/S11/` covering current health signals.
- `docs/notes/system_health.md` detailing observed signals, gaps, and recommended observability hooks.

**Verification test:**
- **Name:** System health baseline captured
- **Commands:**

```text
test -d docs/screenshots/S11
test -f docs/notes/system_health.md
find docs/screenshots/S11 -maxdepth 1 -type f | head -n 1
```

---

## S11b — Implement system health signals

**Status:** ✅ Passed

**Objective:** Implement backend health endpoints and UI indicators for system status.

**Actions to perform:**
- Add a `/health` endpoint to the backend (checking disk space, GPU availability, queue size).
- Add a persistent status bar or indicator in the UI header showing backend connectivity and health.
- Display global alerts for system-wide issues (e.g., "Disk full").

**Deliverables:**
- Updated backend API and UI header.
- Screenshots under `docs/screenshots/S11b/` showing health indicators.

**Verification test:**
- **Name:** System health signals verified
- **Commands:**

```text
test -d docs/screenshots/S11b
find docs/screenshots/S11b -maxdepth 1 -type f | head -n 1
```

---

## S12 — Insight verification infrastructure

**Status:** ⬜ Not started

**Objective:** Implement the test harness and fixtures needed to verify complex visualizations and health signals.

**Actions to perform:**
- Create reusable Playwright fixtures for generating mock job states (training, inference, failed).
- Create mock data files for alignment visualizations (SRT, JSON).
- Set up a visual regression testing helper (if not already present) or standardized screenshot comparison workflow.

**Deliverables:**
- Test infrastructure code (`tests/fixtures/`, `ui/frontend/tests/`).
- `docs/notes/insight_verification_infrastructure.md` documenting how to use the harness.

**Verification test:**
- **Name:** Test infrastructure verified
- **Commands:**

```text
test -f docs/notes/insight_verification_infrastructure.md
# Run a dummy test using the new fixtures
# npm test -- --grep "visual regression"
```

---

## S13 — Guided job templates design

**Status:** ⬜ Not started

**Objective:** Define a template-driven flow that lets users pick common scenarios and auto-populates safe defaults.

**Actions to perform:**
- Review training and inference forms to identify fields suitable for templating.
- Draft UX for selecting a template at form entry and for saving/updating user presets.
- Capture mockups under `docs/screenshots/S13/`.
- Specify validation and guardrails for templates.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S13/`.
- `docs/notes/job_templates_design.md` describing template schemas and flows.

**Verification test:**
- **Name:** Job templates design documented
- **Commands:**

```text
test -d docs/screenshots/S13
test -f docs/notes/job_templates_design.md
find docs/screenshots/S13 -maxdepth 1 -type f | head -n 1
```

---

## S13b — Implement job templates

**Status:** ⬜ Not started

**Objective:** Implement the job template system, allowing users to save and load job configurations.

**Actions to perform:**
- Create a `TemplateManager` component or service in the frontend.
- Allow users to "Save as Template" from a filled form.
- Add a "Load Template" selector to Training and Inference forms.
- Persist templates to local storage or backend.

**Deliverables:**
- Updated forms with Template support.
- Screenshots under `docs/screenshots/S13b/` showing template saving/loading.

**Verification test:**
- **Name:** Job templates verified
- **Commands:**

```text
test -d docs/screenshots/S13b
find docs/screenshots/S13b -maxdepth 1 -type f | head -n 1
```

---

## S14 — Data quality dashboard design

**Status:** ⬜ Not started

**Objective:** Plan a data quality view that surfaces interpretable metrics from training and inference artifacts.

**Actions to perform:**
- Catalog available metrics/features (pause_ms, speaker_change).
- Design dashboard panels for per-job summaries and per-cue drilldowns.
- Define thresholds and highlighting rules for common problems.
- Capture mockups under `docs/screenshots/S14/`.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S14/`.
- `docs/notes/data_quality_dashboard.md` detailing metrics and visualizations.

**Verification test:**
- **Name:** Data quality dashboard design recorded
- **Commands:**

```text
test -d docs/screenshots/S14
test -f docs/notes/data_quality_dashboard.md
find docs/screenshots/S14 -maxdepth 1 -type f | head -n 1
```

---

## S14b — Implement data quality dashboard

**Status:** ⬜ Not started

**Objective:** Implement the data quality dashboard to visualize job metrics.

**Actions to perform:**
- Create `DataQualityDashboard` component.
- Implement charts/tables for key metrics (e.g., pause distribution, cue length).
- Integrate into Job Details view.

**Deliverables:**
- `DataQualityDashboard` component.
- Screenshots under `docs/screenshots/S14b/` showing the dashboard.

**Verification test:**
- **Name:** Data quality dashboard verified
- **Commands:**

```text
test -d docs/screenshots/S14b
find docs/screenshots/S14b -maxdepth 1 -type f | head -n 1
```

---

## S15 — Embedded help center design

**Status:** ⬜ Not started

**Objective:** Define an in-app help center with glossary, quickstart checklists, and guided tours.

**Actions to perform:**
- Identify top questions/confusions.
- Design entry points for help and guided tour steps.
- Capture mockups under `docs/screenshots/S15/`.
- Map help content to existing docs.

**Deliverables:**
- Screenshots or mockups under `docs/screenshots/S15/`.
- `docs/notes/help_center_plan.md` describing content and structure.

**Verification test:**
- **Name:** Help center plan documented
- **Commands:**

```text
test -d docs/screenshots/S15
test -f docs/notes/help_center_plan.md
find docs/screenshots/S15 -maxdepth 1 -type f | head -n 1
```

---

## S15b — Implement help center

**Status:** ⬜ Not started

**Objective:** Implement the embedded help center and onboarding tours.

**Actions to perform:**
- Create `HelpCenter` component (modal or sidebar).
- Implement "Tour" feature (using a library or custom).
- Populate glossary and links from `README.md` and `FRONTEND.md`.

**Deliverables:**
- `HelpCenter` and Tour components.
- Screenshots under `docs/screenshots/S15b/` showing the help UI.

**Verification test:**
- **Name:** Help center verified
- **Commands:**

```text
test -d docs/screenshots/S15b
find docs/screenshots/S15b -maxdepth 1 -type f | head -n 1
```
