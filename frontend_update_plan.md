# ISCE Caption Pipeline Frontend Update Plan — Work-in-Progress

**Plan ID:** `isce_caption_pipeline_frontend_update_v1`
**Repository:** `floke75/ISCE_Caption_Pipeline`
**Plan created (UTC):** `2025-07-05T00:00:00Z`

**Step status summary:** 0/0 passed, 0 failed, 0 in progress (initial draft)

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
- [S04 — Interaction flow review for training and inference](#s04-interaction-flow-review-for-training-and-inference) — ⬜ Not started
- [S05 — Visibility of artifacts and system health signals](#s05-visibility-of-artifacts-and-system-health-signals) — ⬜ Not started

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

## S04 — Interaction flow review for training and inference

**Status:** ⬜ Not started

**Objective:** Evaluate end-to-end usability of the training and inference flows, focusing on clarity of required inputs, parameter affordances, and inline guidance for non-experts.

**Actions to perform:**
- Using the running frontend (from S01), walk through submitting representative training and inference jobs with sample inputs (real or mock paths as allowed).
- Capture screenshots of each critical step (file selection, parameter tuning, submission confirmation, and job monitoring). Save under `docs/screenshots/S04/` with descriptive names.
- Note friction points: unclear placeholders, missing validation, confusing error messages, or non-intuitive parameter labels.
- Propose concrete UX improvements (tooltips, helper text, presets) in the notes.

**Deliverables:**
- Screenshots under `docs/screenshots/S04/` covering the interaction flow.
- `docs/notes/interaction_flow.md` summarizing observations and recommended UX adjustments.

**Verification test:**
- **Name:** Interaction flow documented
- **Commands:**

```text
test -d docs/screenshots/S04
test -f docs/notes/interaction_flow.md
find docs/screenshots/S04 -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Screenshot directory exists and contains at least one file
  - Interaction flow notes exist
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- Include any console/network errors encountered while submitting jobs and suggested mitigations.

---

## S05 — Visibility of artifacts and system health signals

**Status:** ⬜ Not started

**Objective:** Assess how well the UI surfaces interpretable artifacts (SRT previews, alignment diffs, timelines) and backend health signals, then define target improvements for transparency and trust.

**Actions to perform:**
- Audit the job detail/monitoring views for available artifacts (e.g., generated SRT, enriched JSON, logs) and note how they are presented.
- Capture screenshots of artifact presentation and any health indicators (status chips, progress bars, error banners) under `docs/screenshots/S05/`.
- Identify missing insight features (e.g., waveform snippets, cue previews, scoring summaries) and map them to potential data sources.
- Document desired verification and test hooks (e.g., API endpoint checks, artifact schema validations) to support future implementation.

**Deliverables:**
- Screenshots under `docs/screenshots/S05/` showcasing current artifact and health visibility.
- `docs/notes/system_insights.md` detailing observed signals, gaps, and proposed insight widgets/tests.

**Verification test:**
- **Name:** System insight baseline captured
- **Commands:**

```text
test -d docs/screenshots/S05
test -f docs/notes/system_insights.md
find docs/screenshots/S05 -maxdepth 1 -type f | head -n 1
```
- **Expected results:**
  - Screenshot directory exists and has at least one file
  - System insights notes exist
- **Pass criteria:** All commands exit with code 0 AND at least one screenshot file is present.

**Notes:**
- For future steps, align proposed tests with backend endpoints (FastAPI) and frontend renderers to ensure artifacts/health data are validated end-to-end.

---
