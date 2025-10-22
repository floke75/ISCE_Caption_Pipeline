# Control Center UI Screenshot

This document explains how to build the production frontend, launch the control
center locally, and capture a canonical dashboard screenshot for reviewers. The
screenshot **must not** be committed to the repository because the PR system
rejects binary files; instead, convert the capture to a data-URL snippet (steps
5 and 6) or, if needed, upload it as a temporary CI artifact and reference that
URL in your review notes.

To reproduce the screenshot locally:

1. Install the lightweight API dependencies:

   ```bash
   pip install fastapi uvicorn pyyaml
   ```

2. Build the production frontend bundle so the static assets exist:

   ```bash
   (cd ui/frontend && npm install && npm run build)
   ```

   The build emits files into `ui/static`, which is the directory the server
   loads when the `ISCE_UI_STATIC_DIR` environment variable is set.

3. Start the FastAPI server from the repository root and point it at the built
   assets:

   ```bash
   ISCE_UI_STATIC_DIR=ui/static uvicorn ui.server:app --host 0.0.0.0 --port 8000
   ```

4. From another shell, capture the dashboard using Playwright (requires
   `playwright` to be installed and browsers set up; the first capture on a new
   machine may also require `playwright install-deps chromium`):

   ```bash
   playwright install
   playwright screenshot http://127.0.0.1:8000 control-center-ui.png
   ```

5. Convert the capture to a shareable data URL snippet and post it directly in
   your status update or PR comment:

   ```bash
   DATA_URL="data:image/png;base64,$(base64 -w0 control-center-ui.png)"
   echo "![control-center-ui](${DATA_URL})"
   ```

   Paste the echoed markdown into your update so reviewers can load the
   screenshot without downloading external artifacts.

6. Delete the temporary file and confirm the repository is clean before you
   commit any documentation updates:

   ```bash
   rm control-center-ui.png
   git status -sb
   ```

## Referencing the latest capture

When documenting a run, include the inline image tag, viewport size, and the
application state that appears in the screenshot. Example language for a PR or
run summary:

- Screenshot embed: `![control-center-ui](data:image/png;base64,<...>)`
  - Viewport: `1440x900` showing the inference workflow tab and live job monitor
    with the production bundle loaded.
  - Notes: highlight any workflow tabs, inference forms, job monitor entries, or
  configuration panels that appear so reviewers understand the UI context.

## UI tour

1. **Workflow tabs** – Switch between inference, training pair generation,
   model training, and configuration editors from a single navigation bar.
2. **Inference launch form** – Collects media, optional transcript, and override
   inputs with validated pickers and config editors to start a job.
3. **Queue-aware job monitor** – Streams live progress, log access, and artifact
   links for active and historical jobs.
4. **Configuration editors** – Provide YAML-backed editing for pipeline and
   model settings with validation before persisting changes.

### Workflow tabs

The tab strip at the top of the workspace flips between the four job types.
Each tab updates the primary form without navigating away, so operators can
launch different tasks in succession while a shared job monitor stays pinned on
the right.

### Inference launch form

The highlighted inference panel accepts a media file, an optional transcript,
and destination paths via validated path pickers that query the server. Advanced
users can expand the overrides editor to tweak pipeline or model parameters
before submitting the job, and the queue indicator shows where the request will
land in the worker pool.

### Job monitor

The sidebar lists pending and finished jobs with progress bars, status chips,
and quick actions. Selecting a job streams live logs over Server-Sent Events and
exposes artifact download links once each step completes.

### Configuration editors

Pipeline and model configuration cards render the underlying YAML with
conflict-safe editing controls. Changes are validated on the server so partial
saves or invalid syntax never overwrite the active settings.
