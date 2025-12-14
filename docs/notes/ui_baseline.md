# S01 — UI baseline capture

## Environment and commands
- Frontend dependencies from `ui/frontend/package.json`: React 18.3.1 with Vite 5.2.10 and React Query/axios stack (package version 0.1.0). Server started via `npm run dev -- --host 0.0.0.0 --port 8001 --strictPort` after local `npm install`.
- Screenshots captured with local Playwright (`python -m playwright install chromium` + `playwright install-deps chromium`) due to port-forwarding limits in the browser tool.

## Navigation observations
- Default **Inference** tab shows media/transcript/file pickers, override editor, and persistent **Job board** sidebar. Job board showed proxy failures because the FastAPI backend was not running during capture.
- **Training pairs** and **Model training** tabs mirror the inference layout with tailored path pickers and override editors; submission buttons disabled until validation passes.
- **Configuration** tab renders pipeline and segmentation sub-tabs with grouped fields and reset/save actions.

## Issues encountered
- Vite dev server logged repeated proxy errors to `/api/*` (jobs, config, files) since no backend was available on port 8000. These did not block tab rendering but left Job board entries empty.

## Captured assets
- Binary screenshots were removed from version control to keep the PR free of large image files. The intended captures and
  recreation steps are documented in `docs/screenshots/S01/README.md`.

## Verification commands
- Executed: `test -d docs/screenshots/S01`, `test -f docs/notes/ui_baseline.md`, `find docs/screenshots/S01 -maxdepth 1 -type f | head -n 1`.

### Verification command outputs
- `test -d docs/screenshots/S01` → pass
- `test -f docs/notes/ui_baseline.md` → pass
- `find docs/screenshots/S01 -maxdepth 1 -type f | head -n 1` → `docs/screenshots/S01/job-board.png`
