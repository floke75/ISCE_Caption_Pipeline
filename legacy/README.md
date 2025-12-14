# Legacy utilities

This directory collects scripts that are still usable for reference but are not part of the primary pipeline entrypoints.

## Included items
- `dev_console.sh`: Former combined launcher for the FastAPI backend and Vite frontend during local development. The UI can still be run directly via `uvicorn ui.backend.app:app` and `npm run dev` from `ui/frontend/`, so the helper now lives here to keep the main scripts folder focused on active tooling.
