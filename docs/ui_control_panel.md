# ISCE Captioning Control Panel

This UI layer wraps the existing orchestration scripts and training utilities, allowing non-technical operators to launch jobs,
monitor progress, and edit configuration files without the command line.

## Backend API (`ui/server.py`)

- Built with **FastAPI**. Run it with `uvicorn ui.server:app --reload` or set the `ISCE_UI_DEV_SERVER=1` environment variable
  and execute `python -m ui.server`.
- Launches inference and training-pair workflows by calling `run_pipeline.process_inference_file` and
  `run_pipeline.process_training_file` directly. Subprocess stdout is streamed back through a background job manager so the UI
  can show live logs.
- Wraps `scripts/train_model.py` for the iterative reweighting loop. Output paths are created automatically.
- Includes endpoints for reading/writing `pipeline_config.yaml` and `config.yaml` with validation helpers.

## Frontend (`ui/frontend`)

- **React + Vite + Tailwind CSS** single-page app in `ui/frontend`.
- Provides forms for manual inference, training-pair generation, and model training. Jobs appear immediately in the history table
  with live status badges.
- Job logs stream in real time via `/jobs/{id}/logs` and configuration editors expose both effective values and editable
  overrides, with support for adding new keys from the UI.
- Development workflow:
  ```bash
  cd ui/frontend
  npm install
  npm run dev
  ```
  The Vite dev server proxies `/api/*` calls to the FastAPI backend running on port 8000.

## File layout

```
ui/
├── job_manager.py      # Background job runner with incremental log capture
├── config_service.py   # Shared helpers for loading/saving YAML config
├── server.py           # FastAPI application and REST endpoints
└── frontend/           # React SPA for non-technical operators
```

## Deployment notes

- Install new backend dependencies via `pip install -r requirements.txt`.
- Build the production bundle with `npm run build`; the generated assets in `ui/frontend/dist` can be served by your web server
  or packaged into an Electron shell if you need a desktop control panel.
- The API is stateless; job metadata is stored in memory. If you need persistence across restarts, extend `JobManager` to write
  logs and job metadata to disk or a lightweight database.
