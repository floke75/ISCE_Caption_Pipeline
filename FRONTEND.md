# ISCE UI Frontend Field Guide

This guide complements `AGENTS.md` with React-specific context for the Vite-based control center under `ui/frontend/`. It summarizes how the SPA is structured, how it communicates with the FastAPI backend, and what guardrails the UI enforces around file paths and job monitoring.

## Architecture snapshot
- **Entry + routing:** `src/main.tsx` mounts `<App />`, which renders a tabbed workbench and persistent job sidebar. Tabs switch between inference, training pair creation, model training, and configuration forms without any router dependency; state is kept locally in `App.tsx`. 【F:ui/frontend/src/main.tsx†L1-L16】【F:ui/frontend/src/App.tsx†L1-L86】
- **Data layer:** Axios is wrapped in a singleton client with a `/api` base URL and 60s timeout; Vite proxies `/api` to the backend during dev. React Query handles caching/polling across hooks. 【F:ui/frontend/src/api/client.ts†L1-L16】【F:ui/frontend/package.json†L7-L26】
- **Styling:** Global layout and forms rely on CSS modules under `src/styles/` (e.g., `app.css`, `forms.css`, `jobs.css`) imported directly by components. 【F:ui/frontend/src/App.tsx†L8-L9】【F:ui/frontend/src/components/InferenceForm.tsx†L7-L8】

## Job visibility and logs
- **Job list + details:** `JobBoard` polls `/jobs` every 5s and sorts newest-first. Each row shows status, progress bar, message, relative timestamp, and quick actions for copying the workspace path or cancelling pending/running jobs. 【F:ui/frontend/src/hooks/useJobs.ts†L14-L33】【F:ui/frontend/src/components/JobBoard.tsx†L207-L303】
- **Detail panes:** Selecting a job reveals three panels: metadata, runtime parameters, and result payload. Values that look like paths are rendered in `<code>` blocks with copy buttons; JSON payloads are pretty-printed. 【F:ui/frontend/src/components/JobBoard.tsx†L59-L137】【F:ui/frontend/src/components/JobBoard.tsx†L303-L376】
- **Logs:** `JobBoard` uses `useEventStream` for live Server-Sent Events at `/jobs/{id}/logs/stream`, falling back to polling `/jobs/{id}/logs` when SSE is unsupported or fails. The log viewer supports auto-scroll toggling and copying the full buffer. Stream state is surfaced via inline status text. 【F:ui/frontend/src/components/JobBoard.tsx†L220-L295】【F:ui/frontend/src/hooks/useJobs.ts†L35-L51】【F:ui/frontend/src/hooks/useEventStream.ts†L1-L155】

## Forms and path safety
- **File browsing:** All path inputs use `FilePathPicker`, which fetches allowlisted roots and validates paths through backend endpoints before marking them as usable. It supports both file and directory modes and disables submission when validation fails. 【F:ui/frontend/src/components/FilePathPicker.tsx†L1-L214】
- **Inference:** `InferenceForm` requires a media file and optionally takes transcript, output directory, model config path, operator notes, and config/segmentation override JSON. It blocks submission if path validation fails or override JSON is invalid, then POSTs to `/jobs/inference`. 【F:ui/frontend/src/components/InferenceForm.tsx†L1-L116】【F:ui/frontend/src/components/InferenceForm.tsx†L130-L185】
- **Training pairs:** `TrainingPairForm` collects media, transcript, and manual SRT paths plus optional overrides before POSTing to `/jobs/training-pair`. It reuses the same validation and override flow as inference. 【F:ui/frontend/src/components/TrainingPairForm.tsx†L1-L200】
- **Model training:** `ModelTrainingForm` targets `/jobs/model-training`, taking a training corpus directory, optional constraints/model weight output paths, iteration count, and override patches. Submission is disabled until required paths validate and overrides parse. 【F:ui/frontend/src/components/ModelTrainingForm.tsx†L1-L218】

## Configuration surface
- **Pipeline + segmentation overrides:** `OverrideEditor` provides two JSON editors (pipeline and segmentation). It performs JSON parsing + schema-free validation, emits structured patch objects, and blocks parent forms when errors are present. 【F:ui/frontend/src/components/OverrideEditor.tsx†L1-L205】
- **Schema-driven config panel:** `ConfigPanel` fetches the merged pipeline/segmentation configuration plus schema metadata and renders grouped fields by section. Users can toggle advanced fields, edit values inline, persist overrides, or reset to defaults via backend endpoints. 【F:ui/frontend/src/components/ConfigPanel.tsx†L1-L239】

## Networking and streaming helpers
- **React Query hooks:** `useJobs` and `useJobLog` encapsulate polling defaults for job lists and logs, with optional interval overrides. They are shared across `JobBoard` for list refresh and log fallback. 【F:ui/frontend/src/hooks/useJobs.ts†L14-L51】
- **SSE wrapper:** `useEventStream` wraps `EventSource` with heartbeat/error handling, optional event filtering, and automatic reconnect toggles. Callers receive `status`, `supported`, and `disconnect` controls alongside message callbacks. 【F:ui/frontend/src/hooks/useEventStream.ts†L1-L155】

## Types and shared contracts
- **Type definitions:** `src/types.ts` centralizes job records, config schema nodes, and file-browser types so components and hooks stay consistent with backend responses. Update this file first when backend payloads change. 【F:ui/frontend/src/types.ts†L1-L75】【F:ui/frontend/src/types.ts†L77-L118】

## Development ergonomics
- **Scripts:** Use `npm run dev` for live reload via Vite, `npm run build` for type-check + production bundle, and `npm run preview` to serve the built app locally. React Query Devtools are not bundled; debug network calls via the browser console. 【F:ui/frontend/package.json†L7-L22】
- **API base:** During local development, the Vite proxy routes `/api` to the FastAPI backend; when deployed behind the Python server, the same relative path works because the backend mounts the SPA and proxies `/api` routes. No environment variables are required for the frontend itself. 【F:ui/frontend/src/api/client.ts†L1-L16】【F:ui/frontend/package.json†L1-L22】
