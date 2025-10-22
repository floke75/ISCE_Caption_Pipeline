# Building the Control Center UI

The control center UI is built with Vite and is no longer checked into the repository. Build artifacts under `ui/frontend/dist/` and the legacy `ui/static/` directory are ignored to keep the tree clean and guarantee that the API always serves a fresh bundle.

## Prerequisites

- Node.js 18+
- npm 9+ (bundled with Node.js)

## One-time setup

```bash
cd ui/frontend
npm install
```

## Creating a production bundle

```bash
npm run build
```

The compiled assets are written to `ui/frontend/dist/`. The FastAPI server automatically serves this directory when it exists. If the directory is missing, requests to the root UI route return a `503` response describing how to build the frontend.

## Development workflow

- Run `npm run dev` for a hot-reloading development server on port 5173.
- In another terminal start the FastAPI app (`uvicorn ui.server:app --reload`) to exercise API endpoints.
- When the UI is ready to ship, run `npm run build` and restart the FastAPI app so it can serve the new bundle from `ui/frontend/dist/`.

> **Tip:** The log monitor uses Server-Sent Events (SSE). When testing locally behind a proxy, ensure the proxy allows streaming responses.
