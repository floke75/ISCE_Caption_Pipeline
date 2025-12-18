# Frontend readiness and data collection (S00)

## Environment checks
- Node.js available: `node -v` → `v20.19.5`.
- npm available: `npm -v` → `11.4.2` (emitted a warning about unknown env config `http-proxy`).
- Frontend dependencies already present under `ui/frontend/node_modules`, so `npm install` should be a no-op unless dependencies change.

## Frontend commands
- Frontend dev command: `cd ui/frontend && npm run dev`.
- Development server: `cd ui/frontend && npm run dev` (Vite dev server with proxy to `/api`).
- Production build: `cd ui/frontend && npm run build` (TypeScript check + Vite production bundle).
- Preview built assets locally: `cd ui/frontend && npm run preview`.
- Initial setup (if node_modules missing): `cd ui/frontend && npm install`.

Command references come from `FRONTEND.md` and `ui/frontend/package.json`.

## Screenshot and artifact protocol
- Store all UI captures under `docs/screenshots/<STEP_ID>/` using descriptive filenames (e.g., `training-form.png`).
- Take screenshots after the relevant UI is rendered; include the port/route in filenames when helpful.
- For steps without UI work (like S00), no screenshots are required.
- Keep execution logs for verification commands in step Notes; mention any tool limitations (e.g., missing browser automation).

## Constraints and observations
- Dev server expects backend at `http://localhost:8000` via the Vite `/api` proxy; adjust if backend runs elsewhere.
- Keep steps offline-friendly—avoid heavy model downloads unless a later step explicitly requires them.
