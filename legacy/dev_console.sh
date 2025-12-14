#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/ui/frontend"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "[dev-console] Missing 'uvicorn' executable. Activate the backend virtualenv first." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[dev-console] Missing 'npm' executable. Install Node.js to launch the frontend." >&2
  exit 1
fi

cleanup() {
  local status=${1:-$?}
  trap - INT TERM EXIT
  echo
  echo "[dev-console] Shutting down..."
  for pid in "${pids[@]-}"; do
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  exit "$status"
}

trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

pids=()

echo "[dev-console] Starting FastAPI backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
(
  cd "$ROOT_DIR"
  uvicorn ui.backend.app:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload
) &
pids+=($!)

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "[dev-console] Installing frontend dependencies (one-time)..."
  (cd "$FRONTEND_DIR" && npm install)
fi

echo "[dev-console] Starting Vite dev server on http://localhost:${FRONTEND_PORT}"
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host --port "$FRONTEND_PORT"
) &
pids+=($!)

cat <<MSG

[dev-console] Development servers are running:
  • Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}
  • Frontend: http://localhost:${FRONTEND_PORT}

Press Ctrl+C to stop both.
MSG

wait "${pids[@]}"
