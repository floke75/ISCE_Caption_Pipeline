#!/usr/bin/env bash
set -euo pipefail

# Launch the FastAPI control plane and the Vite development server with sensible defaults.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/ui_runtime"

cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${API_PID:-}" ]] && ps -p "$API_PID" > /dev/null 2>&1; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${UI_PID:-}" ]] && ps -p "$UI_PID" > /dev/null 2>&1; then
    kill "$UI_PID" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

echo "[dev_console] Using repository root: $ROOT_DIR"
echo "[dev_console] Starting FastAPI on http://${API_HOST}:${API_PORT}"
"$PYTHON_BIN" -m uvicorn ui_server.app:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --reload \
  --reload-dir ui_server \
  --reload-dir isce \
  --reload-dir configs \
  --reload-dir scripts \
  > "$ROOT_DIR/ui_runtime/dev-api.log" 2>&1 &
API_PID=$!

echo "[dev_console] Installing frontend dependencies (if needed)"
npm install --prefix ui_frontend >/dev/null

echo "[dev_console] Starting Vite dev server on http://${UI_HOST}:${UI_PORT}"
npm run dev --prefix ui_frontend -- --host "$UI_HOST" --port "$UI_PORT" &
UI_PID=$!

echo "[dev_console] FastAPI PID: $API_PID"
echo "[dev_console] Vite PID: $UI_PID"
echo "[dev_console] Press Ctrl+C to stop both services."

wait -n "$API_PID" "$UI_PID"
