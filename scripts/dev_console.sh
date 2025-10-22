#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/ui/frontend"

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "Error: uvicorn is not installed or not on PATH." >&2
  echo "Install FastAPI dependencies (pip install -r requirements.txt) before running this script." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is not installed or not on PATH." >&2
  echo "Install Node.js 18+ to run the frontend development server." >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR" ]]; then
  echo "Error: Could not find frontend directory at $FRONTEND_DIR" >&2
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "warning: node_modules not found. Running 'npm install' in ui/frontend may be required." >&2
fi

if [[ -n "${DEV_CONSOLE_BACKEND_CMD:-}" ]]; then
  # shellcheck disable=SC2206 # Intentional word splitting to honour quoted args in override
  backend_cmd=( ${DEV_CONSOLE_BACKEND_CMD} )
else
  backend_cmd=(
    uvicorn "${UVICORN_APP:-ui.server:app}" \
      --reload \
      --host "${UVICORN_HOST:-127.0.0.1}" \
      --port "${UVICORN_PORT:-8000}"
  )
fi

if [[ -n "${DEV_CONSOLE_FRONTEND_CMD:-}" ]]; then
  # shellcheck disable=SC2206 # Intentional word splitting to honour quoted args in override
  frontend_cmd=( ${DEV_CONSOLE_FRONTEND_CMD} )
else
  frontend_cmd=(
    npm --prefix "$FRONTEND_DIR" run dev -- \
      --host "${FRONTEND_HOST:-127.0.0.1}" \
      --port "${FRONTEND_PORT:-5173}"
  )
fi

pids=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait "${pids[@]}" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

pushd "$ROOT_DIR" >/dev/null

echo "Launching FastAPI backend: ${backend_cmd[*]}"
"${backend_cmd[@]}" &
pids+=($!)

sleep 1

echo "Launching Vite frontend: ${frontend_cmd[*]}"
"${frontend_cmd[@]}" &
pids+=($!)

popd >/dev/null

echo "Both services are running. Backend -> http://${UVICORN_HOST:-127.0.0.1}:${UVICORN_PORT:-8000}  Frontend -> http://${FRONTEND_HOST:-127.0.0.1}:${FRONTEND_PORT:-5173}"

echo "Press Ctrl+C to stop both processes."

wait -n "${pids[@]}"
status=$?
cleanup
exit $status
