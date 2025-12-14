#!/usr/bin/env bash
set -euo pipefail

# Run Ruff linting with repository configuration.
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ruff check \
  isce \
  ui/backend \
  scripts \
  tests
