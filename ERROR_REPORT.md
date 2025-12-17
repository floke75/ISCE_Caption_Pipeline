# Error Report & Structural Analysis

This document outlines technical debts and structural issues identified during the comprehensive test run and cleanup of the ISCE project.

## 1. Dependency Management & Installation Fragility

**Severity:** High
**Location:** `scripts/install.py`, `requirements.txt`

The current installation script (`scripts/install.py`) installs all dependencies from `requirements.txt` in a single pass. However, practical experience and documentation suggest that the environment is sensitive to installation order, particularly regarding `numpy`, `pandas`, and `whisperx`.

*   **Issue:** A single `pip install -r requirements.txt` can lead to resolution conflicts or downgraded packages (e.g., `whisperx` downgrading `numpy`).
*   **Recommendation:** Refactor `scripts/install.py` to implement the "batched installation" strategy documented in `AGENTS.md` (Core -> Speech -> NLP -> Web). This ensures a deterministic and stable environment.

## 2. Deprecated Library Usage (Technical Debt)

**Severity:** Medium
**Location:** `align_make.py`, `tests/test_audio_fallback_integration.py`

The codebase relies on `torchaudio` features that are explicitly marked for removal in version 2.9.

*   **Issue:** Usage of `torchaudio.load` (implicitly uses deprecated backend), `torchaudio.list_audio_backends`, and `torchaudio.io.StreamReader` triggers deprecation warnings.
*   **Impact:** Future updates to `torchaudio` will break audio loading and fallback mechanisms.
*   **Recommendation:** Migrate audio loading logic to use the recommended `TorchCodec` or `ffmpeg-python` bindings where appropriate, and update `align_make.py` to remove reliance on deprecated `torchaudio` backends.

## 3. End-to-End Testing Infrastructure

**Severity:** Medium
**Location:** `tests/e2e/conftest.py`

The current E2E test setup is manual and brittle.

*   **Issue:** Tests require the React frontend to be manually started on port 5173. There is no fixture or script to automatically provision the frontend for testing.
*   **Impact:** CI/CD pipelines cannot easily run E2E tests without complex setup steps.
*   **Recommendation:** Create a pytest fixture that automatically builds and serves the frontend (or runs the dev server) in a background process during the test session, ensuring a self-contained test environment.

## 4. Hardcoded Configuration in Tests

**Severity:** Low
**Location:** `tests/e2e/conftest.py`

*   **Issue:** The frontend URL is hardcoded to `http://localhost:5173`.
*   **Recommendation:** Externalize this configuration to environment variables or `pytest.ini` to allow testing against different deployments (e.g., staging, production build).

## 5. Python Path Configuration (Fixed)

**Severity:** Fixed
**Location:** `pytest.ini`, `ui/backend/tests/`

*   **Issue:** Tests in `ui/backend/tests/` failed with `ModuleNotFoundError: No module named 'ui'` because the project root was not implicitly in the python path.
*   **Resolution:** Added `pythonpath = .` to `pytest.ini`. This allows pytest to correctly resolve top-level packages without manual `PYTHONPATH` manipulation.
