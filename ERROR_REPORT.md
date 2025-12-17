# ISCE Technical Debt & Cleanup Task List

This document outlines a prioritized list of technical debts and structural issues identified in the codebase, formatted as detailed actionable tasks for an LLM coding agent.

---

## 1. Fix Silent Pipeline Failures

**Severity:** Critical
**Components:** `build_training_pair_standalone.py`, `align_make.py`

### Description
Both `build_training_pair_standalone.py` and `align_make.py` wrap their main execution logic in a broad `try-except` block that catches `Exception`, prints a traceback, but allows the script to exit with code `0` (success). This causes the orchestrator (`run_pipeline.py` or `pipelines.py`) to mistakenly believe the step succeeded, leading to confusing `FileNotFoundError` downstream when expected artifacts are missing.

### Analysis
1.  Inspect `build_training_pair_standalone.py`'s `main()` function. Identify the `try-except` block around `process_file`.
2.  Inspect `align_make.py`'s `main()` function. Identify the `try-except` block around `process_file`.
3.  Verify that `sys.exit(1)` is not called in the `except` blocks.

### Planning
1.  Modify `build_training_pair_standalone.py`:
    *   In the `except` block of `main()`, add `sys.exit(1)` after printing the traceback.
2.  Modify `align_make.py`:
    *   In the `except` block of `main()`, add `sys.exit(1)` after printing the traceback.

### Execution
1.  Edit `build_training_pair_standalone.py` to import `sys` (if missing) and add the exit call.
2.  Edit `align_make.py` to import `sys` (if missing) and add the exit call.

### Testing
1.  **Unit Test**: Create a test case that mocks `process_file` to raise an exception.
2.  **Verify**: Run the script via `subprocess.run` in the test and assert that `returncode` is `1` (not `0`).
3.  **Regression**: Ensure normal execution still exits with `0`.

### Code Review
*   Confirm that the traceback is still printed to stderr/stdout before exiting.
*   Confirm that no other exception handling logic is bypassed.

---

## 2. Refactor Dependency Installation (Fragile Environment)

**Severity:** High
**Components:** `scripts/install.py`, `requirements.txt`

### Description
The current `scripts/install.py` installs all dependencies in a single pass using `requirements.txt`. Experience shows this leads to resolution conflicts (e.g., `whisperx` downgrading `numpy`/`pandas`). A batched installation strategy is required for stability.

### Analysis
1.  Review `AGENTS.md` for the recommended installation order: Core -> Speech -> NLP -> Web.
2.  Analyze `scripts/install.py` to see how it currently invokes `pip`.
3.  Analyze `requirements.txt` to split it into logical groups or tags.

### Planning
1.  Update `requirements.txt` to use comment-based markers (e.g., `# --- CORE ---`, `# --- SPEECH ---`) or split into separate files (`requirements/core.txt`, etc.). *Decision: Split into separate files in a `requirements/` directory for cleanliness.*
2.  Refactor `scripts/install.py`:
    *   Define the installation groups and their order.
    *   Implement a function to install each group sequentially.
    *   Keep the existing `spacy` model logic.

### Execution
1.  Create `requirements/` directory.
2.  Split `requirements.txt` into `core.txt`, `speech.txt`, `nlp.txt`, `web.txt`.
3.  Update `scripts/install.py` to iterate through these files in order.
4.  Update `requirements.txt` to be a meta-file that includes the others (if possible) or leave it as a reference, but make `install.py` the source of truth.

### Testing
1.  **Clean Install**: Create a fresh virtual environment.
2.  **Run Script**: Execute `python scripts/install.py`.
3.  **Verify**: Check that all packages are installed and no conflict warnings appeared during the process.
4.  **Smoke Test**: Run `tests/test_environment_smoke.py`.

### Code Review
*   Ensure `scripts/install.py` still supports the `--skip-frontend` and `--gpu` flags.
*   Verify that the split requirements cover all original dependencies.

---

## 3. Handle Interrupted Jobs on Restart

**Severity:** High
**Components:** `ui/backend/job_manager.py`

### Description
The `JobManager` loads existing jobs from disk on startup. If the server crashed or was killed while a job was `running`, that job remains in the `running` state indefinitely in the persisted metadata, blocking the queue and confusing users.

### Analysis
1.  Examine `JobManager._load_existing_jobs` in `ui/backend/job_manager.py`.
2.  Note that it simply loads the record into memory without checking validity of the state relative to the new process.

### Planning
1.  Modify `_load_existing_jobs`:
    *   Iterate through loaded jobs.
    *   If a job's status is `running` or `pending`:
        *   Mark it as `failed` (or `interrupted`).
        *   Update the `error` field with a message like "System restarted while job was active".
        *   Persist the updated state to disk immediately.
2.  Ensure this logic runs before the `JobManager` accepts new jobs.

### Execution
1.  Implement the state check and update logic in `_load_existing_jobs`.

### Testing
1.  **Unit Test**:
    *   Create a mock `JobRecord` on disk with status `running`.
    *   Initialize `JobManager`.
    *   Assert that the loaded job has status `failed` and the error message is present.
2.  **Integration**: Manually start a job, kill the server, restart, and verify UI shows "failed".

### Code Review
*   Check if `pending` jobs should also be failed or if they can be re-queued. (Safe default: fail them to avoid unexpected execution).

---

## 4. Modernize Audio Loading (Deprecation Fix)

**Severity:** Medium
**Components:** `align_make.py`, `tests/test_audio_fallback_integration.py`

### Description
The codebase uses `torchaudio` functions (`load`, `list_audio_backends`, `StreamReader`) that are deprecated and will be removed in version 2.9. This is technical debt that will break the pipeline in the future.

### Analysis
1.  Identify usages of `torchaudio.load`, `torchaudio.backend`, and `torchaudio.io.StreamReader` in `align_make.py`.
2.  Research `torchaudio` 2.x migration guide (recommendation: use `ffmpeg` directly or `torchcodec` if stable, or `torchaudio.load` with updated backend settings).
3.  Since `ffmpeg-python` is already a dependency, prefer using it or the native fallback more robustly.

### Planning
1.  In `align_make.py`:
    *   Replace `torchaudio.load` with `torchaudio.load` (if the warning is just about the backend, ensure the default backend is safe) OR switch to `soundfile` for the fallback path since `soundfile` is already a dependency.
    *   *Decision*: Switch fallback to `soundfile` completely if `torchaudio` is too unstable/deprecated.
2.  Update `tests/test_audio_fallback_integration.py` to test the new implementation.

### Execution
1.  Modify `load_audio_native` in `align_make.py` to use `soundfile.read` instead of `torchaudio.load`. `soundfile` returns numpy arrays directly, matching the requirement for WhisperX.
2.  Remove `torchaudio` specific imports if they are no longer needed for loading.

### Testing
1.  **Unit Test**: Run `tests/test_audio_fallback_integration.py`.
2.  **Verify**: Ensure no `UserWarning` about `torchaudio` deprecations appear.
3.  **Functionality**: Verify that audio is still correctly loaded and resampled (if needed, `soundfile` doesn't resample, so we might need `scipy.signal.resample` or keep `torchaudio.transforms.Resample`). *Correction*: `librosa` is good for this but might add dependency. `torchaudio.transforms` might stay, but loading can move to `soundfile`.

### Code Review
*   Verify that `soundfile` supports the file formats we expect (WAV, FLAC, OGG). MP3 might require `libsndfile` with mp3 support.
*   Ensure resampling logic remains correct.

---

## 5. Automate E2E Test Infrastructure

**Severity:** Medium
**Components:** `tests/e2e/conftest.py`

### Description
E2E tests require a manually started frontend server. This prevents automated testing in CI/CD and makes local testing tedious.

### Analysis
1.  Inspect `tests/e2e/conftest.py`.
2.  Identify where `frontend_url` is defined.

### Planning
1.  Create a session-scoped fixture `frontend_server` in `conftest.py`.
2.  The fixture should:
    *   Check if port 5173 is free.
    *   Start `npm run dev` in a subprocess (using `ui/frontend` as cwd).
    *   Wait for the server to respond (health check loop).
    *   Yield the URL.
    *   Teardown: Kill the subprocess tree.
3.  Update `frontend_url` fixture to depend on `frontend_server`.

### Execution
1.  Implement the subprocess management logic using `subprocess.Popen` and `atexit` or fixture teardown.
2.  Handle `npm` availability check.

### Testing
1.  **Execution**: Run `pytest tests/e2e/` without manually starting the server.
2.  **Verify**: Tests pass and the server process terminates afterwards.

### Code Review
*   Ensure it doesn't leave zombie processes.
*   Ensure it handles port conflicts gracefully (maybe skip if port is in use and assume external server?).

---

## 6. Optimize Global Alignment

**Severity:** Medium (Performance)
**Components:** `build_training_pair_standalone.py`

### Description
The `_global_align` function in `build_training_pair_standalone.py` performs O(NM) comparisons in Python. It repeatedly normalizes tokens inside the inner loop, causing significant overhead.

### Analysis
1.  Inspect `_global_align` and `_match_score`.
2.  Notice `_match_score` calls `_norm_token` on `a` and `b` every time.
3.  `a` comes from `a_tokens[i]` and `b` from `b_tokens[j]`. Each token is normalized M or N times respectively.

### Planning
1.  Pre-calculate normalized versions of `a_tokens` and `b_tokens` before the loop: `a_norm = [_norm_token(t) for t in a_tokens]`.
2.  Modify `_match_score` (or inline it) to accept pre-normalized strings.
3.  (Optional) Investigate if `rapidfuzz` has a bulk alignment function or if we can vectorise this. For now, pre-normalization is a low-hanging fruit.

### Execution
1.  Implement pre-normalization in `_global_align`.
2.  Update the inner loop to use the normalized lists.

### Testing
1.  **Regression**: Run `tests/test_build_training_pair.py` (or create one if missing) to ensure alignment results are identical.
2.  **Benchmark**: Measure execution time on a large transcript before and after.

### Code Review
*   Verify logic equivalence.
