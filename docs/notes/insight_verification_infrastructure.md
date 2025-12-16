# Insight Verification Infrastructure

This document describes the test harness and fixtures implemented to verify complex visualizations (Alignment Viewer, Data Quality Dashboard) and system health signals in the ISCE frontend.

## Location
The verification infrastructure is located in `tests/e2e/`. It uses **Playwright (Python)** to interact with the running frontend application.

- **Fixtures:** `tests/e2e/conftest.py`
- **Tests:** `tests/e2e/test_*.py`
- **Mock Data:** `tests/fixtures/`

## Core Components

### 1. Mock Data
Reusable mock artifacts are stored in `tests/fixtures/` to simulate various job states without running the heavy backend pipeline.
- `mock_inference.enriched.json`: Sample inference output with structural breaks.
- `mock_training.train.words.json`: Sample training output with SB/LB labels.
- `mock_alignment.asr.json`: Sample ASR reference for alignment visualization.

### 2. Pytest Fixtures (`tests/e2e/conftest.py`)
The harness provides fixtures to mock backend API responses, allowing the frontend to be tested in isolation.

- **`mock_job_list(jobs: list)`**: Intercepts `GET /api/jobs` and returns the provided list of job objects.
- **`mock_job_artifacts(job_id, artifacts)`**: Intercepts `GET /api/files/content` to serve mock JSON content for specific files.
- **`mock_health(status)`**: Intercepts `GET /api/health` to simulate system states (e.g., Disk Full).
- **`visual_verifier(name)`**: Helper to capture full-page screenshots and save them to `docs/screenshots/verification/<test_name>/<name>.png`.
- **`frontend_url`**: Returns the base URL of the frontend (defaults to `http://localhost:5173`).

## Usage

### Prerequisites
1. **Start the Frontend:** The Vite dev server must be running.
   ```bash
   cd ui/frontend
   npm run dev
   ```
2. **Install Dependencies:**
   ```bash
   pip install pytest-playwright
   playwright install chromium
   ```

### Running Tests
Run the E2E suite using pytest:
```bash
pytest tests/e2e/
```

To run a specific test and see the browser (headed mode):
```bash
pytest tests/e2e/test_infrastructure_smoke.py --headed
```

### Adding New Verification Tests
1. Create a new test file in `tests/e2e/`.
2. Inject `page`, `mock_job_list`, and `mock_job_artifacts` fixtures.
3. Setup the mock state.
4. Navigate to the page.
5. Use `visual_verifier` to capture the state.

Example:
```python
def test_alignment_view(page, mock_job_list, mock_job_artifacts, visual_verifier, frontend_url):
    # Setup job
    mock_job_list([{"id": "job1", "type": "training", "status": "succeeded", ...}])

    # Setup artifacts
    with open("tests/fixtures/mock_training.train.words.json") as f:
        train_data = json.load(f)
    mock_job_artifacts("job1", {"train.words.json": train_data})

    # Go to job details
    page.goto(f"{frontend_url}/jobs/job1")

    # Verify
    visual_verifier("alignment_view")
```
