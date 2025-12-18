import pytest
import json
from pathlib import Path
from playwright.sync_api import Page, Route

# Default frontend URL
FRONTEND_URL = "http://localhost:5173"

@pytest.fixture(scope="session")
def frontend_url():
    return FRONTEND_URL

@pytest.fixture
def mock_job_artifacts(page: Page):
    """
    Fixture that allows tests to inject mock artifacts for specific jobs.
    Usage:
        mock_job_artifacts(job_id="123", artifacts={"inference.enriched.json": data})
    """
    def _setup(job_id: str, artifacts: dict):
        # Mock the content endpoint
        def handle_content(route: Route):
            # Extract path from query params or url
            # The backend API is likely /api/files/content?path=...
            # But the UI might use encoded paths.
            # We'll assume the UI requests something like /api/files/content?path=...
            url = route.request.url
            if "inference.enriched.json" in url and "inference.enriched.json" in artifacts:
                route.fulfill(status=200, content_type="application/json", body=json.dumps(artifacts["inference.enriched.json"]))
            elif "train.words.json" in url and "train.words.json" in artifacts:
                route.fulfill(status=200, content_type="application/json", body=json.dumps(artifacts["train.words.json"]))
            elif "asr.visual.words.diar.json" in url and "asr.visual.words.diar.json" in artifacts:
                route.fulfill(status=200, content_type="application/json", body=json.dumps(artifacts["asr.visual.words.diar.json"]))
            else:
                route.continue_()

        page.route("**/api/files/content**", handle_content)

        # Also mock the download endpoint if needed
        # page.route("**/api/files/download**", handle_content)

    return _setup

@pytest.fixture
def mock_job_list(page: Page):
    """Mocks the /api/jobs endpoint with a customizable list of jobs."""
    def _setup(jobs: list):
        page.route("**/api/jobs", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(jobs)
        ))
    return _setup

@pytest.fixture
def mock_health(page: Page):
    """Mocks the /api/health endpoint."""
    def _setup(status: dict = None):
        if status is None:
            status = {
                "status": "ok",
                "system": {
                    "disk": {
                        "free_bytes": 100 * 1024**3,
                        "total_bytes": 500 * 1024**3,
                        "percent_used": 20.0,
                        "error": None
                    },
                    "memory": {
                        "available_bytes": 8 * 1024**3,
                        "total_bytes": 16 * 1024**3,
                        "percent_used": 50.0,
                        "error": None
                    },
                    "gpu": {
                        "available": True,
                        "name": "Mock GPU",
                        "device_count": 1
                    }
                },
                "queue": {
                    "pending": 0,
                    "active": 0,
                    "slots_total": 4
                }
            }
        page.route("**/api/health", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(status)
        ))
    return _setup

@pytest.fixture
def visual_verifier(page: Page, request):
    """
    Helper to take screenshots and save them to docs/screenshots/<test_name>/.
    """
    def _verify(name: str):
        # Create directory
        test_name = request.node.name
        screenshot_dir = Path("docs/screenshots/verification") / test_name
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        path = screenshot_dir / f"{name}.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"Captured screenshot: {path}")
        return path
    return _verify
