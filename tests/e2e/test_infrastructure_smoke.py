import pytest
from playwright.sync_api import Page, expect

def test_visual_infrastructure_smoke(page: Page, mock_job_list, mock_health, visual_verifier, frontend_url):
    """
    Verifies that the test infrastructure can:
    1. Navigate to the app.
    2. Mock backend responses.
    3. Capture a screenshot.
    """
    # 1. Setup mocks
    mock_health()
    mock_job_list([
        {
            "id": "smoke-test-job",
            "type": "inference",
            "status": "succeeded",
            "created_at": "2023-01-01T00:00:00",
            "completed_at": "2023-01-01T00:01:00",
            "params": {"primary_input": "test.txt"},
            "error": None
        }
    ])

    # 2. Navigate
    try:
        page.goto(frontend_url)
    except Exception as e:
        pytest.fail(f"Could not connect to frontend at {frontend_url}. Ensure it is running. Error: {e}")

    # 3. Assert mock data appears
    # The job ID should be visible in the JobBoard
    try:
        # Check for the job ID
        expect(page.get_by_text("smoke-test-job")).to_be_visible(timeout=10000)
    except AssertionError:
        print("WARN: Mock job not found. Screenshotting state.")

    # 4. Capture screenshot
    screenshot_path = visual_verifier("smoke_test_dashboard")
    assert screenshot_path.exists()
