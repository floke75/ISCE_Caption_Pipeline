import json
import time
from playwright.sync_api import sync_playwright

MOCK_JOBS = [
    {
        "id": "job-inf-running",
        "jobType": "inference",
        "status": "running",
        "progress": 0.45,
        "message": "Step 2/3: Aligning text...",
        "createdAt": "2023-10-27T10:00:00Z",
        "updatedAt": "2023-10-27T10:05:00Z",
        "workspacePath": "/data/workspaces/job-inf-running",
        "params": {
            "media_path": "/data/media/movie.mp4",
            "transcript_path": "/data/transcripts/movie.txt",
            "config_overrides": {"align_make.do_diarization": True}
        },
        "result": None,
        "error": None
    },
    {
        "id": "job-train-success",
        "jobType": "training_pair",
        "status": "succeeded",
        "progress": 1.0,
        "message": "Training pair generation complete",
        "createdAt": "2023-10-26T15:30:00Z",
        "updatedAt": "2023-10-26T15:35:12Z",
        "workspacePath": "/data/workspaces/job-train-success",
        "params": {
            "media_path": "/data/media/clip.wav",
            "srt_path": "/data/captions/clip.srt"
        },
        "result": {
            "output_manifest": "/data/workspaces/job-train-success/manifest.json",
            "token_count": 1420
        },
        "error": None
    },
    {
        "id": "job-model-fail",
        "jobType": "model_training",
        "status": "failed",
        "progress": 0.1,
        "message": "Error in iteration 1",
        "createdAt": "2023-10-25T09:00:00Z",
        "updatedAt": "2023-10-25T09:01:30Z",
        "workspacePath": "/data/workspaces/job-model-fail",
        "params": {
            "corpus_dir": "/data/corpus/v1",
            "iterations": 3
        },
        "result": None,
        "error": "ValueError: Corpus directory is empty or contains no valid .json files.\n  at scripts/train_model.py:45\n  at main()"
    },
    {
        "id": "job-inf-pending",
        "jobType": "inference",
        "status": "pending",
        "progress": 0.0,
        "message": "Queued",
        "createdAt": "2023-10-27T10:10:00Z", # Newer than running
        "updatedAt": "2023-10-27T10:10:00Z",
        "workspacePath": "/data/workspaces/job-inf-pending",
        "params": {
            "media_path": "/data/media/news.mp4"
        },
        "result": None,
        "error": None
    }
]

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        # Mock /api/jobs
        def handle_jobs(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_JOBS)
            )
        page.route("**/api/jobs", handle_jobs)

        def handle_logs(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({"log": "Mock log output line 1\nMock log output line 2..."})
            )
        page.route("**/api/jobs/*/logs**", handle_logs)

        def handle_stream(route):
             route.fulfill(
                status=200,
                content_type="text/event-stream",
                body="event: message\ndata: Mock stream log\n\n"
            )
        page.route("**/api/jobs/*/logs/stream", handle_stream)


        print("Navigating to app...")
        page.goto("http://localhost:5173/")
        page.wait_for_selector("text=Job monitor")

        # Wait for at least one job row
        page.wait_for_selector(".job-row", timeout=5000)

        # Capture full board
        page.screenshot(path="docs/screenshots/S07/job_board_list.png")
        print("Captured job_board_list.png")

        # The list is sorted by date descending.
        # Order expected:
        # 1. inf-pending (10:10)
        # 2. inf-running (10:00)
        # 3. train-success (prev day 15:30)
        # 4. model-fail (prev day 09:00)

        rows = page.locator(".job-row")

        # 1. Pending
        rows.nth(0).click()
        time.sleep(0.5)
        page.screenshot(path="docs/screenshots/S07/details_pending.png")
        print("Captured details_pending.png")

        # 2. Running
        rows.nth(1).click()
        time.sleep(0.5)
        page.screenshot(path="docs/screenshots/S07/details_running.png")
        print("Captured details_running.png")

        # 3. Success
        rows.nth(2).click()
        time.sleep(0.5)
        page.screenshot(path="docs/screenshots/S07/details_success.png")
        print("Captured details_success.png")

        # 4. Fail
        rows.nth(3).click()
        time.sleep(0.5)
        page.screenshot(path="docs/screenshots/S07/details_fail.png")
        print("Captured details_fail.png")

        browser.close()

if __name__ == "__main__":
    run()
