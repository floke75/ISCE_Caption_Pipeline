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
            "media_path": "/data/media/movie.mp4"
        },
        "result": None,
        "error": None
    },
    {
        "id": "job-train-success",
        "jobType": "training_pair",
        "status": "succeeded",
        "progress": 1.0,
        "message": "Success",
        "createdAt": "2023-10-26T15:30:00Z",
        "updatedAt": "2023-10-26T15:35:12Z",
        "workspacePath": "/data/workspaces/job-train-success",
        "params": {},
        "result": {},
        "error": None
    },
    {
        "id": "job-model-fail",
        "jobType": "model_training",
        "status": "failed",
        "progress": 0.1,
        "message": "Error",
        "createdAt": "2023-10-25T09:00:00Z",
        "updatedAt": "2023-10-25T09:01:30Z",
        "workspacePath": "/data/workspaces/job-model-fail",
        "params": {},
        "result": None,
        "error": "Error"
    }
]

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        def handle_jobs(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(MOCK_JOBS)
            )
        page.route("**/api/jobs", handle_jobs)

        # Mock logs
        page.route("**/api/jobs/*/logs**", lambda r: r.fulfill(status=200, body=json.dumps({"log": ""})))
        page.route("**/api/jobs/*/logs/stream", lambda r: r.fulfill(status=200, content_type="text/event-stream", body=""))

        print("Navigating to app...")
        page.goto("http://localhost:5173/")
        page.wait_for_selector("text=Job monitor")
        page.wait_for_selector(".job-row", timeout=5000)

        # Wait for select to be available
        try:
             page.wait_for_selector("select.status-select", timeout=5000)
        except:
             print("FAILURE: Filter select not found.")
             # Print header HTML
             header_html = page.inner_html(".job-board header")
             print(f"Header HTML: {header_html}")

             page.screenshot(path="docs/screenshots/S07b/debug_fail.png")
             browser.close()
             return

        # 1. Capture Full List with Icons
        page.screenshot(path="docs/screenshots/S07b/job_list_icons.png")
        print("Captured job_list_icons.png")

        # 2. Test Filter: Failed
        print("Selecting 'Failed' filter...")
        page.select_option("select.status-select", "failed")
        time.sleep(0.5)

        # Count rows
        count = page.locator(".job-row").count()
        print(f"Rows visible after filtering: {count}")
        if count != 1:
            print("FAILURE: Expected 1 row after filtering")

        page.screenshot(path="docs/screenshots/S07b/job_list_filtered_failed.png")
        print("Captured job_list_filtered_failed.png")

        # 3. Test Timestamp Tooltip (Verify HTML attribute)
        try:
            title_el = page.locator(".job-row-meta span[title]").first
            title_attr = title_el.get_attribute("title")
            print(f"Timestamp Title: {title_attr}")
            if not title_attr:
                print("FAILURE: Timestamp title missing")
        except:
            print("FAILURE: Could not find timestamp element with title")

        browser.close()

if __name__ == "__main__":
    run()
