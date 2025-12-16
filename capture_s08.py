from playwright.sync_api import sync_playwright

def capture_s08_baseline():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()

        # Route mocks to ensure we see a populated job board
        def handle_jobs(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body="""
                [{
                    "id": "job-123",
                    "jobType": "inference",
                    "status": "succeeded",
                    "progress": 1.0,
                    "message": "Inference complete",
                    "createdAt": "2023-01-01T12:00:00Z",
                    "updatedAt": "2023-01-01T12:05:00Z",
                    "params": {"media_path": "/data/test.mp4", "output_dir": "/data/out"},
                    "result": {"srt_path": "/data/out/test.srt", "json_path": "/data/out/test.json"},
                    "workspacePath": "/data/workspaces/job-123"
                }]
                """
            )

        page.route("**/api/jobs", handle_jobs)

        # Navigate to the app (using the port from the log, usually 5173 or 5174)
        # We saw 5174 in the log
        try:
            page.goto("http://localhost:5174")
            page.wait_for_selector(".job-row")
            page.click(".job-row") # Select the job
            page.wait_for_timeout(1000) # Wait for details

            page.screenshot(path="docs/screenshots/S08/baseline_job_details.png")
            print("Screenshot captured.")
        except Exception as e:
            print(f"Error: {e}")

        browser.close()

if __name__ == "__main__":
    capture_s08_baseline()
