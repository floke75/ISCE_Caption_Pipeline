from playwright.sync_api import sync_playwright

def verify_s08b_artifact_viewer():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()

        # Debug console logs
        page.on("console", lambda msg: print(f"BROWSER LOG: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))

        # Mock Jobs
        def handle_jobs(route):
            # print("Intercepted /api/jobs")
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

        # Mock File Content
        def handle_file_content(route):
            print("Intercepted /api/files/content")
            route.fulfill(
                status=200,
                content_type="application/json",
                body="""
                {
                    "path": "/data/out/test.srt",
                    "content": "1\\n00:00:00,000 --> 00:00:02,000\\nHello World\\n\\n2\\n00:00:02,500 --> 00:00:04,000\\nThis is a test subtitle.",
                    "size": 100,
                    "mimeType": "text/plain",
                    "truncated": false
                }
                """
            )

        page.route("**/api/jobs", handle_jobs)
        page.route("**/api/files/content*", handle_file_content)

        try:
            print("Navigating to http://localhost:5173")
            page.goto("http://localhost:5173")

            print("Waiting for .job-row...")
            page.wait_for_selector(".job-row", timeout=10000)

            print("Clicking job row...")
            page.click(".job-row")
            page.wait_for_timeout(500)

            # Screenshot of "View" link
            page.screenshot(path="docs/screenshots/S08b/job_details_with_link.png")
            print("Captured Job Details with Link")

            # 2. Direct Navigation Test
            print("Testing direct navigation...")
            page.goto("http://localhost:5173/artifacts/view?path=/data/out/test.srt")

            print(f"Current URL: {page.url}")

            # 3. Verify Viewer
            print("Waiting for viewer content...")
            page.wait_for_selector("text=Hello World", timeout=5000)
            page.wait_for_selector("text=Download Raw")

            # Screenshot of Viewer
            page.screenshot(path="docs/screenshots/S08b/artifact_viewer.png")
            print("Captured Artifact Viewer")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="docs/screenshots/S08b/error.png")
            print("Captured error.png")

        browser.close()

if __name__ == "__main__":
    verify_s08b_artifact_viewer()
