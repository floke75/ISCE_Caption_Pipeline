import time
from playwright.sync_api import sync_playwright, expect

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        # Mock jobs endpoint
        mock_jobs = [
            {
                "id": "job-inf-completed",
                "jobType": "inference",
                "status": "succeeded",
                "created_at": "2023-10-27T10:00:00.000000",
                "updated_at": "2023-10-27T10:05:00.000000",
                "message": "Success",
                "progress": 1.0,
                "workspace_path": "/data/workspaces/job-inf-completed",
                "params": {
                    "media_path": "/data/media/movie.mp4",
                    "output_dir": "/data/workspaces/job-inf-completed",
                    "model_config_path": "config.yaml"
                },
                "result": {
                    "output_srt": "/data/workspaces/job-inf-completed/movie.srt",
                    "enriched_tokens": "/data/workspaces/job-inf-completed/movie.enriched.json",
                    "asr_reference": "/data/workspaces/job-inf-completed/movie.asr.json"
                },
                "error": None
            }
        ]

        page.route("**/api/jobs", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=str(mock_jobs).replace("'", '"').replace("None", "null")
        ))

        # Mock specific job details if needed (JobBoard uses the list data usually)

        print("Navigating to Job Board...")
        page.goto("http://localhost:5173/")

        # Click "Job monitor" tab (it's actually the JobBoard sidebar, but let's just wait for it)
        # The JobBoard is always visible? No, it's a sidebar.
        # Wait, the screenshots show it as a persistent right panel?
        # Let's check App.tsx layout.
        # But for now, we just wait for "Job monitor" text.

        expect(page.get_by_text("Job monitor")).to_be_visible()

        # Click the job to select it
        page.get_by_text("Success").click()

        # Wait for details
        expect(page.get_by_text("Results")).to_be_visible()

        # Capture screenshot of the Details and Results panels
        # We can target the .job-details class
        details_panel = page.locator(".job-details")
        details_panel.screenshot(path="docs/screenshots/S08/job_details_artifacts.png")
        print("Captured docs/screenshots/S08/job_details_artifacts.png")

        # Analyze visibility
        # Check if the path is a link
        # In the code, it's just <code className="path-value">
        # So we expect NO link.

        links = details_panel.get_by_role("link").all()
        print(f"Found {len(links)} links in details panel.")

        browser.close()

if __name__ == "__main__":
    run()
