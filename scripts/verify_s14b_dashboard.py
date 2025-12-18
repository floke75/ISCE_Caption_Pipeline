from playwright.sync_api import sync_playwright
import json
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 1024})
        page = context.new_page()

        # Mock jobs API
        page.route("**/api/jobs", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps([
                {
                    "id": "job_training_mock",
                    "jobType": "training_pair",
                    "status": "succeeded",
                    "createdAt": "2023-10-27T10:00:00Z",
                    "updatedAt": "2023-10-27T10:05:00Z",
                    "workspacePath": "/tmp/job_training_mock",
                    "params": {},
                    "result": {
                        "training_json": "/mock/training.json",
                        "asr_reference": "/mock/asr.json"
                    }
                }
            ])
        ))

        # Handle health check requests
        page.route("**/api/health", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({
                "status": "healthy",
                "system": {
                    "disk": {"free_bytes": 10000000000, "total_bytes": 100000000000, "percent_used": 10},
                    "memory": {"available_bytes": 8000000000, "total_bytes": 16000000000, "percent_used": 50},
                    "gpu": {"available": False, "name": None, "device_count": 0}
                },
                "queue": {"pending": 0, "active": 0, "slots_total": 4}
            })
        ))

        # Mock logs
        page.route("**/api/jobs/job_training_mock/logs", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"log": "Mock log..."})
        ))

        # Mock artifact content
        def handle_content(route):
            tokens = []
            for i in range(100):
                tokens.append({
                    "w": "word",
                    "start": i * 0.5,
                    "end": i * 0.5 + 0.4,
                    "pause_after_ms": 600 if i % 10 == 0 else 50,
                    "break_type": "LB" if i % 15 == 0 else "O",
                    "speaker_change": i % 20 == 0
                })
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(tokens)
            )

        page.route("**/files/content?path=%2Fmock%2Ftraining.json", handle_content)

        # Attach console/page error handlers
        def on_console(msg):
             print(f"PAGE CONSOLE: {msg.text}")

        def on_page_error(err):
             print(f"PAGE ERROR: {err}")

        page.on("console", on_console)
        page.on("pageerror", on_page_error)

        # Navigate
        page.goto("http://localhost:5173/")

        # Take a screenshot to see what's happening if it fails
        try:
             page.wait_for_selector(".job-row-title")
        except:
             page.screenshot(path="docs/screenshots/S14b/debug_fail.png")
             raise

        # Interact
        # Click the job in the list (not the tab)
        page.locator(".job-row-title").filter(has_text="Training pair").click()
        page.get_by_role("button", name="Data Quality").click()
        page.wait_for_selector("text=Data Quality Metrics")

        # Screenshot
        os.makedirs("docs/screenshots/S14b", exist_ok=True)
        page.screenshot(path="docs/screenshots/S14b/dashboard_implementation.png")
        print("Screenshot captured.")

        browser.close()

if __name__ == "__main__":
    run()
