import time
import json
import os
from playwright.sync_api import sync_playwright

MOCK_DIR = "ui_data/mocks"
ENRICHED_PATH = os.path.join(MOCK_DIR, "mock.enriched.json")
ASR_PATH = os.path.join(MOCK_DIR, "mock.asr.visual.words.diar.json")

def load_mock(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        # Load mock data
        enriched_data = load_mock(ENRICHED_PATH)
        asr_data = load_mock(ASR_PATH)

        # Mock API routes
        def handle_jobs(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps([
                    {
                        "id": "job-inference-mock",
                        "jobType": "inference",
                        "status": "succeeded",
                        "createdAt": "2023-10-27T10:00:00Z",
                        "updatedAt": "2023-10-27T10:05:00Z",
                        "workspacePath": "/tmp/workspace",
                        "params": {"media_path": "video.mp4"},
                        "result": {
                            "enriched_tokens": "mock.enriched.json",
                            "asr_reference": "mock.asr.visual.words.diar.json"
                        },
                        "progress": 1.0,
                        "message": "Complete"
                    }
                ])
            )

        def handle_file_content(route):
            url = route.request.url
            if "mock.enriched.json" in url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"content": json.dumps(enriched_data)})
                )
            elif "mock.asr.visual.words.diar.json" in url:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps({"content": json.dumps(asr_data)})
                )
            else:
                route.continue_()

        page.route("**/api/jobs", handle_jobs)
        page.route("**/api/files/content*", handle_file_content)

        # Navigate to Dashboard
        print("Navigating to Dashboard...")
        page.goto("http://localhost:5173/")
        time.sleep(2) # Wait for load

        # Click the job
        print("Clicking job...")
        page.wait_for_selector(".job-row")
        page.click(".job-row")
        time.sleep(1)

        # Check for button
        print("Checking for 'Visualise Alignment' button...")
        button = page.locator("text=Visualise Alignment")
        if button.count() > 0:
            print("Button found!")
        else:
            print("Button NOT found!")
            page.screenshot(path="docs/screenshots/S10b/debug_no_button.png")
            browser.close()
            return

        # Click button
        button.click()
        time.sleep(2)

        # Verify Alignment Viewer
        print("Verifying Alignment Viewer...")
        content = page.content()
        if "Inference Alignment" in content:
            print("Header verified.")
        else:
            print("Header missing!")
            if "Training Alignment" in content:
                print("Found 'Training Alignment' instead.")
            if "Error loading artifacts" in content:
                print("Found error message.")
            if "Artifacts loaded but empty" in content:
                print("Found empty artifacts message.")
            # Print a snippet
            print("Page text snippet:", page.locator("body").inner_text()[:500])

        # Screenshot Default (Lines)
        os.makedirs("docs/screenshots/S10b", exist_ok=True)
        page.screenshot(path="docs/screenshots/S10b/inference_lines_view.png")
        print("Captured Lines View")

        # Toggle to Sentences
        print("Toggling to Sentences...")
        page.click("text=Sentences")
        time.sleep(1)

        # Screenshot Sentences
        page.screenshot(path="docs/screenshots/S10b/inference_sentences_view.png")
        print("Captured Sentences View")

        browser.close()

if __name__ == "__main__":
    run()
