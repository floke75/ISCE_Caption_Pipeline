import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        # Route validation to always succeed so we can test the form submission logic
        def handle_validation(route):
            # Mock validation as success with correct schema
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"allowed": true, "exists": true, "isFile": true, "isDir": false, "detail": "Mocked valid"}'
            )

        page.route("**/api/files/validate**", handle_validation)

        print("Navigating to app...")
        page.goto("http://localhost:5173/")
        page.wait_for_selector("text=ISCE Pipeline Control Center")

        # --- Test 1: Inference Form Invalid (Empty) ---
        print("Testing Inference Form Invalid...")
        page.click("text=Inference")
        page.wait_for_selector("h2:has-text('Run inference')")

        page.click("button:has-text('Launch inference run')")
        try:
            page.wait_for_selector("div[role='status']", timeout=2000)
            print("Toast appeared")
        except:
            print("Toast not found")
        page.screenshot(path="docs/screenshots/S06/inference_invalid_toast.png")

        # --- Test 2: Inference Form Backend Error ---
        print("Testing Inference Form Backend Error...")
        def handle_inference_500(route):
            route.fulfill(status=500, content_type="application/json", body='{"detail": "Mocked Internal Server Error"}')
        page.route("**/api/jobs/inference", handle_inference_500)

        page.fill("input[placeholder='/data/media.mp4']", "/mock/media.mp4")
        page.click("body")
        time.sleep(1)

        page.click("button:has-text('Launch inference run')")
        try:
            page.wait_for_selector("text=Mocked Internal Server Error", timeout=5000)
            print("Backend error toast appeared")
        except:
            print("Error toast not found")
        page.screenshot(path="docs/screenshots/S06/inference_backend_error.png")

        # --- Test 3: Training Form Invalid (Disabled Button) ---
        print("Testing Training Form Invalid...")
        page.click("text=Training pairs")
        page.wait_for_selector("h2:has-text('Build training pair')")
        page.screenshot(path="docs/screenshots/S06/training_invalid_disabled.png")

        is_disabled = page.is_disabled("button:has-text('Launch training-pair job')")
        print(f"Training Submit Button Disabled: {is_disabled}")

        # --- Test 4: Training Form Backend Error ---
        print("Testing Training Form Backend Error...")
        def handle_training_500(route):
            route.fulfill(status=500, content_type="application/json", body='{"detail": "Mocked Training Error"}')
        page.route("**/api/jobs/training-pair", handle_training_500)

        page.fill("input[placeholder='/data/media.mp4']", "/mock/media.mp4")
        page.fill("input[placeholder='/data/captions.srt']", "/mock/captions.srt")
        page.click("body")

        print("Waiting for validation and button enable...")
        try:
            page.wait_for_selector("button:has-text('Launch training-pair job'):not([disabled])", timeout=5000)
            print("Button enabled")
        except:
            print("Button did not enable within timeout")

        is_disabled = page.is_disabled("button:has-text('Launch training-pair job')")
        print(f"Training Submit Button Disabled (after fill): {is_disabled}")

        if not is_disabled:
            page.click("button:has-text('Launch training-pair job')")
            try:
                page.wait_for_selector("text=Mocked Training Error", timeout=5000)
                print("Training backend error toast appeared")
            except:
                print("Error toast not found")
            page.screenshot(path="docs/screenshots/S06/training_backend_error.png")

        browser.close()

if __name__ == "__main__":
    run()
