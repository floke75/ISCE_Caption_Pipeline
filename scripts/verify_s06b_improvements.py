import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 800})
        page = context.new_page()

        # Route validation to always succeed
        def handle_validation(route):
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"allowed": true, "exists": true, "isFile": true, "isDir": false, "detail": "Mocked valid"}'
            )

        page.route("**/api/files/validate**", handle_validation)

        print("Navigating to app...")
        page.goto("http://localhost:5173/")
        page.wait_for_selector("text=ISCE Pipeline Control Center")

        # --- Test 1: Inference Form Invalid (Disabled) ---
        print("Testing Inference Form Invalid (Disabled Button)...")
        page.click("text=Inference")
        page.wait_for_selector("h2:has-text('Run inference')")

        # Check disabled state immediately (empty form)
        is_disabled = page.is_disabled("button:has-text('Launch inference run')")
        print(f"Inference Submit Button Disabled (Empty): {is_disabled}")

        page.screenshot(path="docs/screenshots/S06b/inference_disabled.png")

        if not is_disabled:
            print("FAILURE: Button should be disabled")
            # browser.close()
            # return

        # --- Test 2: Inference Form Valid (Enabled) ---
        print("Testing Inference Form Valid (Enabled Button)...")

        # Fill form
        page.fill("input[placeholder='/data/media.mp4']", "/mock/media.mp4")
        page.click("body") # Trigger validation

        print("Waiting for validation and button enable...")
        try:
            page.wait_for_selector("button:has-text('Launch inference run'):not([disabled])", timeout=5000)
            print("Button enabled")
        except:
            print("Button did not enable within timeout")

        is_disabled = page.is_disabled("button:has-text('Launch inference run')")
        print(f"Inference Submit Button Disabled (Filled): {is_disabled}")

        page.screenshot(path="docs/screenshots/S06b/inference_enabled.png")

        # --- Test 3: Submission and Error ---
        print("Testing Submission Error Toast...")
        def handle_inference_500(route):
            route.fulfill(status=500, content_type="application/json", body='{"detail": "Mocked Internal Server Error"}')
        page.route("**/api/jobs/inference", handle_inference_500)

        if not is_disabled:
            page.click("button:has-text('Launch inference run')")
            try:
                page.wait_for_selector("text=Mocked Internal Server Error", timeout=5000)
                print("Backend error toast appeared")
            except:
                print("Error toast not found")
            page.screenshot(path="docs/screenshots/S06b/inference_error_toast.png")

        browser.close()

if __name__ == "__main__":
    run()
