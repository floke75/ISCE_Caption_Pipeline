from playwright.sync_api import sync_playwright

def verify_training_flow():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to home
        page.goto("http://localhost:5173")

        # Navigate to Training Pairs
        page.get_by_text("Training pairs").click()
        page.wait_for_selector("h2.section-title:has-text('Build training pair')")

        # Capture initial state
        page.screenshot(path="docs/screenshots/S04/training_pair_initial.png")

        # Attempt submission with empty form to trigger validation
        page.get_by_role("button", name="Launch training-pair job").click()

        # Wait a moment for validation toasts/messages (since they are debounced or toasts)
        # We look for the toast error "Provide valid media and SRT paths"
        # Since toasts are often transient, we wait for the text to appear
        try:
            page.wait_for_selector("div:has-text('Provide valid media and SRT paths')", timeout=2000)
            print("Validation toast appeared")
        except:
            print("Validation toast did not appear or timed out")

        # Capture validation error state
        page.screenshot(path="docs/screenshots/S04/training_pair_validation_error.png")

        # Fill invalid paths (text input)
        page.get_by_placeholder("/data/media.mp4").fill("/invalid/path.mp4")
        page.get_by_placeholder("/data/captions.srt").fill("/invalid/caps.srt")

        # Wait for validation message (debounced)
        page.wait_for_timeout(1000)

        # Capture invalid path state
        page.screenshot(path="docs/screenshots/S04/training_pair_invalid_paths.png")

        # Navigate to Model Training
        page.get_by_text("Model training").click()
        page.wait_for_selector("h2.section-title:has-text('Train statistical model')")

        # Capture initial state
        page.screenshot(path="docs/screenshots/S04/model_training_initial.png")

        # Attempt submission empty
        page.get_by_role("button", name="Launch training run").click()
        try:
            page.wait_for_selector("div:has-text('Select a valid corpus directory')", timeout=2000)
        except:
             pass
        page.screenshot(path="docs/screenshots/S04/model_training_validation_error.png")

        browser.close()

if __name__ == "__main__":
    verify_training_flow()
