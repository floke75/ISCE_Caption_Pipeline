from playwright.sync_api import sync_playwright

def verify_training_flow_improved():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to home
        page.goto("http://localhost:5173")

        # Navigate to Training Pairs
        page.get_by_text("Training pairs").click()
        page.wait_for_selector("h2.section-title:has-text('Build training pair')")

        # Verify button is disabled initially (empty invalid form)
        button = page.get_by_role("button", name="Launch training-pair job")
        if button.is_enabled():
            print("FAILURE: Submit button should be disabled for empty form")
        else:
            print("SUCCESS: Submit button is disabled for empty form")

        # Capture initial state with improved help/button state
        page.screenshot(path="docs/screenshots/S04/training_pair_improved_initial.png")

        # Fill invalid paths to trigger inline validation (which shouldn't enable button)
        page.get_by_placeholder("/data/media.mp4").fill("/invalid/path.mp4")
        page.get_by_placeholder("/data/captions.srt").fill("/invalid/caps.srt")
        page.wait_for_timeout(500) # wait for validation

        page.screenshot(path="docs/screenshots/S04/training_pair_improved_invalid.png")

        if button.is_enabled():
             print("FAILURE: Submit button enabled despite invalid paths")
        else:
             print("SUCCESS: Submit button remains disabled for invalid paths")

        # Navigate to Model Training
        page.get_by_text("Model training").click()
        page.wait_for_selector("h2.section-title:has-text('Train statistical model')")

        # Capture initial state (should show new help text)
        page.screenshot(path="docs/screenshots/S04/model_training_improved.png")

        # Verify button disabled
        train_btn = page.get_by_role("button", name="Launch training run")
        if train_btn.is_enabled():
             print("FAILURE: Training button should be disabled for empty corpus")
        else:
             print("SUCCESS: Training button is disabled for empty corpus")

        browser.close()

if __name__ == "__main__":
    verify_training_flow_improved()
