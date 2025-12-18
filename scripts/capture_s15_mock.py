import os
from playwright.sync_api import sync_playwright

def capture():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        # Load the mock HTML file
        mock_path = os.path.abspath("tests/e2e/mock_help_center.html")
        page.goto(f"file://{mock_path}")

        # Ensure directory exists
        os.makedirs("docs/screenshots/S15", exist_ok=True)

        # Capture screenshot
        page.screenshot(path="docs/screenshots/S15/help_center_design.png", full_page=True)
        print("Captured help center design mock.")
        browser.close()

if __name__ == "__main__":
    capture()
