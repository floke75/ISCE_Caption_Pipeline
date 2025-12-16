import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        # Load local HTML file
        file_path = os.path.abspath("experiments/mockup_s10.html")
        page.goto(f"file://{file_path}")

        # Take screenshot
        output_path = "docs/screenshots/S10/design_mockup.png"
        page.screenshot(path=output_path)
        print(f"Screenshot saved to {output_path}")

        browser.close()

if __name__ == "__main__":
    run()
