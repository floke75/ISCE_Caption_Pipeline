from playwright.sync_api import sync_playwright
import json
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1024})

        # Mock health to avoid 500 error on header
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

        # Navigate
        page.goto("http://localhost:5173/")

        # Click the Help Center toggle "?"
        page.get_by_title("Open Help Center").click()

        # Verify it opened
        page.wait_for_selector("text=Quickstart Checklist")

        # Screenshot
        os.makedirs("docs/screenshots/S15b", exist_ok=True)
        page.screenshot(path="docs/screenshots/S15b/help_center_implementation.png")
        print("Screenshot captured.")

        browser.close()

if __name__ == "__main__":
    run()
