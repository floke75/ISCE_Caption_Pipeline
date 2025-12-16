import sys
import os
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 720})

        # Mock API to prevent errors
        page.route("**/api/jobs", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='[{"id":"mock1","jobType":"inference","status":"succeeded","progress":1.0,"message":"Done","createdAt":"2023-01-01T00:00:00","updatedAt":"2023-01-01T00:00:00","params":{},"workspacePath":"/tmp"}]'
        ))
        page.route("**/api/health", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"ok"}'
        ))

        try:
            print("Navigating to http://localhost:5173...")
            page.goto("http://localhost:5173", timeout=30000)
            print("Waiting for header...")
            page.wait_for_selector("header.app-header", timeout=10000)

            # Screenshot the header area
            header = page.locator("header.app-header")
            os.makedirs("docs/screenshots/S11", exist_ok=True)
            print("Taking screenshot...")
            header.screenshot(path="docs/screenshots/S11/baseline_header.png")
            print("Screenshot captured at docs/screenshots/S11/baseline_header.png")
        except Exception as e:
            print(f"Error: {e}")
            # Take a full page screenshot for debugging
            try:
                page.screenshot(path="debug_error.png")
                print("Saved debug_error.png")
            except:
                pass
            sys.exit(1)
        finally:
            browser.close()

if __name__ == "__main__":
    run()
