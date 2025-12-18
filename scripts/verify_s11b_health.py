import sys
import os
import time
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 720})

        # Mock Jobs API
        page.route("**/api/jobs", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='[]'
        ))

        # Scenario 1: Healthy CPU
        page.route("**/api/health", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"ok","system":{"disk":{"free_bytes":50000000000,"total_bytes":100000000000,"percent_used":50.0},"memory":{"available_bytes":8000000000,"total_bytes":16000000000,"percent_used":50.0},"gpu":{"available":false,"name":null,"device_count":0}},"queue":{"pending":0,"active":0,"slots_total":3}}'
        ))

        print("Capturing Healthy CPU...")
        try:
            page.goto("http://localhost:5173", timeout=30000)
            page.wait_for_selector(".system-status-container", timeout=10000)
            # Hover to show popover
            page.locator(".system-status-container").hover()
            time.sleep(0.5)
            os.makedirs("docs/screenshots/S11b", exist_ok=True)
            page.locator("header.app-header").screenshot(path="docs/screenshots/S11b/system_healthy_cpu.png")
            print("Captured system_healthy_cpu.png")
        except Exception as e:
            print(f"Error healthy: {e}")

        # Scenario 2: Disk Full
        page.route("**/api/health", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"ok","system":{"disk":{"free_bytes":100000000,"total_bytes":100000000000,"percent_used":99.9},"memory":{"available_bytes":8000000000,"total_bytes":16000000000,"percent_used":50.0},"gpu":{"available":false,"name":null,"device_count":0}},"queue":{"pending":0,"active":0,"slots_total":3}}'
        ))
        print("Capturing Disk Full...")
        try:
            page.reload()
            page.wait_for_selector(".system-status-container", timeout=10000)
            page.locator(".system-status-container").hover()
            time.sleep(0.5)
            page.locator("header.app-header").screenshot(path="docs/screenshots/S11b/system_disk_full.png")
            print("Captured system_disk_full.png")
        except Exception as e:
            print(f"Error disk full: {e}")

        # Scenario 3: High Memory
        page.route("**/api/health", lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body='{"status":"ok","system":{"disk":{"free_bytes":50000000000,"total_bytes":100000000000,"percent_used":50.0},"memory":{"available_bytes":100000000,"total_bytes":16000000000,"percent_used":95.0},"gpu":{"available":false,"name":null,"device_count":0}},"queue":{"pending":0,"active":0,"slots_total":3}}'
        ))
        print("Capturing High Memory...")
        try:
            page.reload()
            page.wait_for_selector(".system-status-container", timeout=10000)
            page.locator(".system-status-container").hover()
            time.sleep(0.5)
            page.locator("header.app-header").screenshot(path="docs/screenshots/S11b/system_high_mem.png")
            print("Captured system_high_mem.png")
        except Exception as e:
            print(f"Error high mem: {e}")

        # Scenario 4: Error
        # For error, we abort request to simulate network failure or 500
        page.route("**/api/health", lambda route: route.fulfill(status=500))
        print("Capturing Error...")
        try:
            page.reload()
            page.wait_for_selector(".system-status.error", timeout=10000)
            page.locator("header.app-header").screenshot(path="docs/screenshots/S11b/system_error.png")
            print("Captured system_error.png")
        except Exception as e:
            print(f"Error failure state: {e}")

        browser.close()

if __name__ == "__main__":
    run()
