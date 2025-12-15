
import asyncio
from playwright.async_api import async_playwright
import urllib.parse
import re
import json

# Mock data to simulate the JSON files
TRAIN_JSON = """{
  "tokens": [
    {"w": "Hello", "start": 0.5, "end": 0.9, "cue_id": 1, "break_type": "O"},
    {"w": "world", "start": 1.0, "end": 1.5, "cue_id": 1, "break_type": "SB"},
    {"w": "This", "start": 2.0, "end": 2.3, "cue_id": 2, "break_type": "O"},
    {"w": "is", "start": 2.35, "end": 2.5, "cue_id": 2, "break_type": "O"},
    {"w": "a", "start": 2.55, "end": 2.6, "cue_id": 2, "break_type": "O"},
    {"w": "test", "start": 2.65, "end": 3.0, "cue_id": 2, "break_type": "SB"}
  ]
}"""

ASR_JSON = """{
  "words": [
    {"w": "Hello", "start": 0.52, "end": 0.88},
    {"w": "world", "start": 1.05, "end": 1.48},
    {"w": "This", "start": 2.01, "end": 2.29},
    {"w": "is", "start": 2.35, "end": 2.51},
    {"w": "test", "start": 2.7, "end": 3.1}
  ]
}"""

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()

        page.on("request", lambda request: print(f">> {request.method} {request.url}"))

        await page.route(re.compile(r".*/api/files/content.*"), lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"content": TRAIN_JSON if "train" in route.request.url else ASR_JSON})
        ))

        target_url = f"http://localhost:5173/jobs/alignment?train={urllib.parse.quote('/tmp/mock.train.words.json')}&asr={urllib.parse.quote('/tmp/mock.asr.visual.words.diar.json')}"
        print(f"Navigating to {target_url}...")

        try:
            await page.goto(target_url, timeout=10000)

            await page.wait_for_selector("h1:has-text('Training Alignment')", timeout=5000)
            await page.wait_for_selector("text=CUE #1", timeout=5000)

            print("Capturing screenshot...")
            await page.screenshot(path="docs/screenshots/S09b/alignment_viewer_implementation.png", full_page=True)
            print("Screenshot saved.")

        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path="docs/screenshots/S09b/error_implementation.png")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
