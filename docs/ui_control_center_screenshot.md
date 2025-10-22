# Control Center UI Screenshot

This document tracks the latest control center dashboard screenshot generated during automated PR review. The image itself is attached to the corresponding pull request run artifacts rather than stored in the repository to avoid bloating the repo size.

To reproduce the screenshot locally:

1. Install the lightweight API dependencies:

   ```bash
   pip install fastapi uvicorn pyyaml
   ```

2. Start the FastAPI server from the repository root:

   ```bash
   uvicorn ui.server:app --host 0.0.0.0 --port 8000
   ```

3. From another shell, capture the dashboard using Playwright (requires `playwright` to be installed and browsers set up):

   ```bash
   playwright install
   playwright screenshot http://127.0.0.1:8000 docs/control-center-ui.png
   ```

4. Commit the updated screenshot if a checked-in artifact is required.

The current PR run captured the UI and attached the image as `control-center-ui.png` in the workflow artifacts.
