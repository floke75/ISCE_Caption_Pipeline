# S01 baseline UI captures (text references)

Binary screenshots were removed to keep the pull request free of large image assets. This README documents the intended captures and how to reproduce them if needed:

- **inference-form** — inference tab showing media/transcript pickers, override editor, and job board sidebar with proxy errors while backend is offline.
- **training-pair-form** — training pairs tab with source/target path pickers and submission disabled until validation passes.
- **model-training-form** — model training tab featuring training corpus selectors and disabled submit button pending validation.
- **config-panel** — configuration tab with pipeline and segmentation sub-tabs plus reset/save controls.
- **job-board** — job board view demonstrating empty state and proxy error banners when the API is unavailable.

To recreate: start the frontend with `npm run dev -- --host 0.0.0.0 --port 8001 --strictPort`, browse each tab listed above, and capture new screenshots into this directory.
