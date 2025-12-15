# Inference Flow Analysis (S05)

**Date:** 2025-07-05
**Reviewer:** Jules

## 1. Overview
The inference flow allows users to submit a media file (and optional transcript) for caption generation. It uses a clean form with file pickers and an advanced configuration override editor.

## 2. Walkthrough Observations

### Initial State
- **Form Fields:** Media file (required), Transcript (optional), Output directory (optional), Model config (optional), Operator notes.
- **Overrides:** Collapsed/Tabbed "Per-run overrides" section separating Pipeline and Segmentation configs.
- **Guidance:** Basic subtitle "Provide a media file..." and helper text on file pickers.

### Interaction
- **File Selection:** `FilePathPicker` provides good validation feedback ("Validating...", "Path verified").
- **Overrides:** The `OverrideEditor` is powerful but exposes the entire raw configuration tree. Users must know exactly where `do_diarization` or `beam_width` lives to change them.
- **Submission:** Toast feedback "Inference job queued" is clear.

### Pain Points
1.  **Hidden Controls:** Common settings like **Diarization** (on/off) and **Beam Width** (accuracy vs speed) are buried in the override tree.
2.  **Lack of Presets:** No quick way to select "Draft" vs "Broadcast" quality without manually tweaking parameters.
3.  **Model Config Ambiguity:** The "Model config" field accepts a file path but doesn't explain *why* a user would provide one versus using overrides. It overrides the base `config.yaml`.
4.  **Transcript Usage:** It's not immediately obvious that providing a transcript switches the mode from "ASR-only" to "Alignment".

## 3. Recommended UX Improvements

### High Priority
-   **Surface Key Toggles:** Move critical flags out of the override tree into the main form as first-class citizens:
    -   `Diarization` (Checkbox, defaults to config value).
    -   `Beam Width` (Slider or Number, e.g., 1-10).
-   **Add Presets:** Introduce a "Preset" dropdown that pre-fills overrides/toggles.
    -   *Standard* (Default)
    -   *High Precision* (Higher beam width, aggressive refinement)
    -   *Fast Draft* (Lower beam width, no refinement)

### Medium Priority
-   **Clarify Tooltips:** Add info icons or expanded helper text.
    -   *Transcript:* "Upload a corrected script to align heavily edited text."
    -   *Model Config:* "Advanced: Load a full alternative configuration file."
-   **Validation:** Ensure "Output directory" validation clearly states if it will be created.

## 4. Artifacts
-   `docs/screenshots/S05/inference_empty.png`: Initial form state.
-   `docs/screenshots/S05/inference_filled.png`: Form with valid inputs.
-   `docs/screenshots/S05/inference_submitted.png`: Success toast state.
