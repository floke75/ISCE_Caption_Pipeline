# UI Guidance Inventory and Gaps (S02)

**Date:** 2025-07-05
**Reviewer:** Jules

## 1. Existing Guidance Inventory

### Header
- **Link:** "Repository" -> `https://github.com/floke75/ISCE_Caption_Pipeline`
- **Badge:** "Beta" indicator.

### Inference Form
- **Subtitles:** "Provide a media file and optional transcript to generate an SRT subtitle file."
- **File Pickers:**
  - "Media file path": Helper "Absolute media file path on the host". Placeholder example.
  - "Transcript": Placeholder "Optional".
  - "Output directory": Placeholder "Override output folder".
  - "Model config": Placeholder "config.yaml".
- **Validation:** Real-time validation message (e.g., "Validating path...", "Path verified", "Path does not exist").
- **Notes:** Placeholder "Optional instructions or labels for this run".

### Training Pair Form
- **Subtitles:** "Generate enriched training JSON from an SRT file and matching media."
- **File Pickers:**
  - "Media file path": Required.
  - "SRT file path": Required.
- **Notes:** Placeholder "Context for this corpus artifact".

### Model Training Form
- **Subtitles:** "Launch the iterative weighting loop using an enriched training corpus."
- **Inputs:**
  - "Iterations": Number input, min 1.
  - "Error boost factor": Number input, step 0.1.
- **Notes:** Placeholder "Optional".

### Configuration Panel
- **Subtitles:** "Edit the most common knobs with validation and type hints."
- **Field Help:** Displays `description` from backend config metadata.
- **Advanced:** "Show advanced" toggle hides less common fields.
- **Overrides:** "Raw overrides" editor with "Reset overrides" button.

## 2. Evidence
Screenshots stored in `docs/screenshots/S02/`:
- `inference_guidance.png`
- `training_pair_guidance.png`
- `model_training_guidance.png`
- `config_panel_guidance.png`

## 3. Identified Gaps

### High Priority
1.  **Model Training Parameters:**
    - "Error boost factor" and "Iterations" lack explanation. Users do not know typical values or the impact of these settings.
    - **Action:** Add tooltips or explicit helper text explaining the trade-off (e.g., "Higher boost penalizes errors more aggressively").

2.  **Config Overrides:**
    - The "Raw overrides" section is opaque. Users don't know the schema without guessing or reading code.
    - **Action:** Provide a schema reference or example snippets.

3.  **SRT Requirements:**
    - `TrainingPairForm` does not specify SRT encoding (UTF-8) or formatting constraints.
    - **Action:** Add helper text to SRT picker.

### Medium Priority
1.  **Global Help Access:**
    - The only external link is to the repo root.
    - **Action:** Add a "Help" menu or links to specific doc sections (e.g., "How to run inference", "Training guide").

2.  **Inference Defaults:**
    - "Model config" is optional, but it's not clear what the default is (internal default vs. `config.yaml` in root).
    - **Action:** Clarify fallback behavior in helper text.

3.  **Diarization Toggles:**
    - (Observed in code/config but not prominent in Inference form screenshots) Diarization is a key feature but might be hidden in overrides or config.
    - **Action:** Expose high-level toggles (like "Enable Diarization") in the main form instead of burying in config.

### Low Priority
1.  **Onboarding:**
    - No "First run" tour.
    - **Action:** Consider a simple dismissal banner for first-time users.
