# UI Baseline & Navigation Audit

**Date:** 2025-07-05
**Step:** S01
**Environment:**
- **Node:** v20.19.5
- **NPM:** 11.4.2 (approx)
- **Vite:** v5.4.21
- **Backend:** FastAPI (Uvicorn 0.38.0)
- **Frontend Port:** 5173
- **Backend Port:** 8000

## Navigation Structure

The application uses a single-page architecture with a persistent layout.

### Primary Navigation (Tabs)
Located at the top of the workbench area.
1.  **Inference** (Default) - `InferenceForm`
2.  **Training pairs** - `TrainingPairForm`
3.  **Model training** - `ModelTrainingForm`
4.  **Configuration** - `ConfigPanel`

### Secondary Navigation (Configuration)
Within the Configuration tab, there are likely sub-tabs or sections:
- Pipeline configuration
- Segmentation configuration
(inferred from playwright selector ambiguity)

### Persistent Elements
- **Header:** Title "ISCE Pipeline Control Center" and repository link.
- **Sidebar (Right):** `JobBoard` showing job list and detailed logs/artifacts.

## Captured Baseline
Screenshots are stored in `docs/screenshots/S01/`.

1.  `inference-form.png`: The default landing state. Shows the inference form and the empty/initial job board.
2.  `training-pair-form.png`: The training data generation interface.
3.  `model-training-form.png`: The model training interface.
4.  `config-panel.png`: The configuration editor interface.

## Observations
- The UI loaded successfully without visible errors in the console logs (based on `frontend.log` clean startup).
- Navigation between tabs is instantaneous (client-side state).
- The Job Board sidebar remains visible across all tabs, allowing monitoring while configuring new jobs.
