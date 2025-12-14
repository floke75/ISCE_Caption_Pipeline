# Deletions Log

## scripts/evaluate_model.py
- **Reason for removal:** Vulture scan flagged the script as unused, and it depended on a non-existent `isce/evaluate.py`, causing import errors if invoked.
- **Safety check:** The script was not referenced by the UI job runner, hot-folder orchestrator, pipelines, or tests; removal does not affect supported execution paths.
- **Replacement:** No direct replacement; model evaluation should use maintained tooling within the pipeline or future dedicated utilities.
