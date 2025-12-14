# Deletions Log

## scripts/evaluate_model.py (relocated to `legacy/evaluate_model.py`)
- **Reason for relocation:** Vulture scan flagged the script as unused and it depended on a missing `isce/evaluate.py`; moving it to `legacy/` keeps it out of the supported surface while preserving the reference implementation for possible revival.
- **Safety check:** The script is not referenced by the UI job runner, hot-folder orchestrator, pipelines, or tests, so moving it does not affect supported execution paths.
- **Replacement:** No direct replacement; future evaluation tooling should be built intentionally or by reviving this legacy stub with maintained dependencies.
