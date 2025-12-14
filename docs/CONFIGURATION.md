# Configuration loading guide

This repository uses a two-part configuration pattern:

1. **`pipeline_config.py` (module)** – ships with the codebase and exposes
   `load_pipeline_config(default_settings, yaml_path="pipeline_config.yaml")`.
   The loader deep-merges YAML values over the provided defaults and resolves
   path placeholders such as `{project_root}` and `{pipeline_root}`.
2. **`pipeline_config.yaml` (user-editable YAML)** – lives alongside the code
   and contains environment-specific overrides. The orchestrators and stage
   scripts read this YAML (or an explicitly supplied alternative) through the
   loader above. Optional local copies (for example, `pipeline_config.local.yaml`)
   can be passed with `--config-file` without changing code.

## How scripts pick up configuration

- **`run_pipeline.py`** calls `load_pipeline_config(DEFAULT_SETTINGS)` at startup
  and expects overrides in `pipeline_config.yaml` unless `--config-file` points
  elsewhere.
- **`align_make.py`** and **`build_training_pair_standalone.py`** accept
  `--config-file` flags. They pass the resulting path into
  `load_pipeline_config`, so the same YAML layout works across stages.
- **UI backend** reuses `pipeline_config.yaml` through `pipeline_config.py` when
  staging jobs. UI overrides are merged separately by `ui/backend/config_service.py`.

## Recommended flow

1. Keep defaults portable (relative `{project_root}` placeholders) inside each
   script's `DEFAULT_SETTINGS`.
2. Copy the committed template (`pipeline_config.example.yaml`) to a local YAML
   file and adjust paths or tokens there.
3. When running any stage manually, provide `--config-file <your_yaml>` if the
   overrides are not stored in the root `pipeline_config.yaml`.

This separation keeps the loader logic in code (`pipeline_config.py`) while user
settings stay in YAML files checked into or ignored by git as appropriate.
