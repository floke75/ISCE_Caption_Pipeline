# ISCE Caption Pipeline Baseline Snapshot

- **Branch:** work
- **Commit:** 2992f0da54b2029ebc8ebd762b7cb7e13e9a97d9
- **OS:** Linux 4187e600a408 6.12.13 #1 SMP Thu Mar 13 11:34:50 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
- **Python:** 3.12.12
- **Pip:** pip 25.2 from /root/.pyenv/versions/3.12.12/lib/python3.12/site-packages/pip (python 3.12)

## Top-level file inventory (depth ≤ 2)
- Root files: `AGENTS.md`, `FRONTEND.md`, `README.md`, `align_make.py`, `build_training_pair_standalone.py`, `config.yaml`, `isce_repo_cleanup_plan.json`, `main.py`, `pipeline_config.py`, `pipeline_config.yaml`, `pipeline_config.example.yaml`, `requirements.txt`, `run_pipeline.py`.
- Docs: `docs/BASELINE.md`, `docs/SECURITY_NOTES.md`, `docs/beam_search_walkthrough.md`, `docs/build_training_pair_comparison.md`, `docs/spacy_feature_impact.md`, `docs/alt_build_training_pair_standalone.py`.
- Python package (`isce/`): `beam_search.py`, `config.py`, `data_validation.py`, `io_utils.py`, `model_builder.py`, `postprocess.py`, `scorer.py`, `srt_writer.py`, `token_normalization.py`, `types.py`, `__init__.py`.
- Scripts: `scripts/dev_console.sh`, `scripts/evaluate_model.py`, `scripts/install.py`, `scripts/train_model.py`, `scripts/__init__.py`.
- Tests: `tests/conftest.py`, all Stage 1–3 coverage like `test_beam_search.py`, `test_build_training_pair.py`, `test_config.py`, `test_data_validation.py`, `test_enrichment_features.py`, `test_io_utils.py`, `test_main.py`, `test_model_builder_constraints.py`, `test_model_builder_features.py`, `test_postprocess.py`, `test_scorer.py`, `test_segment.py`, `test_srt_writer.py`, `test_token_normalization.py`, `test_train_model.py`, `test_training_data_integrity.py`, and fixtures under `tests/fixtures/`.
- UI/backend: `ui/backend/app.py`, `ui/backend/pipelines.py`, `ui/backend/config_service.py`, and API routes in `ui/backend/api/`.
- UI/frontend: Vite/React app under `ui/frontend/` with `package.json`, `tsconfig.json`, and `src/`.
- Data/workspace folders: `ui_data/` (job inputs/artifacts for UI flows).

## Baseline command outputs

```bash
git rev-parse HEAD
2992f0da54b2029ebc8ebd762b7cb7e13e9a97d9

python --version
Python 3.12.12

pip --version
pip 25.2 from /root/.pyenv/versions/3.12.12/lib/python3.12/site-packages/pip (python 3.12)

git status
On branch work
nothing to commit, working tree clean
```
