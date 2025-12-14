# ISCE Caption Pipeline Baseline Snapshot

- **Branch:** work
- **Commit:** 8820021c4d9465f66c279f2aacfc255ae5c87048
- **Python:** 3.12.12
- **OS:** Linux 4309c66b3daa 6.12.13 #1 SMP Thu Mar 13 11:34:50 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
- **Top-level inventory (depth ≤ 2):**
  - Root scripts/configs: `align_make.py`, `build_training_pair_standalone.py`, `main.py`, `run_pipeline.py`, `pipeline_config.py`, `pipeline_config.yaml`, `config.yaml`, `pipeline_config.example.yaml`, `requirements.txt`, `README.md`, `FRONTEND.md`.
  - Docs: `docs/beam_search_walkthrough.md`, `docs/build_training_pair_comparison.md`, `docs/spacy_feature_impact.md`, `docs/alt_build_training_pair_standalone.py`.
  - Python package: `isce/` (beam search, scorer, postprocess, srt writer, model builder, data validation).
  - Tests and fixtures: `tests/test_beam_search.py`, `tests/fixtures/`.
  - Scripts: `scripts/install.py`, `scripts/train_model.py`.
  - UI backend/frontend: `ui/backend/` (FastAPI app, pipelines, config service, API routes), `ui/frontend/` (React SPA assets).
  - Data/work dirs: `_intermediate/`, `_output/`, `ui_data/` (job inputs/artifacts folders referenced by UI).

## Baseline command outputs

```bash
git rev-parse HEAD
8820021c4d9465f66c279f2aacfc255ae5c87048

python --version
Python 3.12.12

pip --version
pip 25.2 from /root/.pyenv/versions/3.12.12/lib/python3.12/site-packages/pip (python 3.12)

git status
On branch work
nothing to commit, working tree clean
```
