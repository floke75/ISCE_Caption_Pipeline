# Testing guide

This repository ships targeted unit tests and a lightweight smoke path that avoids heavyweight ASR/diarization downloads. The notes below focus on running tests in constrained environments.

## Running the focused unit suite

Most development iterations can rely on the fast unit tests:

```bash
python -m pip install -r requirements.txt
pytest -q
```

See `docs/BASELINE.md` for the environment used to capture the cleanup baseline.

## Mocking Stage 1 for smoke tests

`align_make.py` now supports a deterministic mock mode for CI and local smoke checks. When you pass `--mock-asr-json`, the script skips audio extraction, WhisperX, and diarization, and instead copies the provided JSON to the expected `_align` location for downstream stages.

Example (using the bundled fixtures):

```bash
rm -rf tests/_artifacts
mkdir -p tests/_artifacts
python align_make.py \
  --input-file tests/fixtures/demo.mp4 \
  --out-root tests/_artifacts \
  --config-file tests/fixtures/pipeline_config.test.yaml \
  --mock-asr-json tests/fixtures/demo.asr.visual.words.diar.json
```

The command writes `tests/_artifacts/_align/demo.asr.visual.words.diar.json` without reading the media file or downloading any models. This makes it safe to chain with later stages in offline environments.
