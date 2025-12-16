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

## End-to-end smoke command

For a single-command integration check that links all three stages together without WhisperX, use the smoke runner:

```bash
python scripts/smoke_e2e.py \
  --workdir tests/_artifacts \
  --media tests/fixtures/demo.mp4 \
  --transcript tests/fixtures/demo.txt \
  --mock-asr tests/fixtures/demo.asr.visual.words.diar.json \
  --pipeline-config tests/fixtures/pipeline_config.test.yaml \
  --segmentation-config tests/fixtures/config.test.yaml
```

The script writes intermediate artifacts under `tests/_artifacts/_align` and `tests/_artifacts/_inference_input`, then emits an SRT to `tests/_artifacts/output/demo.srt`. It exits non-zero if any expected file is missing.

## Audio processing without FFMPEG

In environments where the `ffmpeg` binary is unavailable (e.g., restricted containers), `align_make.py` implements a native fallback using `torchaudio` and `soundfile`.

- **Mechanism:** If `ffmpeg` fails or is not found, the script uses `torchaudio` to load the audio file, resample it to 16kHz mono, and save it as a temporary WAV file using `soundfile`. It also uses a native loader to pass the audio data directly to WhisperX, bypassing its internal `ffmpeg` calls.
- **Verification:** Run `pytest tests/test_audio_fallback_integration.py` to verify this fallback behavior.
- **Limitation:** This fallback supports audio formats handled by `soundfile`/`torchaudio` (WAV, FLAC, MP3). For complex video containers (MP4, MKV), it may fail if the underlying libraries cannot demux the stream without `ffmpeg`. In such cases, convert the input to WAV beforehand or use the `--mock-asr-json` mode.
