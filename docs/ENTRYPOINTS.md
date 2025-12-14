# Entry points and orchestration

This repository ships two orchestration surfaces that drive the same canonical pipeline chain (`align_make.py -> build_training_pair_standalone.py -> main.py`). The FastAPI job runner used by the Web UI is the **primary** entrypoint for production-style runs; the hot-folder script (`run_pipeline.py`) remains available as a **legacy**/offline alternative.

## Canonical chain (UI job runner)
The UI backend (`ui/backend/pipelines.py`) calls the three stage scripts in order. Each invocation is logged via `ctx.stream_command` with explicit flags:

- **Audio alignment**
  ```bash
  python align_make.py --input-file <media> --out-root <intermediate_dir> --config-file <pipeline_config.yaml>
  ```
- **Token enrichment** (uses an edited transcript when present; otherwise enters ASR-only mode)
  ```bash
  python build_training_pair_standalone.py \
    --primary-input <transcript_or_asr> \
    --asr-reference <intermediate_dir/_align/<name>.asr.visual.words.diar.json> \
    --out-inference-dir <intermediate_dir/_inference_input> \
    --config-file <pipeline_config.yaml> \
    --output-basename <name> [--asr-only-mode]
  ```
- **Segmentation**
  ```bash
  python main.py --input <intermediate_dir/_inference_input/<name>.enriched.json> --output <output_dir/<name>.srt> --config <config.yaml>
  ```

Training pair jobs follow the same first step and then call:
```bash
python build_training_pair_standalone.py \
  --primary-input <captions.srt> \
  --asr-reference <intermediate_dir/_align/<name>.asr.visual.words.diar.json> \
  --out-training-dir <intermediate_dir/_training> \
  --config-file <pipeline_config.yaml>
```

## Legacy hot-folder orchestrator
`run_pipeline.py` polls drop folders and invokes the same stage scripts. The assembled commands mirror the UI runner:

- Inference alignment: `python align_make.py --input-file <media> --out-root <intermediate_dir> --config-file <pipeline_config.yaml>`
- Inference enrichment (adds `--asr-only-mode --output-basename <name>` when no TXT is present):
  `python build_training_pair_standalone.py --primary-input <txt_or_asr> --asr-reference <…asr.visual.words.diar.json> --out-inference-dir <intermediate_dir/_inference_input> --config-file <pipeline_config.yaml> [--asr-only-mode --output-basename <name>]`
- Inference segmentation: `python main.py --input <…enriched.json> --output <output_dir/<name>.srt> --config <config.yaml>`
- Training pair creation: `python build_training_pair_standalone.py --primary-input <captions.srt> --asr-reference <…asr.visual.words.diar.json> --out-training-dir <intermediate_dir/_training> --config-file <pipeline_config.yaml>`

Both orchestrators rely on the same stage scripts and configs; prefer the UI job runner unless you explicitly need the hot-folder automation.
