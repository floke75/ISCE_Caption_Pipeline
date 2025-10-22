"""Pipeline runners invoked by the :mod:`job_manager`."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from .job_manager import JobContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def _copy_into(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    if source.resolve() == target.resolve():
        return target
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _record_result(ctx: JobContext, result: Dict[str, Any]) -> None:
    ctx.finalize("succeeded", result=result)


def run_inference(ctx: JobContext) -> None:
    params = ctx.record.params
    media_path = Path(params["media_path"]).expanduser()
    transcript_path = params.get("transcript_path")
    output_dir_override = params.get("output_dir")
    overrides = params.get("config_overrides") or {}

    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    ctx.update(message="Preparing workspace", progress=0.05)
    runtime = ctx.effective_config(overrides)
    config_path = Path(runtime["__path__"])
    pipeline_root = Path(runtime.get("pipeline_root", ctx.record.workspace / "pipeline"))
    intermediate_dir = Path(runtime.get("intermediate_dir", pipeline_root / "_intermediate"))
    output_dir = Path(output_dir_override or runtime.get("output_dir", ctx.record.workspace / "output"))
    txt_dir = Path(runtime.get("txt_placement_folder", pipeline_root / "txt"))

    repo_root = Path(runtime.get("project_root", REPO_ROOT))
    model_config_path = Path(params.get("model_config_path", repo_root / "config.yaml"))

    inputs_dir = ctx.record.workspace / "inputs"
    media_copy = _copy_into(media_path, inputs_dir / "media")
    transcript_copy = None
    if transcript_path:
        transcript_path = Path(transcript_path).expanduser()
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        transcript_copy = _copy_into(transcript_path, inputs_dir / "text")

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    base_name = media_copy.stem
    ctx.update(message="Running audio alignment", progress=0.2)
    ctx.stream_command(
        [
            sys.executable,
            str(repo_root / "align_make.py"),
            "--input-file",
            str(media_copy),
            "--out-root",
            str(intermediate_dir),
            "--config-file",
            str(config_path),
        ],
        cwd=repo_root,
    )

    asr_reference = intermediate_dir / "_align" / f"{base_name}.asr.visual.words.diar.json"
    if not asr_reference.exists():
        raise FileNotFoundError(f"Expected ASR output was not created: {asr_reference}")

    ctx.update(message="Enriching tokens", progress=0.55)
    primary_input = transcript_copy if transcript_copy else asr_reference
    args = [
        sys.executable,
        str(repo_root / "build_training_pair_standalone.py"),
        "--primary-input",
        str(primary_input),
        "--asr-reference",
        str(asr_reference),
        "--out-inference-dir",
        str(intermediate_dir / "_inference_input"),
        "--config-file",
        str(config_path),
    ]
    if transcript_copy is None:
        args.extend(["--asr-only-mode", "--output-basename", base_name])
    ctx.stream_command(args, cwd=repo_root)

    enriched_path = intermediate_dir / "_inference_input" / f"{base_name}.enriched.json"
    if not enriched_path.exists():
        raise FileNotFoundError(f"Enrichment step did not produce {enriched_path.name}")

    ctx.update(message="Segmenting", progress=0.8)
    output_path = _ensure_parent(Path(output_dir) / f"{base_name}.srt")
    ctx.stream_command(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--input",
            str(enriched_path),
            "--output",
            str(output_path),
            "--config",
            str(model_config_path),
        ],
        cwd=repo_root,
    )

    artifacts_dir = ctx.record.workspace / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    final_copy = _copy_into(output_path, artifacts_dir)

    ctx.update(message="Inference complete", progress=1.0)
    _record_result(
        ctx,
        {
            "output_srt": str(output_path),
            "workspace_artifact": str(final_copy),
            "enriched_tokens": str(enriched_path),
            "asr_reference": str(asr_reference),
        },
    )


def run_training_pair(ctx: JobContext) -> None:
    params = ctx.record.params
    media_path = Path(params["media_path"]).expanduser()
    srt_path = Path(params["srt_path"]).expanduser()
    overrides = params.get("config_overrides") or {}

    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    ctx.update(message="Preparing workspace", progress=0.05)
    runtime = ctx.effective_config(overrides)
    config_path = Path(runtime["__path__"])
    pipeline_root = Path(runtime.get("pipeline_root", ctx.record.workspace / "pipeline"))
    intermediate_dir = Path(runtime.get("intermediate_dir", pipeline_root / "_intermediate"))

    repo_root = Path(runtime.get("project_root", REPO_ROOT))

    inputs_dir = ctx.record.workspace / "inputs"
    media_copy = _copy_into(media_path, inputs_dir / "media")
    srt_copy = _copy_into(srt_path, inputs_dir / "captions")

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    base_name = media_copy.stem

    ctx.update(message="Generating ASR", progress=0.25)
    ctx.stream_command(
        [
            sys.executable,
            str(repo_root / "align_make.py"),
            "--input-file",
            str(media_copy),
            "--out-root",
            str(intermediate_dir),
            "--config-file",
            str(config_path),
        ],
        cwd=repo_root,
    )

    asr_reference = intermediate_dir / "_align" / f"{base_name}.asr.visual.words.diar.json"
    if not asr_reference.exists():
        raise FileNotFoundError(f"Expected ASR output was not created: {asr_reference}")

    ctx.update(message="Building training pair", progress=0.65)
    training_dir = intermediate_dir / "_training"
    training_dir.mkdir(parents=True, exist_ok=True)
    ctx.stream_command(
        [
            sys.executable,
            str(repo_root / "build_training_pair_standalone.py"),
            "--primary-input",
            str(srt_copy),
            "--asr-reference",
            str(asr_reference),
            "--out-training-dir",
            str(training_dir),
            "--config-file",
            str(config_path),
        ],
        cwd=repo_root,
    )

    expected_json = training_dir / f"{srt_copy.stem}.train.words.json"
    if not expected_json.exists():
        raise FileNotFoundError(f"Training data was not produced: {expected_json}")

    artifacts_dir = ctx.record.workspace / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    final_copy = _copy_into(expected_json, artifacts_dir)

    ctx.update(message="Training pair ready", progress=1.0)
    _record_result(
        ctx,
        {
            "training_json": str(expected_json),
            "workspace_artifact": str(final_copy),
            "asr_reference": str(asr_reference),
        },
    )


def run_model_training(ctx: JobContext) -> None:
    params = ctx.record.params
    corpus_dir = Path(params["corpus_dir"]).expanduser()
    overrides = params.get("config_overrides") or {}
    iterations = params.get("iterations")
    error_boost = params.get("error_boost_factor")

    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    ctx.update(message="Preparing workspace", progress=0.05)
    runtime = ctx.effective_config(overrides)
    config_path = Path(runtime["__path__"])
    repo_root = Path(runtime.get("project_root", REPO_ROOT))

    artifacts_dir = ctx.record.workspace / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = _copy_into(corpus_dir, ctx.record.workspace / "corpus") if corpus_dir.is_file() else corpus_dir

    constraints_path = _ensure_parent(artifacts_dir / "constraints.json")
    weights_path = _ensure_parent(artifacts_dir / "model_weights.json")

    args = [
        sys.executable,
        str(repo_root / "scripts" / "train_model.py"),
        "--corpus",
        str(corpus_path if corpus_path.is_dir() else corpus_path.parent),
        "--constraints",
        str(constraints_path),
        "--weights",
        str(weights_path),
        "--config",
        str(config_path),
    ]
    if iterations:
        args.extend(["--iterations", str(iterations)])
    if error_boost:
        args.extend(["--error-boost-factor", str(error_boost)])

    ctx.update(message="Training model", progress=0.3)
    ctx.stream_command(args, cwd=repo_root)

    ctx.update(message="Model training complete", progress=1.0)
    _record_result(
        ctx,
        {
            "constraints": str(constraints_path),
            "weights": str(weights_path),
        },
    )

