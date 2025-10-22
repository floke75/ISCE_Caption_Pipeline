from __future__ import annotations

import os
import select
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml

from run_pipeline import setup_directories
from ui_server.config_service import PipelineConfigService, ModelConfigService, _deep_merge
from ui_server.job_manager import JobContext, JobCancelledError
from ui_server.path_validation import require_path
from ui_server.schemas import (
    InferenceJobRequest,
    ModelTrainingJobRequest,
    TrainingPairJobRequest,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


class CommandError(RuntimeError):
    pass


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Attempt to terminate the process gracefully, then force kill if needed."""

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _stream_command(
    command: Iterable[Any],
    cwd: Path,
    log_file: Path,
    cancel_event: threading.Event | None = None,
) -> None:
    cmd = [str(item) for item in command]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        try:
            stdout = process.stdout
            assert stdout is not None
            while True:
                if cancel_event and cancel_event.is_set():
                    if process.poll() is None:
                        _terminate_process(process)
                        raise JobCancelledError("Command cancelled")
                    # The process has already exited; ignore late cancellation.

                ready, _, _ = select.select([stdout], [], [], 0.1)
                if ready:
                    line = stdout.readline()
                    if line:
                        handle.write(line)
                        handle.flush()
                        continue

                return_code = process.poll()
                if return_code is not None:
                    # Process has finished; drain any remaining output.
                    remaining = stdout.read()
                    if remaining:
                        handle.write(remaining)
                        handle.flush()
                    break

            process.wait()
            handle.flush()
            if process.returncode != 0:
                raise CommandError(
                    f"Command {' '.join(cmd)} failed with exit code {process.returncode}"
                )
        finally:
            if process.stdout:
                process.stdout.close()

def _resolve_placeholders(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve template placeholders in the config using top-level context."""

    context = {k: v for k, v in config.items() if isinstance(v, str)}

    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        if isinstance(value, str) and "{" in value and "}" in value:
            try:
                return value.format(**context)
            except KeyError:
                return value
        return value

    return _resolve(config)


def _prepare_pipeline_config(
    pipeline_service: PipelineConfigService,
    workspace: Path,
    overrides: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Path]:
    overrides = overrides or {}
    base_config = pipeline_service.load(resolved=False)
    merged = _deep_merge(base_config, overrides)
    merged["project_root"] = str(REPO_ROOT)
    pipeline_root = workspace / "pipeline"
    dirs = {
        "pipeline_root": pipeline_root,
        "drop_folder_inference": pipeline_root / "1_DROP_FOLDER_INFERENCE",
        "drop_folder_training": pipeline_root / "2_DROP_FOLDER_TRAINING",
        "srt_placement_folder": pipeline_root / "3_MANUAL_SRT_PLACEMENT",
        "txt_placement_folder": pipeline_root / "4_MANUAL_TXT_PLACEMENT",
        "processed_dir": pipeline_root / "_processed",
        "intermediate_dir": pipeline_root / "_intermediate",
        "output_dir": pipeline_root / "_output",
    }
    for key, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        merged[key] = str(path)
    # Sub directories expected by orchestrator helpers
    (dirs["processed_dir"] / "inference").mkdir(parents=True, exist_ok=True)
    (dirs["processed_dir"] / "training").mkdir(parents=True, exist_ok=True)
    (dirs["processed_dir"] / "srt").mkdir(parents=True, exist_ok=True)
    (dirs["intermediate_dir"] / "_align").mkdir(parents=True, exist_ok=True)
    (dirs["intermediate_dir"] / "_inference_input").mkdir(parents=True, exist_ok=True)
    (dirs["intermediate_dir"] / "_training").mkdir(parents=True, exist_ok=True)

    merged = _resolve_placeholders(merged)
    setup_directories(merged)

    pipeline_config_path = workspace / "pipeline_config.yaml"
    pipeline_config_path.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return merged, pipeline_config_path


def _prepare_model_config(
    model_service: ModelConfigService,
    workspace: Path,
    overrides: Dict[str, Any] | None = None,
) -> Path:
    overrides = overrides or {}
    base_model = model_service.load()
    merged = _deep_merge(base_model, overrides)
    model_config_path = workspace / "config.yaml"
    model_config_path.write_text(
        yaml.safe_dump(merged, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return model_config_path


def run_inference_job(
    context: JobContext,
    request: InferenceJobRequest,
    pipeline_service: PipelineConfigService,
    model_service: ModelConfigService,
) -> Dict[str, Any]:
    workspace = context.workspace()
    log_file = context.log_path()
    context.update(stage="preparing", progress=0.02)

    pipeline_config, pipeline_config_path = _prepare_pipeline_config(
        pipeline_service, workspace, request.pipeline_overrides
    )
    model_config_path = _prepare_model_config(model_service, workspace, request.model_overrides)
    context.raise_if_cancelled()

    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    media_src = require_path(request.media_path, kind="file", purpose="Media file")
    media_dst = inputs_dir / media_src.name
    if media_src != media_dst:
        shutil.copy2(media_src, media_dst)

    transcript_dst = None
    if request.transcript_path:
        transcript_src = require_path(
            request.transcript_path,
            kind="file",
            purpose="Transcript file",
        )
        transcript_name = media_dst.stem + transcript_src.suffix
        transcript_dst = Path(pipeline_config["txt_placement_folder"]) / transcript_name
        shutil.copy2(transcript_src, transcript_dst)
        context.raise_if_cancelled()

    stages = [
        (
            "Aligning audio",
            [
                "python",
                REPO_ROOT / "align_make.py",
                "--input-file",
                media_dst,
                "--out-root",
                pipeline_config["intermediate_dir"],
                "--config-file",
                pipeline_config_path,
            ],
        ),
        (
            "Enriching transcript",
            _build_inference_command(media_dst, pipeline_config, pipeline_config_path, transcript_dst),
        ),
        (
            "Segmenting captions",
            [
                "python",
                REPO_ROOT / "main.py",
                "--input",
                Path(pipeline_config["intermediate_dir"]) / "_inference_input" / f"{media_dst.stem}.enriched.json",
                "--output",
                Path(pipeline_config["output_dir"]) / f"{media_dst.stem}.srt",
                "--config",
                model_config_path,
            ],
        ),
    ]

    for index, (stage, command) in enumerate(stages, start=1):
        context.update(stage=stage, progress=(index - 1) / len(stages))
        context.raise_if_cancelled()
        _stream_command(command, REPO_ROOT, log_file, context.cancel_event)
        context.update(progress=index / len(stages))

    final_srt = Path(pipeline_config["output_dir"]) / f"{media_dst.stem}.srt"
    delivered_to = None
    if request.output_directory:
        target_dir = require_path(
            request.output_directory,
            kind="directory",
            must_exist=False,
            allow_create=True,
            purpose="Output directory",
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        delivered_to = target_dir / final_srt.name
        shutil.copy2(final_srt, delivered_to)

    result = {
        "srt_path": str(final_srt),
        "workspace": str(workspace),
        "intermediate_dir": pipeline_config["intermediate_dir"],
    }
    if transcript_dst:
        result["transcript_used"] = str(transcript_dst)
    if delivered_to:
        result["delivered_to"] = str(delivered_to)
    return result


def _build_inference_command(
    media_dst: Path,
    pipeline_config: Dict[str, Any],
    pipeline_config_path: Path,
    transcript_dst: Path | None,
) -> Iterable[Any]:
    asr_reference = Path(pipeline_config["intermediate_dir"]) / "_align" / f"{media_dst.stem}.asr.visual.words.diar.json"
    inference_dir = Path(pipeline_config["intermediate_dir"]) / "_inference_input"
    command: list[Any] = [
        "python",
        REPO_ROOT / "build_training_pair_standalone.py",
        "--primary-input",
        transcript_dst if transcript_dst else asr_reference,
        "--asr-reference",
        asr_reference,
        "--out-inference-dir",
        inference_dir,
        "--config-file",
        pipeline_config_path,
    ]
    if not transcript_dst:
        command.extend(["--asr-only-mode", "--output-basename", media_dst.stem])
    return command


def run_training_pair_job(
    context: JobContext,
    request: TrainingPairJobRequest,
    pipeline_service: PipelineConfigService,
) -> Dict[str, Any]:
    workspace = context.workspace()
    log_file = context.log_path()
    context.update(stage="preparing", progress=0.02)

    pipeline_config, pipeline_config_path = _prepare_pipeline_config(
        pipeline_service, workspace, request.pipeline_overrides
    )
    context.raise_if_cancelled()

    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    media_src = require_path(request.media_path, kind="file", purpose="Media file")
    srt_src = require_path(request.srt_path, kind="file", purpose="SRT file")

    media_dst = inputs_dir / media_src.name
    srt_dst = Path(pipeline_config["srt_placement_folder"]) / srt_src.name
    shutil.copy2(media_src, media_dst)
    shutil.copy2(srt_src, srt_dst)
    context.raise_if_cancelled()

    stages = [
        (
            "Aligning audio",
            [
                "python",
                REPO_ROOT / "align_make.py",
                "--input-file",
                media_dst,
                "--out-root",
                pipeline_config["intermediate_dir"],
                "--config-file",
                pipeline_config_path,
            ],
        ),
        (
            "Creating training pair",
            [
                "python",
                REPO_ROOT / "build_training_pair_standalone.py",
                "--primary-input",
                srt_dst,
                "--asr-reference",
                Path(pipeline_config["intermediate_dir"]) / "_align" / f"{media_dst.stem}.asr.visual.words.diar.json",
                "--out-training-dir",
                Path(pipeline_config["intermediate_dir"]) / "_training",
                "--config-file",
                pipeline_config_path,
            ],
        ),
    ]

    for index, (stage, command) in enumerate(stages, start=1):
        context.update(stage=stage, progress=(index - 1) / len(stages))
        context.raise_if_cancelled()
        _stream_command(command, REPO_ROOT, log_file, context.cancel_event)
        context.update(progress=index / len(stages))

    training_file = Path(pipeline_config["intermediate_dir"]) / "_training" / f"{srt_dst.stem}.train.words.json"
    delivered_to = None
    if request.output_directory:
        target_dir = require_path(
            request.output_directory,
            kind="directory",
            must_exist=False,
            allow_create=True,
            purpose="Output directory",
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        delivered_to = target_dir / training_file.name
        shutil.copy2(training_file, delivered_to)

    result = {
        "training_file": str(training_file),
        "workspace": str(workspace),
    }
    if delivered_to:
        result["delivered_to"] = str(delivered_to)
    return result


def run_model_training_job(
    context: JobContext,
    request: ModelTrainingJobRequest,
    model_service: ModelConfigService,
) -> Dict[str, Any]:
    workspace = context.workspace()
    log_file = context.log_path()
    context.update(stage="preparing", progress=0.05)

    model_config_path = _prepare_model_config(model_service, workspace, request.model_overrides)
    context.raise_if_cancelled()

    corpus_dir = require_path(
        request.corpus_dir,
        kind="directory",
        purpose="Corpus directory",
    )

    constraints_path = (
        require_path(
            request.constraints_path,
            kind="file",
            must_exist=False,
            allow_create=True,
            purpose="Constraints output",
        )
        if request.constraints_path
        else workspace / "models" / "constraints.json"
    )
    weights_path = (
        require_path(
            request.weights_path,
            kind="file",
            must_exist=False,
            allow_create=True,
            purpose="Weights output",
        )
        if request.weights_path
        else workspace / "models" / "model_weights.json"
    )
    constraints_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    context.raise_if_cancelled()

    command = [
        "python",
        REPO_ROOT / "scripts" / "train_model.py",
        "--corpus",
        corpus_dir,
        "--constraints",
        constraints_path,
        "--weights",
        weights_path,
        "--config",
        model_config_path,
        "--iterations",
        request.iterations,
        "--error-boost-factor",
        request.error_boost_factor,
    ]
    context.update(stage="Training model", progress=0.1)
    context.raise_if_cancelled()
    _stream_command(command, REPO_ROOT, log_file, context.cancel_event)
    context.update(progress=1.0)

    return {
        "constraints_path": str(constraints_path),
        "weights_path": str(weights_path),
        "workspace": str(workspace),
    }
