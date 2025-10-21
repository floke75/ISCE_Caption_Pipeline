"""Utilities that execute pipeline tasks for the web UI."""
from __future__ import annotations

import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from pipeline_config import load_pipeline_config
from run_pipeline import (
    DEFAULT_SETTINGS as ORCHESTRATOR_DEFAULTS,
    process_inference_file,
    process_training_file,
    setup_directories,
)

from scripts import train_model

from ui.jobs import JobStatus, JobStore


_train_model_lock = Lock()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _initial_runtime_config(job_id: str, overrides: Optional[Dict[str, Any]] = None, pipeline_config_path: Optional[Path] = None) -> Dict[str, Any]:
    repo_root = _repo_root()
    config_path = pipeline_config_path or (repo_root / "pipeline_config.yaml")

    config = load_pipeline_config(ORCHESTRATOR_DEFAULTS, str(config_path))

    runtime_root = repo_root / "ui_data" / "jobs" / job_id
    intermediate_dir = runtime_root / "intermediate"
    output_dir = runtime_root / "output"
    processed_dir = runtime_root / "processed"
    txt_dir = runtime_root / "txt_inputs"
    drop_inf = runtime_root / "drop_inference"
    drop_train = runtime_root / "drop_training"
    srt_dir = runtime_root / "srt_inputs"

    runtime_settings = {
        "project_root": str(repo_root),
        "pipeline_root": str(runtime_root),
        "intermediate_dir": str(intermediate_dir),
        "output_dir": str(output_dir),
        "processed_dir": str(processed_dir),
        "txt_placement_folder": str(txt_dir),
        "srt_placement_folder": str(srt_dir),
        "drop_folder_inference": str(drop_inf),
        "drop_folder_training": str(drop_train),
    }

    config = _deep_update(config, runtime_settings)

    if overrides:
        config = _deep_update(config, overrides)

    setup_directories(config)
    return config


def _copy_transcript(media_path: Path, transcript_path: Path, target_dir: Path) -> Path:
    base_name = media_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"{base_name}.txt"
    shutil.copyfile(transcript_path, destination)
    return destination


def run_inference_job(
    store: JobStore,
    job_id: str,
    media_path: Path,
    transcript_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    pipeline_config_path: Optional[Path] = None,
) -> None:
    store.set_status(job_id, JobStatus.RUNNING)

    try:
        media_path = media_path.expanduser().resolve()
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        transcript_copy: Optional[Path] = None
        if transcript_path:
            transcript_path = transcript_path.expanduser().resolve()
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
            if transcript_path.suffix.lower() != ".txt":
                raise ValueError("Transcript file must be a .txt file for inference workflows.")

        config = _initial_runtime_config(
            job_id,
            overrides=config_overrides,
            pipeline_config_path=pipeline_config_path,
        )
        txt_dir = Path(config["txt_placement_folder"])

        if transcript_path:
            transcript_copy = _copy_transcript(media_path, transcript_path, txt_dir)

        log_stream = store.log_stream(job_id)

        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            process_inference_file(media_path, config)

        base_name = media_path.stem
        intermediate_dir = Path(config["intermediate_dir"])
        output_dir = Path(config["output_dir"])

        enriched = intermediate_dir / "_inference_input" / f"{base_name}.enriched.json"
        final_srt = output_dir / f"{base_name}.srt"

        result: Dict[str, Any] = {
            "enriched_json": str(enriched),
            "srt_path": str(final_srt),
            "job_artifacts": str(output_dir.parent),
        }

        if output_path:
            output_path = output_path.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(final_srt, output_path)
            result["copied_to"] = str(output_path)

        if transcript_copy and transcript_copy.exists():
            result["transcript_copy"] = str(transcript_copy)

        store.set_result(job_id, result)
        store.set_status(job_id, JobStatus.SUCCEEDED)
    except Exception as exc:
        store.set_status(job_id, JobStatus.FAILED, error=str(exc))
        raise


def run_training_pair_job(
    store: JobStore,
    job_id: str,
    media_path: Path,
    srt_path: Path,
    config_overrides: Optional[Dict[str, Any]] = None,
    pipeline_config_path: Optional[Path] = None,
) -> None:
    store.set_status(job_id, JobStatus.RUNNING)

    try:
        media_path = media_path.expanduser().resolve()
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        srt_path = srt_path.expanduser().resolve()
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        config = _initial_runtime_config(
            job_id,
            overrides=config_overrides,
            pipeline_config_path=pipeline_config_path,
        )

        log_stream = store.log_stream(job_id)

        with redirect_stdout(log_stream), redirect_stderr(log_stream):
            process_training_file(media_path, srt_path, config)

        base_name = media_path.stem
        intermediate_dir = Path(config["intermediate_dir"])
        training_json = intermediate_dir / "_training" / f"{base_name}.train.words.json"

        store.set_result(
            job_id,
            {
                "training_json": str(training_json),
                "job_artifacts": str(intermediate_dir.parent),
            },
        )
        store.set_status(job_id, JobStatus.SUCCEEDED)
    except Exception as exc:
        store.set_status(job_id, JobStatus.FAILED, error=str(exc))
        raise


def run_model_training_job(
    store: JobStore,
    job_id: str,
    corpus_dir: Path,
    iterations: int = 3,
    error_boost_factor: float = 1.0,
    constraints_output: Optional[Path] = None,
    weights_output: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> None:
    store.set_status(job_id, JobStatus.RUNNING)

    try:
        corpus_dir = corpus_dir.expanduser().resolve()
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        repo_root = _repo_root()
        default_config = repo_root / "config.yaml"
        config_path = (config_path or default_config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        job_dir = repo_root / "ui_data" / "jobs" / job_id / "training"
        job_dir.mkdir(parents=True, exist_ok=True)

        constraints_output = (constraints_output or (job_dir / "constraints.json")).expanduser().resolve()
        weights_output = (weights_output or (job_dir / "model_weights.json")).expanduser().resolve()
        constraints_output.parent.mkdir(parents=True, exist_ok=True)
        weights_output.parent.mkdir(parents=True, exist_ok=True)

        log_stream = store.log_stream(job_id)

        with _train_model_lock:
            argv_backup = list(sys.argv)
            sys.argv = [
                "train_model",
                "--corpus",
                str(corpus_dir),
                "--constraints",
                str(constraints_output),
                "--weights",
                str(weights_output),
                "--config",
                str(config_path),
                "--iterations",
                str(iterations),
                "--error-boost-factor",
                str(error_boost_factor),
            ]

            try:
                with redirect_stdout(log_stream), redirect_stderr(log_stream):
                    try:
                        train_model.main()
                    except SystemExit as exc:
                        if exc.code not in (0, None):
                            raise RuntimeError(
                                f"Training script exited with code {exc.code}"
                            ) from exc
            finally:
                sys.argv = argv_backup
    except Exception as exc:
        store.set_status(job_id, JobStatus.FAILED, error=str(exc))
        raise

    artifacts = {
        "constraints": str(constraints_output),
        "weights": str(weights_output),
        "config": str(config_path),
    }
    store.set_result(job_id, artifacts)
    store.set_status(job_id, JobStatus.SUCCEEDED)
