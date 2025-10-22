"""FastAPI application powering the ISCE control panel."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config_service import (
    CORE_CONFIG_PATH,
    PIPELINE_CONFIG_PATH,
    ensure_directories,
    ensure_path,
    load_core_config,
    load_effective_pipeline_config,
    load_pipeline_overrides,
    runtime_defaults,
    write_core_config,
    write_pipeline_overrides,
)
from .job_manager import Job, job_manager
from .schemas import (
    ConfigEnvelope,
    ConfigUpdate,
    ConfigWriteResult,
    InferenceRequest,
    JobListResponse,
    JobLogsResponse,
    JobResponse,
    TrainModelRequest,
    TrainingPairRequest,
    job_to_schema,
)

from run_pipeline import process_inference_file, process_training_file


DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _load_allowed_origins() -> list[str]:
    raw = os.getenv("ISCE_UI_ALLOWED_ORIGINS")
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or DEFAULT_ALLOWED_ORIGINS


allowed_origins = _load_allowed_origins()
allow_credentials = os.getenv("ISCE_UI_ALLOW_CREDENTIALS", "true").lower() == "true"

if allow_credentials and any(origin == "*" for origin in allowed_origins):
    allow_credentials = False


app = FastAPI(title="ISCE Captioning Control Panel", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    return resolved


def _prepare_inference_job(
    media_path: Path,
    transcript_path: Optional[Path],
    overrides: Dict[str, str],
) -> Dict[str, Any]:
    cfg = load_effective_pipeline_config(overrides)
    ensure_directories(cfg)
    intermediate_dir = Path(cfg["intermediate_dir"])
    for subdir in ["_align", "_inference_input", "_training"]:
        (intermediate_dir / subdir).mkdir(parents=True, exist_ok=True)

    if transcript_path is not None:
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
        target_dir = Path(cfg["txt_placement_folder"])
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / f"{media_path.stem}.txt"
        print(f"Copying transcript to {destination}")
        shutil.copyfile(transcript_path, destination)

    print("Running inference pipeline...")
    process_inference_file(media_path, cfg)

    output_path = Path(cfg["output_dir"]) / f"{media_path.stem}.srt"
    return {"output": str(output_path), "config": cfg}


def _prepare_training_pair_job(
    media_path: Path,
    srt_path: Path,
    overrides: Dict[str, str],
) -> Dict[str, Any]:
    cfg = load_effective_pipeline_config(overrides)
    ensure_directories(cfg)
    intermediate_dir = Path(cfg["intermediate_dir"])
    for subdir in ["_align", "_training"]:
        (intermediate_dir / subdir).mkdir(parents=True, exist_ok=True)

    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    print("Running training pair pipeline...")
    process_training_file(media_path, srt_path, cfg)

    training_dir = Path(cfg["intermediate_dir"]) / "_training"
    output_path = training_dir / f"{media_path.stem}.train.words.json"
    return {"output": str(output_path), "config": cfg}


def _run_training_script(request: TrainModelRequest) -> Dict[str, str]:
    corpus_dir = _resolve_path(request.corpus_dir)
    constraints_path = ensure_path(_resolve_path(request.constraints_path) or request.constraints_path)
    weights_path = ensure_path(_resolve_path(request.weights_path) or request.weights_path)
    config_path = _resolve_path(request.config_path) or Path(request.config_path)

    if corpus_dir is None or not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {request.corpus_dir}")

    argv = [
        "--corpus",
        str(corpus_dir),
        "--constraints",
        str(constraints_path),
        "--weights",
        str(weights_path),
        "--config",
        str(config_path),
        "--iterations",
        str(request.iterations),
        "--error-boost-factor",
        str(request.error_boost_factor),
    ]

    cmd = [sys.executable, "-m", "scripts.train_model", *argv]

    print("Launching model training script with command:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    return {
        "constraints": str(constraints_path),
        "weights": str(weights_path),
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/jobs", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    jobs = [job_to_schema(job) for job in job_manager.list_jobs()]
    return JobListResponse(jobs=jobs)


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        job = job_manager.get_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_schema(job)


@app.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
def get_job_logs(job_id: str, start: int = Query(0, ge=0)) -> JobLogsResponse:
    try:
        payload = job_manager.get_logs(job_id, start=start)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobLogsResponse(**payload)


@app.post("/jobs/inference", response_model=JobResponse)
def launch_inference(request: InferenceRequest) -> JobResponse:
    media_path = _resolve_path(request.media_path)
    transcript_path = _resolve_path(request.transcript_path) if request.transcript_path else None

    if media_path is None or not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    overrides = request.pipeline_overrides.dict(exclude_none=True) if request.pipeline_overrides else {}

    metadata = {
        "media": str(media_path),
        "transcript": str(transcript_path) if transcript_path else None,
    }

    def runner(job: Job) -> None:
        result = _prepare_inference_job(media_path, transcript_path, overrides)
        job_manager.update_metadata(job, {"output": result["output"], "config_snapshot": result.get("config")})

    job = job_manager.create_job("inference", runner, metadata=metadata)
    return job_to_schema(job)


@app.post("/jobs/training-pair", response_model=JobResponse)
def launch_training_pair(request: TrainingPairRequest) -> JobResponse:
    media_path = _resolve_path(request.media_path)
    srt_path = _resolve_path(request.srt_path)

    if media_path is None or not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    if srt_path is None or not srt_path.exists():
        raise HTTPException(status_code=404, detail="SRT file not found")

    overrides = request.pipeline_overrides.dict(exclude_none=True) if request.pipeline_overrides else {}
    metadata = {
        "media": str(media_path),
        "srt": str(srt_path),
    }

    def runner(job: Job) -> None:
        result = _prepare_training_pair_job(media_path, srt_path, overrides)
        job_manager.update_metadata(job, {"output": result["output"], "config_snapshot": result.get("config")})

    job = job_manager.create_job("training_pair", runner, metadata=metadata)
    return job_to_schema(job)


@app.post("/jobs/model-training", response_model=JobResponse)
def launch_model_training(request: TrainModelRequest) -> JobResponse:
    overrides_metadata = request.dict()

    def runner(job: Job) -> None:
        result = _run_training_script(request)
        job_manager.update_metadata(job, result)

    job = job_manager.create_job("model_training", runner, metadata=overrides_metadata)
    return job_to_schema(job)


@app.get("/config/pipeline", response_model=ConfigEnvelope)
def get_pipeline_config() -> ConfigEnvelope:
    return ConfigEnvelope(
        defaults=runtime_defaults(),
        overrides=load_pipeline_overrides(),
        resolved=load_effective_pipeline_config(),
    )


@app.put("/config/pipeline", response_model=ConfigWriteResult)
def update_pipeline_config(update: ConfigUpdate) -> ConfigWriteResult:
    write_pipeline_overrides(update.content)
    return ConfigWriteResult(path=str(PIPELINE_CONFIG_PATH), updated=True)


@app.get("/config/core", response_model=ConfigEnvelope)
def get_core_config() -> ConfigEnvelope:
    content = load_core_config()
    return ConfigEnvelope(defaults={}, overrides=content, resolved=content)


@app.put("/config/core", response_model=ConfigWriteResult)
def update_core_config(update: ConfigUpdate) -> ConfigWriteResult:
    write_core_config(update.content)
    return ConfigWriteResult(path=str(CORE_CONFIG_PATH), updated=True)


if os.getenv("ISCE_UI_DEV_SERVER") == "1":
    import uvicorn

    if __name__ == "__main__":
        uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=True)
