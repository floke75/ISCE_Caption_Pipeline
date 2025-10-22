"""FastAPI application providing the UI backend."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config_service import ConfigField, ConfigService
from .job_manager import JobManager, JobQueueFull
from . import pipelines

REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = REPO_ROOT / "ui_data"

max_workers = int(os.getenv("PIPELINE_MAX_CONCURRENT_JOBS", "3"))
queue_limit_env: Optional[str] = os.getenv("PIPELINE_MAX_QUEUED_JOBS")
queue_limit = int(queue_limit_env) if queue_limit_env else None

if max_workers < 1:
    raise ValueError("PIPELINE_MAX_CONCURRENT_JOBS must be at least 1")

config_service = ConfigService(REPO_ROOT, STORAGE_ROOT)
job_manager = JobManager(STORAGE_ROOT, config_service, max_workers=max_workers, queue_limit=queue_limit)

app = FastAPI(title="ISCE Pipeline UI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigFieldModel(BaseModel):
    path: List[str]
    label: str
    field_type: str = Field(alias="fieldType")
    section: str
    description: Optional[str] = None
    options: Optional[List[Any]] = None
    advanced: bool = False

    @classmethod
    def from_field(cls, field: ConfigField) -> "ConfigFieldModel":
        return cls(
            path=field.path,
            label=field.label,
            fieldType=field.field_type,
            section=field.section,
            description=field.description,
            options=field.options,
            advanced=field.advanced,
        )

    class Config:
        allow_population_by_field_name = True


class ConfigSnapshot(BaseModel):
    effective: Dict[str, Any]
    overrides: Dict[str, Any]
    fields: List[ConfigFieldModel]


class ConfigPatch(BaseModel):
    updates: Dict[str, Any]


class ConfigReplace(BaseModel):
    overrides: Dict[str, Any]


class ConfigYamlUpdate(BaseModel):
    yaml: str


class JobModel(BaseModel):
    id: str
    job_type: str = Field(alias="jobType")
    status: str
    progress: float
    message: str
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class InferenceJobRequest(BaseModel):
    media_path: str
    transcript_path: Optional[str] = None
    output_dir: Optional[str] = None
    model_config_path: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class TrainingPairRequest(BaseModel):
    media_path: str
    srt_path: str
    config_overrides: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ModelTrainingRequest(BaseModel):
    corpus_dir: str
    iterations: Optional[int] = None
    error_boost_factor: Optional[float] = None
    config_overrides: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


@app.get("/api/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config/pipeline", response_model=ConfigSnapshot)
def get_pipeline_config() -> ConfigSnapshot:
    effective = config_service.effective_config()
    overrides = config_service.stored_overrides()
    fields = [ConfigFieldModel.from_field(f) for f in config_service.describe_fields()]
    return ConfigSnapshot(effective=effective, overrides=overrides, fields=fields)


@app.put("/api/config/pipeline", response_model=ConfigSnapshot)
def update_pipeline_config(payload: ConfigPatch) -> ConfigSnapshot:
    patch = config_service.build_patch(payload.updates)
    config_service.apply_patch(patch)
    return get_pipeline_config()


@app.put("/api/config/pipeline/replace", response_model=ConfigSnapshot)
def replace_pipeline_config(payload: ConfigReplace) -> ConfigSnapshot:
    config_service.save_overrides(payload.overrides)
    return get_pipeline_config()


@app.get("/api/config/pipeline/raw")
def get_pipeline_config_yaml() -> Dict[str, str]:
    import yaml

    overrides = config_service.stored_overrides()
    return {"yaml": yaml.safe_dump(overrides, allow_unicode=True, sort_keys=False)}


@app.put("/api/config/pipeline/raw", response_model=ConfigSnapshot)
def update_pipeline_config_yaml(payload: ConfigYamlUpdate) -> ConfigSnapshot:
    import yaml

    try:
        overrides = yaml.safe_load(payload.yaml) if payload.yaml.strip() else {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=400, detail="YAML payload must represent a mapping")
    config_service.save_overrides(overrides)
    return get_pipeline_config()


def _serialize_job(record) -> JobModel:
    return JobModel(
        id=record.id,
        jobType=record.job_type,
        status=record.status,
        progress=record.progress,
        message=record.message,
        createdAt=record.created_at.isoformat(),
        updatedAt=record.updated_at.isoformat(),
        params=record.params,
        result=record.result,
        error=record.error,
    )


@app.get("/api/jobs", response_model=List[JobModel])
def list_jobs() -> List[JobModel]:
    return [_serialize_job(job) for job in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobModel)
def get_job(job_id: str) -> JobModel:
    try:
        record = job_manager.get_job(job_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return _serialize_job(record)


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str, tail: int = Query(4000, ge=100, le=20000)) -> Dict[str, Any]:
    try:
        log = job_manager.get_log_tail(job_id, limit=tail)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return {"log": log}


def _format_sse(data: str, *, event: Optional[str] = None) -> str:
    lines = data.rstrip("\n").splitlines() or [""]
    payload = []
    if event:
        payload.append(f"event: {event}")
    for line in lines:
        payload.append(f"data: {line}")
    payload.append("")
    return "\n".join(payload) + "\n"


def _log_event_stream(job_id: str, poll_interval: float = 0.5) -> Iterator[str]:
    last_size = 0
    while True:
        try:
            record = job_manager.get_job(job_id)
        except KeyError:
            yield _format_sse("Job not found", event="error")
            return
        log_path = record.log_path
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
                fh.seek(last_size)
                chunk = fh.read()
                if chunk:
                    last_size = fh.tell()
                    yield _format_sse(chunk)
                    continue
        status = record.status
        if status in {"succeeded", "failed", "cancelled"}:
            yield _format_sse(status, event="complete")
            return
        yield ": heartbeat\n\n"
        time.sleep(poll_interval)


@app.get("/api/jobs/{job_id}/log/stream")
async def stream_job_log(job_id: str) -> StreamingResponse:
    generator = _log_event_stream(job_id)
    return StreamingResponse(generator, media_type="text/event-stream")


@app.post("/api/jobs/inference", response_model=JobModel, status_code=201)
def create_inference_job(payload: InferenceJobRequest) -> JobModel:
    try:
        record = job_manager.create_job(
            "inference",
            params=payload.dict(exclude_none=True),
            runner=pipelines.run_inference,
        )
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _serialize_job(record)


@app.post("/api/jobs/training-pair", response_model=JobModel, status_code=201)
def create_training_pair_job(payload: TrainingPairRequest) -> JobModel:
    try:
        record = job_manager.create_job(
            "training_pair",
            params=payload.dict(exclude_none=True),
            runner=pipelines.run_training_pair,
        )
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _serialize_job(record)


@app.post("/api/jobs/model-training", response_model=JobModel, status_code=201)
def create_model_training_job(payload: ModelTrainingRequest) -> JobModel:
    try:
        record = job_manager.create_job(
            "model_training",
            params=payload.dict(exclude_none=True),
            runner=pipelines.run_model_training,
        )
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _serialize_job(record)


__all__ = ["app"]

