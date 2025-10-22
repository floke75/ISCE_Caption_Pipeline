from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config_store import ConfigStore
from .job_manager import JobManager
from .schemas import InferenceRequest, ModelTrainingRequest, TrainingPairsRequest
from .tasks import PipelineTasks

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

pipeline_store = ConfigStore(REPO_ROOT / "pipeline_config.yaml")
model_store = ConfigStore(REPO_ROOT / "config.yaml")
job_manager = JobManager(REPO_ROOT / "ui_data" / "jobs")
tasks = PipelineTasks(job_manager, pipeline_store, model_store)

app = FastAPI(title="ISCE Pipeline Control Center", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    status: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    progress: float
    stage: Optional[str]
    message: Optional[str]
    artifacts: list[Dict[str, Any]]
    params: Dict[str, Any]
    result: Dict[str, Any]


class LogChunk(BaseModel):
    content: str
    offset: int
    complete: bool


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: Optional[Dict[str, Any]] = None
    yaml: Optional[str] = None

    @model_validator(mode="after")
    def ensure_payload(cls, values: "ConfigUpdate") -> "ConfigUpdate":  # type: ignore[override]
        if not values.config and not values.yaml:
            raise ValueError("Either 'config' or 'yaml' must be supplied.")
        return values


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    records = sorted(job_manager.list_jobs(), key=lambda rec: rec.created_at, reverse=True)
    return [JobResponse(**record.to_dict()) for record in records]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    try:
        record = job_manager.get(job_id)
    except KeyError as exc:  # noqa: PERF203
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return JobResponse(**record.to_dict())


@app.get("/api/jobs/{job_id}/log", response_model=LogChunk)
def get_job_log(job_id: str, offset: int = Query(0, ge=0), limit: Optional[int] = Query(None, ge=1, le=1_000_000)) -> LogChunk:
    try:
        chunk = job_manager.read_log(job_id, offset=offset, limit=limit)
    except KeyError as exc:  # noqa: PERF203
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return LogChunk(**chunk)


@app.post("/api/jobs/inference", response_model=JobResponse, status_code=201)
def create_inference_job(payload: InferenceRequest) -> JobResponse:
    record = tasks.launch_inference(payload)
    return JobResponse(**record.to_dict())


@app.post("/api/jobs/training-pairs", response_model=JobResponse, status_code=201)
def create_training_pairs_job(payload: TrainingPairsRequest) -> JobResponse:
    record = tasks.launch_training_pairs(payload)
    return JobResponse(**record.to_dict())


@app.post("/api/jobs/model-training", response_model=JobResponse, status_code=201)
def create_model_training_job(payload: ModelTrainingRequest) -> JobResponse:
    record = tasks.launch_model_training(payload)
    return JobResponse(**record.to_dict())


@app.get("/api/config/pipeline")
def read_pipeline_config() -> Dict[str, Any]:
    return {"config": pipeline_store.read(), "yaml": pipeline_store.dump_yaml()}


@app.put("/api/config/pipeline")
def update_pipeline_config(update: ConfigUpdate) -> Dict[str, Any]:
    if update.yaml is not None:
        parsed = yaml.safe_load(update.yaml) or {}
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="Parsed YAML must be a mapping")
        pipeline_store.write(parsed)
    elif update.config is not None:
        pipeline_store.update(update.config)
    return {"config": pipeline_store.read(), "yaml": pipeline_store.dump_yaml()}


@app.get("/api/config/model")
def read_model_config() -> Dict[str, Any]:
    return {"config": model_store.read(), "yaml": model_store.dump_yaml()}


@app.put("/api/config/model")
def update_model_config(update: ConfigUpdate) -> Dict[str, Any]:
    if update.yaml is not None:
        parsed = yaml.safe_load(update.yaml) or {}
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="Parsed YAML must be a mapping")
        model_store.write(parsed)
    elif update.config is not None:
        model_store.update(update.config)
    return {"config": model_store.read(), "yaml": model_store.dump_yaml()}


@app.exception_handler(ValueError)
def handle_validation_error(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=False)
