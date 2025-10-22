from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from ui_server.config_service import ModelConfigService, PipelineConfigService
from ui_server.job_manager import JobManager
from ui_server.pipeline_runner import (
    run_inference_job,
    run_model_training_job,
    run_training_pair_job,
)
from ui_server.schemas import (
    InferenceJobRequest,
    JobDetail,
    JobSummary,
    LogChunk,
    ModelTrainingJobRequest,
    TrainingPairJobRequest,
)


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    if hasattr(model, "dict"):
        return model.dict()  # type: ignore[no-any-return]
    raise TypeError("Unsupported model type")


def _infer_name(default_prefix: str, path: str | None) -> str:
    if path:
        return f"{default_prefix} {Path(path).stem}"
    return default_prefix


def _job_summary(record) -> JobSummary:
    return JobSummary(
        id=record.id,
        job_type=record.job_type,
        name=record.name,
        status=record.status,
        progress=record.progress,
        stage=record.stage,
        message=record.message,
        error=record.error,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        result=record.result,
        metrics=record.metrics,
    )


def _job_detail(record) -> JobDetail:
    return JobDetail(
        **_job_summary(record).dict(),
        parameters=record.parameters,
        workspace=str(record.workspace),
    )


pipeline_service = PipelineConfigService()
model_service = ModelConfigService()
job_manager = JobManager(Path("ui_runtime"), max_workers=2)

app = FastAPI(title="ISCE Pipeline Control Center", version="1.0.0")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/jobs", response_model=list[JobSummary])
def list_jobs() -> list[JobSummary]:
    return [_job_summary(job) for job in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str) -> JobDetail:
    record = job_manager.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_detail(record)


@app.get("/api/jobs/{job_id}/logs", response_model=LogChunk)
def get_job_logs(job_id: str, offset: int = Query(0, ge=0)) -> LogChunk:
    record = job_manager.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    chunk = job_manager.read_log(job_id, offset=offset)
    return LogChunk(**chunk)


@app.post("/api/jobs/inference", response_model=JobDetail, status_code=201)
def create_inference_job(request: InferenceJobRequest) -> JobDetail:
    payload = _model_to_dict(request)
    name = payload.get("name") or _infer_name("Inference", payload.get("media_path"))
    record = job_manager.create_job(
        "inference",
        name,
        payload,
        lambda ctx: run_inference_job(ctx, InferenceJobRequest(**payload), pipeline_service, model_service),
    )
    return _job_detail(record)


@app.post("/api/jobs/training-pair", response_model=JobDetail, status_code=201)
def create_training_pair_job(request: TrainingPairJobRequest) -> JobDetail:
    payload = _model_to_dict(request)
    name = payload.get("name") or _infer_name("Training pair", payload.get("media_path"))
    record = job_manager.create_job(
        "training_pair",
        name,
        payload,
        lambda ctx: run_training_pair_job(ctx, TrainingPairJobRequest(**payload), pipeline_service),
    )
    return _job_detail(record)


@app.post("/api/jobs/model-training", response_model=JobDetail, status_code=201)
def create_model_training_job(request: ModelTrainingJobRequest) -> JobDetail:
    payload = _model_to_dict(request)
    name = payload.get("name") or _infer_name("Model training", payload.get("corpus_dir"))
    record = job_manager.create_job(
        "model_training",
        name,
        payload,
        lambda ctx: run_model_training_job(ctx, ModelTrainingJobRequest(**payload), model_service),
    )
    return _job_detail(record)


@app.get("/api/config/pipeline", response_model=Dict[str, Any])
def get_pipeline_config() -> Dict[str, Any]:
    return pipeline_service.load()


@app.get("/api/config/pipeline/yaml", response_class=PlainTextResponse)
def get_pipeline_config_yaml() -> str:
    config = pipeline_service.load()
    import yaml  # local import to avoid dependency at module import time

    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


@app.put("/api/config/pipeline", response_model=Dict[str, Any])
def save_pipeline_config(config: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return pipeline_service.save(config)


@app.patch("/api/config/pipeline", response_model=Dict[str, Any])
def update_pipeline_config(changes: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return pipeline_service.update(changes)


@app.get("/api/config/model", response_model=Dict[str, Any])
def get_model_config() -> Dict[str, Any]:
    return model_service.load()


@app.put("/api/config/model", response_model=Dict[str, Any])
def save_model_config(config: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return model_service.save(config)


@app.patch("/api/config/model", response_model=Dict[str, Any])
def update_model_config(changes: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return model_service.update(changes)


@app.get("/api/config/model/yaml", response_class=PlainTextResponse)
def get_model_config_yaml() -> str:
    config = model_service.load()
    import yaml

    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


@app.get("/healthz", response_class=PlainTextResponse)
def healthcheck() -> str:
    return "ok"
