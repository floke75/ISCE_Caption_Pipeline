"""FastAPI application that powers the operator UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config_service import PipelineConfigService
from .job_manager import JobManager
from .pipeline import (
    build_inference_steps,
    build_training_pair_steps,
    build_training_steps,
)
from .schemas import (
    ConfigPayload,
    InferenceJobRequest,
    JobInfo,
    TrainingJobRequest,
    TrainingPairJobRequest,
)


def create_app() -> FastAPI:
    project_root = Path(__file__).resolve().parents[2]
    config_service = PipelineConfigService(project_root / "pipeline_config.yaml")
    jobs_dir = project_root / "ui_runtime" / "jobs"
    manager = JobManager(jobs_dir, config_service=config_service)

    app = FastAPI(title="ISCE Pipeline Control Center", version="1.0.0")

    allowed_origins = os.getenv("UI_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/jobs", response_model=list[JobInfo])
    def list_jobs() -> list[JobInfo]:
        return [JobInfo(**item) for item in manager.list_jobs()]

    @app.get("/api/jobs/{job_id}", response_model=JobInfo)
    def get_job(job_id: str) -> JobInfo:
        record = manager.get_job(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobInfo(**record.to_dict())

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, tail: int = 1000) -> Dict[str, str]:
        return {"log": manager.read_log(job_id, tail=tail)}

    @app.post("/api/jobs/inference", response_model=JobInfo)
    def start_inference(request: InferenceJobRequest) -> JobInfo:
        job = manager.start_job(
            "inference",
            params=request.runtime_params(),
            build_steps=lambda workspace, cfg: build_inference_steps(workspace, cfg, request.runtime_params()),
            config_overrides=request.configOverrides,
        )
        return JobInfo(**job.to_dict())

    @app.post("/api/jobs/training-pair", response_model=JobInfo)
    def start_training_pair(request: TrainingPairJobRequest) -> JobInfo:
        job = manager.start_job(
            "training_pair",
            params=request.runtime_params(),
            build_steps=lambda workspace, cfg: build_training_pair_steps(workspace, cfg, request.runtime_params()),
            config_overrides=request.configOverrides,
        )
        return JobInfo(**job.to_dict())

    @app.post("/api/jobs/training", response_model=JobInfo)
    def start_training(request: TrainingJobRequest) -> JobInfo:
        job = manager.start_job(
            "training",
            params=request.runtime_params(),
            build_steps=lambda workspace, cfg: build_training_steps(workspace, cfg, request.runtime_params()),
            config_overrides=request.configOverrides,
        )
        return JobInfo(**job.to_dict())

    @app.get("/api/config/pipeline", response_model=ConfigPayload)
    def get_pipeline_config() -> ConfigPayload:
        return ConfigPayload(data=config_service.load())

    @app.put("/api/config/pipeline", response_model=ConfigPayload)
    def update_pipeline_config(payload: ConfigPayload) -> ConfigPayload:
        merged = config_service.apply_patch(payload.data)
        return ConfigPayload(data=merged)

    return app


app = create_app()
