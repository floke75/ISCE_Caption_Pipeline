"""FastAPI server powering the interactive pipeline UI."""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ui.jobs import JobStore
from ui.pipeline_runner import (
    run_inference_job,
    run_model_training_job,
    run_training_pair_job,
)


class InferenceRequest(BaseModel):
    media_path: str = Field(..., description="Path to the media file (audio/video).")
    transcript_path: Optional[str] = Field(
        None,
        description="Optional path to a prepared transcript (.txt).",
    )
    output_path: Optional[str] = Field(
        None,
        description="Optional destination for the generated SRT file.",
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Runtime overrides merged into pipeline_config.yaml values.",
    )
    pipeline_config_path: Optional[str] = Field(
        None,
        description="Alternate pipeline_config.yaml to load instead of the repository default.",
    )


class TrainingPairRequest(BaseModel):
    media_path: str = Field(..., description="Path to the media file (audio/video).")
    srt_path: str = Field(..., description="Path to the ground-truth SRT file.")
    config_overrides: Optional[Dict[str, Any]] = None
    pipeline_config_path: Optional[str] = None


class ModelTrainingRequest(BaseModel):
    corpus_dir: str = Field(..., description="Directory containing *.json training files.")
    iterations: int = Field(3, ge=1, description="Number of reweighting iterations.")
    error_boost_factor: float = Field(1.0, ge=0.0, description="Weight increment for misclassified samples.")
    constraints_output: Optional[str] = Field(
        None,
        description="Optional output path for constraints.json.",
    )
    weights_output: Optional[str] = Field(
        None,
        description="Optional output path for model_weights.json.",
    )
    config_path: Optional[str] = Field(
        None,
        description="Optional alternate config.yaml path.",
    )


class ConfigUpdate(BaseModel):
    content: str


app = FastAPI(title="ISCE Caption Pipeline UI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_store = JobStore()
executor = ThreadPoolExecutor(max_workers=4)

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "ui" / "frontend"
PIPELINE_CONFIG_PATH = REPO_ROOT / "pipeline_config.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "config.yaml"
UI_DATA_DIR = REPO_ROOT / "ui_data"
UI_DATA_DIR.mkdir(exist_ok=True)


@app.on_event("shutdown")
def _shutdown_executor() -> None:
    executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/jobs")
def list_jobs() -> Dict[str, Any]:
    records = sorted(job_store.list(), key=lambda r: r.created_at, reverse=True)
    return {"jobs": [r.to_dict() for r in records]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    record = job_store.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.to_dict()


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> Dict[str, str]:
    record = job_store.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"logs": job_store.export_logs(job_id)}


@app.post("/jobs/inference", status_code=202)
def create_inference_job(payload: InferenceRequest) -> Dict[str, str]:
    kwargs = {
        "media_path": Path(payload.media_path),
        "transcript_path": Path(payload.transcript_path) if payload.transcript_path else None,
        "output_path": Path(payload.output_path) if payload.output_path else None,
        "config_overrides": payload.config_overrides,
        "pipeline_config_path": Path(payload.pipeline_config_path) if payload.pipeline_config_path else None,
    }
    job_id = uuid.uuid4().hex
    job_store.create(job_id, "inference", payload.dict())
    executor.submit(
        run_inference_job,
        job_store,
        job_id,
        kwargs["media_path"],
        kwargs["transcript_path"],
        kwargs["output_path"],
        kwargs["config_overrides"],
        kwargs["pipeline_config_path"],
    )
    return {"job_id": job_id}


@app.post("/jobs/training-pair", status_code=202)
def create_training_pair_job(payload: TrainingPairRequest) -> Dict[str, str]:
    job_id = uuid.uuid4().hex
    job_store.create(job_id, "training_pair", payload.dict())
    executor.submit(
        run_training_pair_job,
        job_store,
        job_id,
        Path(payload.media_path),
        Path(payload.srt_path),
        payload.config_overrides,
        Path(payload.pipeline_config_path) if payload.pipeline_config_path else None,
    )
    return {"job_id": job_id}


@app.post("/jobs/model-training", status_code=202)
def create_model_training_job(payload: ModelTrainingRequest) -> Dict[str, str]:
    job_id = uuid.uuid4().hex
    job_store.create(job_id, "model_training", payload.dict())
    executor.submit(
        run_model_training_job,
        job_store,
        job_id,
        Path(payload.corpus_dir),
        payload.iterations,
        payload.error_boost_factor,
        Path(payload.constraints_output) if payload.constraints_output else None,
        Path(payload.weights_output) if payload.weights_output else None,
        Path(payload.config_path) if payload.config_path else None,
    )
    return {"job_id": job_id}


@app.get("/config/pipeline")
def read_pipeline_config() -> Dict[str, str]:
    if not PIPELINE_CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="pipeline_config.yaml not found")
    return {
        "path": str(PIPELINE_CONFIG_PATH),
        "content": PIPELINE_CONFIG_PATH.read_text(encoding="utf-8"),
    }


@app.put("/config/pipeline")
def write_pipeline_config(payload: ConfigUpdate) -> Dict[str, str]:
    try:
        import yaml

        yaml.safe_load(payload.content)  # validation only
    except Exception as exc:  # pragma: no cover - we want to surface validation errors clearly
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc

    PIPELINE_CONFIG_PATH.write_text(payload.content, encoding="utf-8")
    return {"status": "ok"}


@app.get("/config/model")
def read_model_config() -> Dict[str, str]:
    if not MODEL_CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="config.yaml not found")
    return {
        "path": str(MODEL_CONFIG_PATH),
        "content": MODEL_CONFIG_PATH.read_text(encoding="utf-8"),
    }


@app.put("/config/model")
def write_model_config(payload: ConfigUpdate) -> Dict[str, str]:
    try:
        import yaml

        yaml.safe_load(payload.content)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc

    MODEL_CONFIG_PATH.write_text(payload.content, encoding="utf-8")
    return {"status": "ok"}


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_frontend() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built")
    return FileResponse(index_path)
