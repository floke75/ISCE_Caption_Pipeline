"""FastAPI server exposing the ISCE pipeline through a web UI friendly API."""
from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import yaml
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, root_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipeline_config import load_pipeline_config
from run_pipeline import (  # type: ignore  # pylint: disable=wrong-import-position
    DEFAULT_SETTINGS,
    process_inference_file,
    process_training_file,
    setup_directories,
)
from scripts.train_model import run_training  # type: ignore  # pylint: disable=wrong-import-position

PIPELINE_CONFIG_FILE = REPO_ROOT / "pipeline_config.yaml"
MODEL_CONFIG_FILE = REPO_ROOT / "config.yaml"


class JobStatus(str):
    """Simple enum-like container for job states."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class JobRecord:
    """Represents a background job managed by the API."""

    id: str
    job_type: str
    status: str
    created_at: datetime
    updated_at: datetime
    params: Dict[str, Any]
    log: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobLogStream(io.TextIOBase):
    """File-like object that writes log chunks into a job record."""

    def __init__(self, manager: "JobManager", job_id: str, original: Optional[io.TextIOBase]):
        self._manager = manager
        self._job_id = job_id
        self._original = original

    def write(self, data: str) -> int:  # type: ignore[override]
        if not data:
            return 0
        self._manager.append_log(self._job_id, data)
        if self._original is not None:
            self._original.write(data)
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        if self._original is not None:
            self._original.flush()


class JobManager:
    """Threaded job runner that keeps track of submission metadata and logs."""

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def submit(self, job_type: str, params: Dict[str, Any], func: Callable[[], Dict[str, Any] | None]) -> JobRecord:
        job_id = uuid4().hex
        record = JobRecord(
            id=job_id,
            job_type=job_type,
            status=JobStatus.QUEUED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            params=params,
        )
        with self._lock:
            self._jobs[job_id] = record

        self._executor.submit(self._run_job, job_id, func)
        return self.get(job_id)

    def _run_job(self, job_id: str, func: Callable[[], Dict[str, Any] | None]) -> None:
        self._set_status(job_id, JobStatus.RUNNING)
        try:
            with self._capture_logs(job_id):
                result = func() or {}
            self._complete(job_id, result)
        except Exception as exc:  # pylint: disable=broad-except
            self.append_log(job_id, f"\n[ERROR] {exc}\n")
            self._fail(job_id, str(exc))

    def _set_status(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            job.updated_at = datetime.utcnow()

    def _complete(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.SUCCEEDED
            job.result = result
            job.updated_at = datetime.utcnow()

    def _fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = datetime.utcnow()

    @contextlib.contextmanager
    def _capture_logs(self, job_id: str):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        stdout_stream = JobLogStream(self, job_id, original_stdout)
        stderr_stream = JobLogStream(self, job_id, original_stderr)
        try:
            with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
                yield
        finally:
            stdout_stream.flush()
            stderr_stream.flush()

    def append_log(self, job_id: str, chunk: str) -> None:
        if not chunk:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.log += chunk
            job.updated_at = datetime.utcnow()

    def list(self) -> List[JobRecord]:
        with self._lock:
            return [copy.deepcopy(record) for record in self._jobs.values()]

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return copy.deepcopy(record)


job_manager = JobManager()


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return a new dict."""

    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pipeline_settings() -> Dict[str, Any]:
    """Load the pipeline configuration from disk with defaults."""

    return load_pipeline_config(DEFAULT_SETTINGS, str(PIPELINE_CONFIG_FILE))


def ensure_pipeline_dirs(cfg: Dict[str, Any]) -> None:
    """Create required directories before running the pipeline."""

    try:
        setup_directories(cfg)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing configuration key: {exc}") from exc


def _prepare_transcript(cfg: Dict[str, Any], media_path: Path, transcript_path: Optional[Path]) -> None:
    if transcript_path is None:
        return
    if transcript_path.suffix.lower() != ".txt":
        raise HTTPException(status_code=400, detail="Transcript must be a .txt file")
    if not transcript_path.exists():
        raise HTTPException(status_code=400, detail=f"Transcript not found: {transcript_path}")

    destination_dir = Path(cfg["intermediate_dir"]) / "_ui_txt_inputs"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{media_path.stem}.txt"
    shutil.copy2(transcript_path, destination)
    cfg["txt_placement_folder"] = str(destination_dir)


class InferenceRequest(BaseModel):
    media_file: str = Field(..., description="Absolute path to the media file.")
    transcript_file: Optional[str] = Field(None, description="Optional transcript to align against.")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Per-run overrides for pipeline_config values.")

    @root_validator
    def _validate_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        media = values.get("media_file")
        if not media:
            raise ValueError("media_file is required")
        if not Path(media).exists():
            raise ValueError(f"Media file not found: {media}")
        transcript = values.get("transcript_file")
        if transcript and not Path(transcript).exists():
            raise ValueError(f"Transcript file not found: {transcript}")
        return values


class TrainingPairRequest(BaseModel):
    media_file: str = Field(..., description="Absolute path to the media file.")
    srt_file: str = Field(..., description="Ground-truth SRT file to align.")
    config_overrides: Optional[Dict[str, Any]] = None

    @root_validator
    def _validate_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        media = values.get("media_file")
        srt = values.get("srt_file")
        if not Path(media).exists():
            raise ValueError(f"Media file not found: {media}")
        if not Path(srt).exists():
            raise ValueError(f"SRT file not found: {srt}")
        return values


class ModelTrainingRequest(BaseModel):
    corpus_dir: str = Field(..., description="Directory containing *.json training files.")
    constraints_path: str = Field(..., description="Output path for constraints.json")
    weights_path: str = Field(..., description="Output path for model_weights.json")
    config_path: str = Field("config.yaml", description="Path to the feature configuration YAML")
    iterations: int = Field(3, ge=1, description="Number of training iterations")
    error_boost_factor: float = Field(1.0, ge=0.0, description="Weight increment for misclassified samples")

    @root_validator
    def _check_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        corpus_dir = Path(values.get("corpus_dir", ""))
        if not corpus_dir.exists() or not corpus_dir.is_dir():
            raise ValueError(f"Corpus directory not found: {corpus_dir}")
        return values


class JobSummary(BaseModel):
    id: str
    job_type: str
    status: str
    created_at: datetime
    updated_at: datetime


class JobDetail(JobSummary):
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    log: str


def job_to_summary(record: JobRecord) -> JobSummary:
    return JobSummary(
        id=record.id,
        job_type=record.job_type,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def job_to_detail(record: JobRecord) -> JobDetail:
    return JobDetail(
        **job_to_summary(record).dict(),
        params=record.params,
        result=record.result,
        error=record.error,
        log=record.log,
    )


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ValueError("Configuration must be a mapping")
            return data
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read {path.name}: {exc}") from exc


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to write {path.name}: {exc}") from exc


api_router = APIRouter(prefix="/api")


@api_router.post("/jobs/inference", response_model=JobSummary, status_code=201)
def launch_inference_job(payload: InferenceRequest) -> JobSummary:
    cfg = load_pipeline_settings()
    if payload.config_overrides:
        cfg = deep_update(cfg, payload.config_overrides)

    media_path = Path(payload.media_file)
    transcript_path = Path(payload.transcript_file) if payload.transcript_file else None

    ensure_pipeline_dirs(cfg)
    _prepare_transcript(cfg, media_path, transcript_path)

    def runner() -> Dict[str, Any]:
        process_inference_file(media_path, cfg)
        result = {
            "media_file": str(media_path),
            "output_srt": str(Path(cfg["output_dir"]) / f"{media_path.stem}.srt"),
            "enriched_json": str(Path(cfg["intermediate_dir"]) / "_inference_input" / f"{media_path.stem}.enriched.json"),
        }
        return result

    record = job_manager.submit(
        job_type="inference",
        params={"media_file": payload.media_file, "transcript_file": payload.transcript_file},
        func=runner,
    )
    return job_to_summary(record)


@api_router.post("/jobs/training-pair", response_model=JobSummary, status_code=201)
def launch_training_pair_job(payload: TrainingPairRequest) -> JobSummary:
    cfg = load_pipeline_settings()
    if payload.config_overrides:
        cfg = deep_update(cfg, payload.config_overrides)

    media_path = Path(payload.media_file)
    srt_path = Path(payload.srt_file)

    ensure_pipeline_dirs(cfg)

    def runner() -> Dict[str, Any]:
        process_training_file(media_path, srt_path, cfg)
        result = {
            "media_file": str(media_path),
            "srt_file": str(srt_path),
            "train_json": str(Path(cfg["intermediate_dir"]) / "_training" / f"{media_path.stem}.train.words.json"),
        }
        return result

    record = job_manager.submit(
        job_type="training_pair",
        params={"media_file": payload.media_file, "srt_file": payload.srt_file},
        func=runner,
    )
    return job_to_summary(record)


@api_router.post("/jobs/model-training", response_model=JobSummary, status_code=201)
def launch_model_training_job(payload: ModelTrainingRequest) -> JobSummary:
    def runner() -> Dict[str, Any]:
        return run_training(
            corpus=payload.corpus_dir,
            constraints_path=payload.constraints_path,
            weights_path=payload.weights_path,
            config_path=payload.config_path,
            iterations=payload.iterations,
            error_boost_factor=payload.error_boost_factor,
        )

    record = job_manager.submit(
        job_type="model_training",
        params=payload.dict(),
        func=runner,
    )
    return job_to_summary(record)


@api_router.get("/jobs", response_model=List[JobSummary])
def list_jobs() -> List[JobSummary]:
    return [job_to_summary(job) for job in job_manager.list()]


@api_router.get("/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str) -> JobDetail:
    try:
        record = job_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from exc
    return job_to_detail(record)


@api_router.get("/config/pipeline")
def read_pipeline_config(format: str = Query("json", description="Response format: json or yaml")) -> JSONResponse | PlainTextResponse:
    cfg = read_yaml(PIPELINE_CONFIG_FILE)
    if format == "yaml":
        yaml_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        return PlainTextResponse(yaml_text)
    return JSONResponse({"path": str(PIPELINE_CONFIG_FILE), "data": cfg})


@api_router.put("/config/pipeline", response_model=Dict[str, Any])
def update_pipeline_config(data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Pipeline configuration must be an object")
    write_yaml(PIPELINE_CONFIG_FILE, data)
    return {"path": str(PIPELINE_CONFIG_FILE), "data": data}


@api_router.get("/config/model")
def read_model_config(format: str = Query("json", description="Response format: json or yaml")) -> JSONResponse | PlainTextResponse:
    cfg = read_yaml(MODEL_CONFIG_FILE)
    if format == "yaml":
        yaml_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        return PlainTextResponse(yaml_text)
    return JSONResponse({"path": str(MODEL_CONFIG_FILE), "data": cfg})


@api_router.put("/config/model", response_model=Dict[str, Any])
def update_model_config(data: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Model configuration must be an object")
    write_yaml(MODEL_CONFIG_FILE, data)
    return {"path": str(MODEL_CONFIG_FILE), "data": data}


app = FastAPI(
    title="ISCE Pipeline Control Plane",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


def _cors_allow_origins() -> List[str]:
    """Return the list of origins allowed to access the API."""

    raw = os.getenv("UI_CORS_ORIGINS", "")
    if raw:
        origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
        if origins:
            return origins
    # Default to common local development hosts.
    return ["http://localhost:3000", "http://localhost:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.get("/")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


def _mount_frontend(app: FastAPI) -> None:
    dist_dir = REPO_ROOT / "ui" / "frontend" / "dist"
    if dist_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(dist_dir), html=True), name="ui-frontend")


_mount_frontend(app)
