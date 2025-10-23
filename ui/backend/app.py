"""Main FastAPI application for the ISCE pipeline UI backend.

This application serves as the central API gateway for the web-based user
interface. It is responsible for handling all client requests related to:

-   **Health Checks**: A simple endpoint to confirm the server is running.
-   **Configuration Management**: Endpoints for reading, updating, and validating
    both the main pipeline configuration (`pipeline_config.yaml`) and the
    segmentation model configuration (`config.yaml`). It supports both
    structured patch updates and raw YAML editing.
-   **Job Management**: A full suite of endpoints for creating, listing,
    retrieving, and canceling long-running pipeline jobs (e.g., inference,
    training pair generation, model training).
-   **Log Streaming**: Real-time log streaming for active jobs using Server-Sent
    Events (SSE), allowing the frontend to display live progress.
-   **File System Browsing**: A secure API for browsing the host file system
    within a predefined set of allowed root directories, enabling users to
    select input and output paths safely.

The application is structured to use several key services:
-   `ConfigService`: Manages loading, validation, and persistence of configuration.
-   `JobManager`: Handles the lifecycle of background processing jobs, including
    queuing, execution, and state management.
-   `FileBrowser`: Enforces security for file system access.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from .api.routes.files import FileBrowser, create_file_router
from .config_service import ConfigField, ConfigService, build_segmentation_field_catalog
from .job_manager import JobCancellationError, JobManager, JobQueueFull
from . import pipelines

REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_ROOT = REPO_ROOT / "ui_data"

max_workers = int(os.getenv("PIPELINE_MAX_CONCURRENT_JOBS", "3"))
queue_limit_env: Optional[str] = os.getenv("PIPELINE_MAX_QUEUED_JOBS")
queue_limit = int(queue_limit_env) if queue_limit_env else None

if max_workers < 1:
    raise ValueError("PIPELINE_MAX_CONCURRENT_JOBS must be at least 1")

pipeline_config_service = ConfigService(REPO_ROOT, STORAGE_ROOT)
segmentation_config_service = ConfigService(
    REPO_ROOT,
    STORAGE_ROOT,
    base_config_path=REPO_ROOT / "config.yaml",
    overrides_path=STORAGE_ROOT / "config" / "segmentation_overrides.yaml",
    field_catalog=build_segmentation_field_catalog(),
)
job_manager = JobManager(
    STORAGE_ROOT,
    pipeline_config_service,
    segmentation_config_service=segmentation_config_service,
    max_workers=max_workers,
    queue_limit=queue_limit,
)


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    parts = [part for part in slug.split("_") if part]
    return "_".join(parts)


def _load_file_roots() -> List[Tuple[str, str, Path]]:
    env_value = os.getenv("PIPELINE_FILE_ROOTS")
    roots: List[Tuple[str, str, Path]] = []
    seen_ids: Set[str] = set()

    if env_value:
        for index, raw in enumerate(env_value.split(os.pathsep), start=1):
            chunk = raw.strip()
            if not chunk:
                continue
            if "=" in chunk:
                label_raw, path_raw = chunk.split("=", 1)
                label = label_raw.strip() or f"Root {index}"
                path_text = path_raw.strip()
            else:
                path_text = chunk
                tentative = Path(path_text).expanduser()
                label = tentative.name or tentative.as_posix()
            identifier = _slugify(label) or f"root{index}"
            candidate = identifier
            suffix = 1
            while candidate in seen_ids:
                suffix += 1
                candidate = f"{identifier}_{suffix}"
            seen_ids.add(candidate)
            roots.append((candidate, label, Path(path_text)))
    else:
        roots.extend(
            [
                ("workspace", "Workspace storage", STORAGE_ROOT),
                ("repository", "Repository", REPO_ROOT),
            ]
        )

    return roots


file_browser = FileBrowser(_load_file_roots())

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


class ConfigTreeNodeModel(BaseModel):
    key: str
    path: List[str]
    label: str
    value_type: str = Field(alias="valueType")
    description: Optional[str] = None
    default: Any = None
    current: Any = None
    options: Optional[List[Any]] = None
    advanced: bool = False
    overridden: bool = False
    children: List["ConfigTreeNodeModel"] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class ConfigFieldModel(BaseModel):
    path: List[str]
    label: str
    field_type: str = Field(alias="fieldType")
    section: str
    description: Optional[str] = None
    options: Optional[List[Any]] = None
    advanced: bool = False
    read_only: bool = Field(default=False, alias="readOnly")

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
            readOnly=field.read_only,
        )

    model_config = ConfigDict(populate_by_name=True)


class ConfigSnapshot(BaseModel):
    effective: Dict[str, Any]
    overrides: Dict[str, Any]
    fields: List[ConfigFieldModel]
    config_schema: List[ConfigTreeNodeModel] = Field(alias="schema")

    model_config = ConfigDict(populate_by_name=True)


class ConfigPatch(BaseModel):
    updates: Dict[str, Any]


class ConfigReplace(BaseModel):
    overrides: Dict[str, Any]


class ConfigYamlUpdate(BaseModel):
    yaml: str


ConfigTreeNodeModel.update_forward_refs()

app.include_router(create_file_router(file_browser))


def _snapshot_for(service: ConfigService) -> ConfigSnapshot:
    effective = service.effective_config()
    overrides = service.stored_overrides()
    fields = [ConfigFieldModel.from_field(f) for f in service.describe_fields()]
    schema = [ConfigTreeNodeModel(**node) for node in service.describe_tree()]
    return ConfigSnapshot(effective=effective, overrides=overrides, fields=fields, config_schema=schema)


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
    workspace_path: str = Field(alias="workspacePath")

    model_config = ConfigDict(populate_by_name=True)


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
    """Provides a simple health check endpoint.

    Returns:
        A dictionary with a single key "status" set to "ok".
    """
    return {"status": "ok"}


@app.get("/api/config/pipeline", response_model=ConfigSnapshot)
def get_pipeline_config() -> ConfigSnapshot:
    """Retrieves the current state of the main pipeline configuration.

    This endpoint returns a comprehensive snapshot of the `pipeline_config.yaml`
    settings, including the effective (merged) configuration, the stored
    overrides, and a detailed schema of all available fields for UI rendering.

    Returns:
        A `ConfigSnapshot` object containing the full configuration details.
    """
    return _snapshot_for(pipeline_config_service)


@app.put("/api/config/pipeline", response_model=ConfigSnapshot)
def update_pipeline_config(payload: ConfigPatch) -> ConfigSnapshot:
    """Applies a partial update to the pipeline configuration overrides.

    This endpoint accepts a dictionary of dot-separated keys and values,
    allowing the frontend to update specific, nested configuration fields
    without needing to send the entire configuration object.

    Args:
        payload: A `ConfigPatch` object containing the updates.

    Returns:
        The new, updated `ConfigSnapshot`.
    """
    patch = pipeline_config_service.build_patch(payload.updates)
    pipeline_config_service.apply_patch(patch)
    return get_pipeline_config()


@app.put("/api/config/pipeline/replace", response_model=ConfigSnapshot)
def replace_pipeline_config(payload: ConfigReplace) -> ConfigSnapshot:
    """Replaces the entire pipeline configuration override file.

    Args:
        payload: A `ConfigReplace` object with the new override dictionary.

    Returns:
        The new `ConfigSnapshot`.
    """
    pipeline_config_service.save_overrides(payload.overrides)
    return get_pipeline_config()


@app.get("/api/config/pipeline/raw")
def get_pipeline_config_yaml() -> Dict[str, str]:
    """Retrieves the raw YAML content of the pipeline configuration overrides.

    Returns:
        A dictionary containing the YAML content as a string.
    """
    import yaml

    overrides = pipeline_config_service.stored_overrides()
    return {"yaml": yaml.safe_dump(overrides, allow_unicode=True, sort_keys=False)}


@app.put("/api/config/pipeline/raw", response_model=ConfigSnapshot)
def update_pipeline_config_yaml(payload: ConfigYamlUpdate) -> ConfigSnapshot:
    """Updates the pipeline configuration from a raw YAML string.

    This endpoint is used by the raw YAML editor in the UI. It parses the
    provided YAML string and, if valid, saves it as the new override file.

    Args:
        payload: A `ConfigYamlUpdate` object containing the new YAML string.

    Raises:
        HTTPException: If the provided string is not valid YAML or does not
                       represent a dictionary.

    Returns:
        The new `ConfigSnapshot` after the update.
    """
    import yaml

    try:
        overrides = yaml.safe_load(payload.yaml) if payload.yaml.strip() else {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=400, detail="YAML payload must represent a mapping")
    pipeline_config_service.save_overrides(overrides)
    return get_pipeline_config()


@app.get("/api/config/segmentation", response_model=ConfigSnapshot)
def get_segmentation_config() -> ConfigSnapshot:
    """Retrieves the current state of the segmentation model configuration.

    This endpoint returns a comprehensive snapshot of the `config.yaml`
    settings, including the effective (merged) configuration, the stored
    overrides, and a detailed schema for UI rendering.

    Returns:
        A `ConfigSnapshot` object containing the full configuration details.
    """
    return _snapshot_for(segmentation_config_service)


@app.put("/api/config/segmentation", response_model=ConfigSnapshot)
def update_segmentation_config(payload: ConfigPatch) -> ConfigSnapshot:
    """Applies a partial update to the segmentation configuration overrides.

    Args:
        payload: A `ConfigPatch` object containing the updates.

    Returns:
        The new, updated `ConfigSnapshot`.
    """
    patch = segmentation_config_service.build_patch(payload.updates)
    segmentation_config_service.apply_patch(patch)
    return get_segmentation_config()


@app.put("/api/config/segmentation/replace", response_model=ConfigSnapshot)
def replace_segmentation_config(payload: ConfigReplace) -> ConfigSnapshot:
    """Replaces the entire segmentation configuration override file.

    Args:
        payload: A `ConfigReplace` object with the new override dictionary.

    Returns:
        The new `ConfigSnapshot`.
    """
    segmentation_config_service.save_overrides(payload.overrides)
    return get_segmentation_config()


@app.get("/api/config/segmentation/raw")
def get_segmentation_config_yaml() -> Dict[str, str]:
    """Retrieves the raw YAML content of the segmentation configuration overrides.

    Returns:
        A dictionary containing the YAML content as a string.
    """
    import yaml

    overrides = segmentation_config_service.stored_overrides()
    return {"yaml": yaml.safe_dump(overrides, allow_unicode=True, sort_keys=False)}


@app.put("/api/config/segmentation/raw", response_model=ConfigSnapshot)
def update_segmentation_config_yaml(payload: ConfigYamlUpdate) -> ConfigSnapshot:
    """Updates the segmentation configuration from a raw YAML string.

    Args:
        payload: A `ConfigYamlUpdate` object containing the new YAML string.

    Raises:
        HTTPException: If the provided string is not valid YAML or does not
                       represent a dictionary.

    Returns:
        The new `ConfigSnapshot` after the update.
    """
    import yaml

    try:
        overrides = yaml.safe_load(payload.yaml) if payload.yaml.strip() else {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=400, detail="YAML payload must represent a mapping")
    segmentation_config_service.save_overrides(overrides)
    return get_segmentation_config()


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
        workspacePath=str(record.workspace),
    )


@app.get("/api/jobs", response_model=List[JobModel])
def list_jobs() -> List[JobModel]:
    """Lists all known jobs, including pending, active, and completed.

    Returns:
        A list of `JobModel` objects, sorted from most to least recent.
    """
    return [_serialize_job(job) for job in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobModel)
def get_job(job_id: str) -> JobModel:
    """Retrieves the details of a single job by its ID.

    Args:
        job_id: The unique identifier of the job.

    Raises:
        HTTPException: If no job with the specified ID is found.

    Returns:
        The `JobModel` object for the requested job.
    """
    try:
        record = job_manager.get_job(job_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return _serialize_job(record)


def _job_log_payload(job_id: str, tail: int) -> Dict[str, Any]:
    try:
        log = job_manager.get_log_tail(job_id, limit=tail)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return {"log": log}


@app.get("/api/jobs/{job_id}/log")
def get_job_log(job_id: str, tail: int = Query(4000, ge=100, le=20000)) -> Dict[str, Any]:
    """Retrieves the tail of a job's log file.

    Args:
        job_id: The ID of the job.
        tail: The number of characters to retrieve from the end of the log.

    Returns:
        A dictionary containing the log content.
    """
    return _job_log_payload(job_id, tail)


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, tail: int = Query(4000, ge=100, le=20000)) -> Dict[str, Any]:
    """Retrieves the tail of a job's log file (alias for /log).

    Args:
        job_id: The ID of the job.
        tail: The number of characters to retrieve from the end of the log.

    Returns:
        A dictionary containing the log content.
    """
    return _job_log_payload(job_id, tail)


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


def _stream_job_log(job_id: str) -> StreamingResponse:
    generator = _log_event_stream(job_id)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(generator, media_type="text/event-stream", headers=headers)


@app.get("/api/jobs/{job_id}/logs/stream")
def stream_job_logs(job_id: str) -> StreamingResponse:
    """Streams the log of a running job in real-time using SSE.

    This endpoint establishes a long-lived connection with the client, sending
    new log entries as they are written. It also sends a completion event
    when the job finishes.

    Args:
        job_id: The ID of the job to monitor.

    Returns:
        A `StreamingResponse` that sends `text/event-stream` data.
    """
    return _stream_job_log(job_id)


@app.get("/api/jobs/{job_id}/log/stream")
def stream_job_log_legacy(job_id: str) -> StreamingResponse:
    """Legacy alias for the log streaming endpoint."""
    return _stream_job_log(job_id)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobModel)
def cancel_job(job_id: str) -> JobModel:
    """Requests cancellation of a pending or active job.

    Args:
        job_id: The ID of the job to cancel.

    Raises:
        HTTPException: If the job is not found or is already completed and
                       cannot be cancelled.

    Returns:
        The updated `JobModel` with a 'cancelled' status.
    """
    try:
        record = job_manager.cancel_job(job_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except JobCancellationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _serialize_job(record)


@app.post("/api/jobs/inference", response_model=JobModel, status_code=201)
def create_inference_job(payload: InferenceJobRequest) -> JobModel:
    """Creates and queues a new inference job.

    Args:
        payload: An `InferenceJobRequest` with the job parameters.

    Raises:
        HTTPException: If the job queue is full.

    Returns:
        The newly created `JobModel`.
    """
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
    """Creates and queues a new training pair generation job.

    Args:
        payload: A `TrainingPairRequest` with the job parameters.

    Raises:
        HTTPException: If the job queue is full.

    Returns:
        The newly created `JobModel`.
    """
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
    """Creates and queues a new model training job.

    Args:
        payload: A `ModelTrainingRequest` with the job parameters.

    Raises:
        HTTPException: If the job queue is full.

    Returns:
        The newly created `JobModel`.
    """
    try:
        record = job_manager.create_job(
            "model_training",
            params=payload.dict(exclude_none=True),
            runner=pipelines.run_model_training,
        )
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _serialize_job(record)


__all__ = ["app", "file_browser"]

