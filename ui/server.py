from __future__ import annotations

import asyncio
import json
import os
import queue
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config_store import ConfigStore
from .job_manager import JobManager
from .schemas import InferenceRequest, ModelTrainingRequest, TrainingPairsRequest
from .tasks import PipelineTasks

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = REPO_ROOT / "ui" / "frontend" / "dist"
STATIC_OVERRIDE = os.environ.get("ISCE_UI_STATIC_DIR")
STATIC_DIR = Path(STATIC_OVERRIDE).expanduser().resolve() if STATIC_OVERRIDE else FRONTEND_DIST

pipeline_store = ConfigStore(REPO_ROOT / "pipeline_config.yaml")
model_store = ConfigStore(REPO_ROOT / "config.yaml")
job_manager = JobManager(REPO_ROOT / "ui_data" / "jobs", max_workers=3, queue_limit=24)
tasks = PipelineTasks(job_manager, pipeline_store, model_store)

app = FastAPI(title="ISCE Pipeline Control Center", version="1.1.0")

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
    queue_position: Optional[int] = Field(default=None, alias="queue_position")


class LogChunk(BaseModel):
    content: str
    offset: int
    complete: bool


class FileEntry(BaseModel):
    name: str
    path: str
    type: Literal["file", "directory"]


class FileBrowserResponse(BaseModel):
    path: str
    parent: Optional[str]
    entries: list[FileEntry]


class FileValidationRequest(BaseModel):
    path: str
    expect: Literal["file", "directory", "any"] = "any"


class FileValidationResponse(BaseModel):
    path: str
    type: Literal["file", "directory"]


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
def get_job_log(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1, le=1_000_000),
) -> LogChunk:
    try:
        chunk = job_manager.read_log(job_id, offset=offset, limit=limit)
    except KeyError as exc:  # noqa: PERF203
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return LogChunk(**chunk)


@app.get("/api/jobs/{job_id}/log/stream")
async def stream_job_log(request: Request, job_id: str) -> StreamingResponse:
    try:
        job_manager.get(job_id)
    except KeyError as exc:  # noqa: PERF203
        raise HTTPException(status_code=404, detail="Job not found") from exc

    start_offset = 0
    last_event_id = request.headers.get("last-event-id")
    if last_event_id is not None:
        try:
            start_offset = max(int(last_event_id), 0)
        except ValueError:
            start_offset = 0

    async def event_source() -> Any:
        offset = start_offset
        idle_cycles = 0
        while True:
            chunk = await asyncio.to_thread(job_manager.read_log, job_id, offset, None)
            content = chunk["content"]
            offset = chunk["offset"]
            payload = {"offset": offset, "complete": chunk["complete"]}

            if content:
                payload["content"] = content
                data = json.dumps(payload)
                yield f"id: {offset}\ndata: {data}\n\n"
                idle_cycles = 0
            elif idle_cycles >= 5:
                yield ": keep-alive\n\n"
                idle_cycles = 0
            else:
                idle_cycles += 1

            if chunk["complete"]:
                complete_payload = json.dumps(
                    {"offset": offset, "timestamp": datetime.utcnow().isoformat() + "Z"}
                )
                yield f"id: {offset}\nevent: complete\ndata: {complete_payload}\n\n"
                break

            if await request.is_disconnected():
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/jobs/inference", response_model=JobResponse, status_code=201)
def create_inference_job(payload: InferenceRequest) -> JobResponse:
    try:
        record = tasks.launch_inference(payload)
    except queue.Full as exc:
        raise HTTPException(status_code=429, detail="Job queue is full, try again later") from exc
    return JobResponse(**record.to_dict())


@app.post("/api/jobs/training-pairs", response_model=JobResponse, status_code=201)
def create_training_pairs_job(payload: TrainingPairsRequest) -> JobResponse:
    try:
        record = tasks.launch_training_pairs(payload)
    except queue.Full as exc:
        raise HTTPException(status_code=429, detail="Job queue is full, try again later") from exc
    return JobResponse(**record.to_dict())


@app.post("/api/jobs/model-training", response_model=JobResponse, status_code=201)
def create_model_training_job(payload: ModelTrainingRequest) -> JobResponse:
    try:
        record = tasks.launch_model_training(payload)
    except queue.Full as exc:
        raise HTTPException(status_code=429, detail="Job queue is full, try again later") from exc
    return JobResponse(**record.to_dict())


@app.get("/api/config/pipeline")
def read_pipeline_config() -> Dict[str, Any]:
    return {"config": pipeline_store.read(), "yaml": pipeline_store.dump_yaml()}


@app.put("/api/config/pipeline")
def update_pipeline_config(update: ConfigUpdate) -> Dict[str, Any]:
    if update.yaml is not None:
        try:
            parsed = yaml.safe_load(update.yaml) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
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
        try:
            parsed = yaml.safe_load(update.yaml) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="Parsed YAML must be a mapping")
        model_store.write(parsed)
    elif update.config is not None:
        model_store.update(update.config)
    return {"config": model_store.read(), "yaml": model_store.dump_yaml()}


@app.get("/api/files/browse", response_model=FileBrowserResponse)
def browse_files(path: Optional[str] = Query(None, alias="path")) -> FileBrowserResponse:
    target = _resolve_path(path) if path else REPO_ROOT
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path does not exist")

    directory = target if target.is_dir() else target.parent
    try:
        entries = list(directory.iterdir())
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    max_entries = 500
    directories: list[FileEntry] = []
    files: list[FileEntry] = []
    for item in sorted(entries, key=lambda entry: entry.name.lower()):
        if item.name.startswith("."):
            continue
        entry_type: Literal["file", "directory"] = "directory" if item.is_dir() else "file"
        entry = FileEntry(name=item.name, path=str(item.resolve()), type=entry_type)
        if entry_type == "directory":
            directories.append(entry)
        else:
            files.append(entry)
        if len(directories) + len(files) >= max_entries:
            break

    parent = directory.parent if directory != directory.parent else None
    return FileBrowserResponse(
        path=str(directory.resolve()),
        parent=str(parent.resolve()) if parent and parent.exists() else None,
        entries=[*directories, *files],
    )


@app.post("/api/files/validate", response_model=FileValidationResponse)
def validate_path(payload: FileValidationRequest) -> FileValidationResponse:
    target = _resolve_path(payload.path)
    if not target.exists():
        if payload.expect == "any":
            parent = target.parent if target.parent != target else None
            if parent and parent.exists():
                return FileValidationResponse(path=str(target), type="file")
        raise HTTPException(status_code=404, detail="Path does not exist")

    entry_type: Literal["file", "directory"] = "directory" if target.is_dir() else "file"
    if payload.expect == "file" and entry_type != "file":
        raise HTTPException(status_code=400, detail="Expected a file path")
    if payload.expect == "directory" and entry_type != "directory":
        raise HTTPException(status_code=400, detail="Expected a directory path")

    return FileValidationResponse(path=str(target.resolve()), type=entry_type)


def _resolve_path(path: Optional[str]) -> Path:
    if path is None:
        return REPO_ROOT
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.exists() else candidate.resolve(strict=False)


def _build_static_app(directory: Path):  # pragma: no cover - exercised via integration
    if directory.exists():
        return StaticFiles(directory=directory, html=True)

    message = (
        "Control center frontend has not been built. Run 'npm install' and 'npm run build' "
        "inside ui/frontend to generate the production bundle."
    )

    async def not_ready_app(scope, receive, send):
        if scope["type"] != "http":
            response = PlainTextResponse("Unsupported scope", status_code=500)
            await response(scope, receive, send)
            return
        if scope.get("method", "").upper() == "HEAD":
            response = PlainTextResponse("", status_code=503)
            await response(scope, receive, send)
            return
        response = PlainTextResponse(message, status_code=503)
        await response(scope, receive, send)

    return not_ready_app


@app.exception_handler(ValueError)
def handle_validation_error(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.mount("/", _build_static_app(STATIC_DIR), name="ui")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=False)
