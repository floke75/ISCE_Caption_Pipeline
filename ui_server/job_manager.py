from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class JobStatus(str):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    id: str
    job_type: str
    name: str
    parameters: Dict[str, Any]
    status: str = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    log_path: Path = field(default_factory=Path)
    workspace: Path = field(default_factory=Path)
    metrics: Dict[str, Any] = field(default_factory=dict)


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str) -> None:
        self._manager = manager
        self._job_id = job_id

    @property
    def job_id(self) -> str:
        return self._job_id

    def update(self, *, progress: Optional[float] = None, stage: Optional[str] = None,
               message: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        self._manager.update_job(
            self._job_id,
            progress=progress,
            stage=stage,
            message=message,
            metrics=metrics,
        )

    def log(self, text: str) -> None:
        self._manager.append_log(self._job_id, text)

    def workspace(self) -> Path:
        job = self._manager.get_job(self._job_id)
        if not job:
            raise RuntimeError("Job context is no longer available")
        return job.workspace

    def log_path(self) -> Path:
        job = self._manager.get_job(self._job_id)
        if not job:
            raise RuntimeError("Job context is no longer available")
        return job.log_path


class JobManager:
    def __init__(self, runtime_root: Path, max_workers: int = 3) -> None:
        self._runtime_root = runtime_root
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        self._jobs_root = self._runtime_root / "jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._semaphore = threading.BoundedSemaphore(max_workers)
        self._load_jobs()

    @property
    def runtime_root(self) -> Path:
        return self._runtime_root

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True))

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return copy_job(self._jobs.get(job_id))

    def create_job(
        self,
        job_type: str,
        name: str,
        parameters: Dict[str, Any],
        runner: callable[[JobContext], Dict[str, Any]],
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        workspace = self._jobs_root / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "job.log"
        record = JobRecord(
            id=job_id,
            job_type=job_type,
            name=name,
            parameters=parameters,
            log_path=log_path,
            workspace=workspace,
        )
        with self._lock:
            self._jobs[job_id] = record
            self._save_job(record)
        thread = threading.Thread(target=self._run_job, args=(record.id, runner), daemon=True)
        thread.start()
        return copy_job(record)

    def _run_job(self, job_id: str, runner: callable[[JobContext], Dict[str, Any]]) -> None:
        context = JobContext(self, job_id)
        with self._semaphore:
            self.update_job(
                job_id,
                status=JobStatus.RUNNING,
                started_at=datetime.utcnow(),
                stage="initialising",
            )
            try:
                result = runner(context)
            except Exception as exc:  # pylint: disable=broad-except
                self.update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    finished_at=datetime.utcnow(),
                    error=str(exc),
                )
                context.log(f"[ERROR] {exc}")
                return
            self.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                finished_at=datetime.utcnow(),
                progress=1.0,
                stage="completed",
                result=result,
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            if status:
                record.status = status
            if progress is not None:
                record.progress = max(0.0, min(1.0, progress))
            if stage:
                record.stage = stage
            if message:
                record.message = message
            if result is not None:
                record.result = result
            if error:
                record.error = error
            if metrics:
                record.metrics.update(metrics)
            if started_at:
                record.started_at = started_at
            if finished_at:
                record.finished_at = finished_at
            self._save_job(record)

    def append_log(self, job_id: str, text: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        timestamp = datetime.utcnow().isoformat()
        with job.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {text}\n")

    def read_log(self, job_id: str, offset: int = 0) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job or not job.log_path.exists():
            return {"content": "", "next_offset": 0}
        with job.log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(offset)
            content = handle.read()
            next_offset = handle.tell()
        return {"content": content, "next_offset": next_offset}

    def _load_jobs(self) -> None:
        for meta_path in self._jobs_root.glob("*/job.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            job_id = data.get("id") or meta_path.parent.name
            workspace = self._resolve_workspace(data.get("workspace"), job_id)
            workspace.mkdir(parents=True, exist_ok=True)
            record = JobRecord(
                id=job_id,
                job_type=data.get("job_type", "unknown"),
                name=data.get("name", job_id),
                parameters=data.get("parameters", {}),
                status=data.get("status", JobStatus.PENDING),
                created_at=_parse_datetime(data.get("created_at")) or datetime.utcnow(),
                started_at=_parse_datetime(data.get("started_at")),
                finished_at=_parse_datetime(data.get("finished_at")),
                progress=float(data.get("progress", 0.0)),
                stage=data.get("stage"),
                message=data.get("message"),
                result=data.get("result"),
                error=data.get("error"),
                log_path=workspace / "job.log",
                workspace=workspace,
                metrics=data.get("metrics", {}),
            )
            if record.status in {JobStatus.RUNNING, JobStatus.PENDING}:
                record.status = JobStatus.CANCELLED
                record.message = "Job state reset after server restart."
                if not record.finished_at:
                    record.finished_at = datetime.utcnow()
            with self._lock:
                self._jobs[job_id] = record
                self._save_job(record)

    def _resolve_workspace(self, stored: Optional[str], job_id: str) -> Path:
        if stored:
            path = Path(stored)
            if not path.is_absolute():
                return (self._runtime_root / path).resolve()
            return path
        return self._jobs_root / job_id

    def _save_job(self, record: JobRecord) -> None:
        record.workspace.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": record.id,
            "job_type": record.job_type,
            "name": record.name,
            "parameters": record.parameters,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "finished_at": record.finished_at.isoformat() if record.finished_at else None,
            "progress": record.progress,
            "stage": record.stage,
            "message": record.message,
            "result": record.result,
            "error": record.error,
            "metrics": record.metrics,
            "workspace": self._relative_workspace(record.workspace),
        }
        meta_path = record.workspace / "job.json"
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, default=_json_default, indent=2)

    def _relative_workspace(self, workspace: Path) -> str:
        try:
            return str(workspace.relative_to(self._runtime_root))
        except ValueError:
            return str(workspace)


def copy_job(record: Optional[JobRecord]) -> Optional[JobRecord]:
    if record is None:
        return None
    return JobRecord(
        id=record.id,
        job_type=record.job_type,
        name=record.name,
        parameters=record.parameters.copy(),
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        progress=record.progress,
        stage=record.stage,
        message=record.message,
        result=record.result.copy() if record.result else None,
        error=record.error,
        log_path=record.log_path,
        workspace=record.workspace,
        metrics=record.metrics.copy(),
    )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)
