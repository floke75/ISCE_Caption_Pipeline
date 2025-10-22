"""Background job orchestration for the UI backend."""
from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from .config_service import ConfigService, _recursive_update

JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


@dataclass
class JobRecord:
    """Represents a background pipeline execution."""

    id: str
    job_type: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    params: Dict[str, Any]
    workspace: Path
    log_path: Path
    progress: float = 0.0
    message: str = "Pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "message": self.message,
            "params": self.params,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_file(cls, path: Path) -> "JobRecord":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(
            id=payload["id"],
            job_type=payload["job_type"],
            status=payload.get("status", "pending"),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload.get("updated_at", payload["created_at"])),
            params=payload.get("params", {}),
            workspace=path.parent,
            log_path=path.parent / "job.log",
            progress=payload.get("progress", 0.0),
            message=payload.get("message", ""),
            result=payload.get("result"),
            error=payload.get("error"),
        )

    def write_metadata(self) -> None:
        payload = self.to_dict()
        payload.update({
            "workspace": str(self.workspace),
            "log_path": str(self.log_path),
        })
        meta_path = self.workspace / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


class JobContext:
    """Context object passed to pipeline runners for status/log updates."""

    def __init__(self, manager: "JobManager", record: JobRecord) -> None:
        self._manager = manager
        self.record = record
        self._log_lock = threading.Lock()
        record.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = record.log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        line = f"[{timestamp}] {message}\n"
        with self._log_lock:
            self._log_file.write(line)
            self._log_file.flush()

    def stream_command(self, command: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
        """Run a subprocess and stream its output into the job log."""
        import subprocess

        display = " ".join(str(c) for c in command)
        self.log(f"$ {display}")
        process = subprocess.Popen(
            [str(c) for c in command],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            with self._log_lock:
                self._log_file.write(line)
                self._log_file.flush()
        process.stdout.close()
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {display}")
        with self._log_lock:
            self._log_file.flush()

    def update(self, *, progress: Optional[float] = None, message: Optional[str] = None) -> None:
        self._manager.update_job(self.record.id, progress=progress, message=message)

    def effective_config(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._manager.prepare_runtime_config(self.record, overrides or {})

    def finalize(self, status: JobStatus, *, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        self._manager.finalize_job(self.record.id, status=status, result=result, error=error)

    def close(self) -> None:
        with self._log_lock:
            self._log_file.close()

    def __enter__(self) -> "JobContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class JobManager:
    """Coordinates background pipeline jobs."""

    def __init__(self, storage_root: Path, config_service: ConfigService, max_workers: int = 3) -> None:
        self._storage_root = storage_root
        self._config_service = config_service
        self._jobs: Dict[str, JobRecord] = {}
        self._futures: Dict[str, Future[Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs_root = storage_root / "jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._load_existing_jobs()

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    def create_job(self, job_type: str, params: Dict[str, Any], runner) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        workspace = self._jobs_root / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        record = JobRecord(
            id=job_id,
            job_type=job_type,
            status="pending",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            params=params,
            workspace=workspace,
            log_path=workspace / "job.log",
        )
        record.write_metadata()
        with self._lock:
            self._jobs[job_id] = record
            future = self._executor.submit(self._run_job, record, runner)
            self._futures[job_id] = future
        return record

    def _run_job(self, record: JobRecord, runner) -> None:
        with JobContext(self, record) as ctx:
            self.update_job(record.id, status="running", progress=0.02, message="Starting")
            try:
                runner(ctx)
                with self._lock:
                    current = self._jobs[record.id]
                    if current.status == "running":
                        ctx.finalize("succeeded")
            except Exception as exc:  # pragma: no cover - defensive
                ctx.log(f"Job failed: {exc}")
                ctx.finalize("failed", error=str(exc))
                raise

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            if status is not None:
                record.status = status
            if progress is not None:
                record.progress = max(0.0, min(1.0, progress))
            if message is not None:
                record.message = message
            record.updated_at = datetime.utcnow()
            record.write_metadata()

    def finalize_job(
        self,
        job_id: str,
        *,
        status: JobStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = status
            record.result = result
            record.error = error
            record.progress = 1.0 if status == "succeeded" else record.progress
            record.updated_at = datetime.utcnow()
            record.write_metadata()

    def prepare_runtime_config(self, record: JobRecord, overrides: Dict[str, Any]) -> Dict[str, Any]:
        base = self._config_service.base_config()
        stored_overrides = self._config_service.stored_overrides()
        config = _recursive_update(base, stored_overrides)
        config["project_root"] = str(self._storage_root.parent)
        pipeline_root = record.workspace / "pipeline"
        config["pipeline_root"] = str(pipeline_root)
        runtime = _recursive_update(config, overrides)
        runtime = self._config_service.resolve_paths(runtime)
        pipeline_root.mkdir(parents=True, exist_ok=True)
        self._write_runtime_config(record.workspace, runtime)
        runtime["__path__"] = str(record.workspace / "pipeline_config.runtime.yaml")
        return runtime

    def _write_runtime_config(self, workspace: Path, config: Dict[str, Any]) -> None:
        config_path = workspace / "pipeline_config.runtime.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)

    # ------------------------------------------------------------------
    # Query APIs
    # ------------------------------------------------------------------
    def list_jobs(self) -> List[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            return self._jobs[job_id]

    def get_log_tail(self, job_id: str, limit: int = 4000) -> str:
        record = self.get_job(job_id)
        if not record.log_path.exists():
            return ""
        with record.log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(size - limit, 0))
            return fh.read()

    # ------------------------------------------------------------------
    # Bootstrapping
    # ------------------------------------------------------------------
    def _load_existing_jobs(self) -> None:
        for meta_path in self._jobs_root.glob("*/metadata.json"):
            record = JobRecord.from_file(meta_path)
            self._jobs[record.id] = record


