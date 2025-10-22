"""Background job management for the UI service."""

from __future__ import annotations

import contextlib
import io
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


JobTarget = Callable[["Job"], Any]


@dataclass
class Job:
    """Represents a long-running pipeline task."""

    id: str
    job_type: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": self.metadata,
            "error": self.error,
        }


class _JobLogWriter(io.StringIO):
    """Captures stdout/stderr while preserving streaming semantics."""

    def __init__(self, manager: "JobManager", job: Job) -> None:
        super().__init__()
        self._manager = manager
        self._job = job
        self._buffer = ""

    def write(self, text: str) -> int:  # type: ignore[override]
        if not text:
            return 0
        with self._manager._lock:
            self._buffer += text
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                self._job.logs.append(line + "\n")
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        if not self._buffer:
            return
        with self._manager._lock:
            self._job.logs.append(self._buffer)
            self._buffer = ""


class JobManager:
    """Thread-based job runner with incremental log capture."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def get_logs(self, job_id: str, start: int = 0) -> Dict[str, Any]:
        job = self.get_job(job_id)
        with self._lock:
            total = len(job.logs)
            sliced = job.logs[start:]
        return {"lines": sliced, "next_index": start + len(sliced), "total": total}

    def create_job(
        self,
        job_type: str,
        target: JobTarget,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, job_type=job_type, metadata=metadata or {})
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job, target), daemon=True)
        thread.start()
        return job

    def update_metadata(self, job: Job, data: Dict[str, Any]) -> None:
        with self._lock:
            job.metadata.update(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_job(self, job: Job, target: JobTarget) -> None:
        job.started_at = time.time()
        job.status = "running"
        writer = _JobLogWriter(self, job)
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(writer))
            stack.enter_context(contextlib.redirect_stderr(writer))
            try:
                target(job)
            except Exception as exc:  # pragma: no cover - defensive guard
                traceback.print_exc()
                with self._lock:
                    job.status = "failed"
                    job.error = f"{exc.__class__.__name__}: {exc}"
                return
            finally:
                writer.flush()
        with self._lock:
            if job.status != "failed":
                job.status = "succeeded"
            job.finished_at = time.time()


job_manager = JobManager()
"""Global job manager instance used by the API."""
