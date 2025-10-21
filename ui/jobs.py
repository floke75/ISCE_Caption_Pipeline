"""Job management utilities for the web UI backend."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional


class JobStatus(str, Enum):
    """Enumerates the lifecycle states for a submitted job."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class JobRecord:
    """In-memory representation of a submitted job."""

    id: str
    kind: str
    params: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    logs: List[str] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the job record to a JSON-friendly dict."""

        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "params": self.params,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "result": self.result,
            "error": self.error,
            "log_lines": len(self.logs),
        }


class JobLogStream:
    """A writable stream object that mirrors output into the job record."""

    def __init__(self, store: "JobStore", job_id: str, mirror: bool = True):
        self._store = store
        self._job_id = job_id
        self._mirror = mirror

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._mirror:
            sys.__stdout__.write(text)
            sys.__stdout__.flush()
        self._store.append_log(self._job_id, text)
        return len(text)

    def flush(self) -> None:
        if self._mirror:
            sys.__stdout__.flush()


class JobStore:
    """Thread-safe registry for active and completed jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def create(self, job_id: str, kind: str, params: Dict[str, Any]) -> JobRecord:
        with self._lock:
            record = JobRecord(id=job_id, kind=kind, params=params)
            self._jobs[job_id] = record
            return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def set_status(self, job_id: str, status: JobStatus, error: Optional[str] = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.updated_at = datetime.utcnow()
            job.error = error

    def set_result(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.result = result
            job.updated_at = datetime.utcnow()

    def append_log(self, job_id: str, text: str) -> None:
        if not text:
            return
        lines = text.splitlines()
        if text.endswith("\n"):
            lines.append("")
        with self._lock:
            job = self._jobs[job_id]
            for line in lines:
                if line:
                    job.logs.append(line)
                else:
                    job.logs.append("")
            if len(job.logs) > 5000:
                # keep memory usage bounded
                job.logs = job.logs[-5000:]
            job.updated_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def log_stream(self, job_id: str, mirror: bool = True) -> JobLogStream:
        return JobLogStream(self, job_id, mirror=mirror)

    def export_logs(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs[job_id]
            return "\n".join(job.logs)
