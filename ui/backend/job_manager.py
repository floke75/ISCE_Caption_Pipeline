"""Background job orchestration used by the UI backend."""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import subprocess
import yaml

from .config_service import PipelineConfigService


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobStep:
    name: str
    command: List[str]
    cwd: Optional[Path] = None


@dataclass
class JobRecord:
    id: str
    job_type: str
    params: Dict[str, Any]
    created_at: float
    status: str = JobStatus.QUEUED
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: float = 0.0
    message: Optional[str] = None
    log_path: Path | None = None
    workspace: Path | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "jobType": self.job_type,
            "params": self.params,
            "createdAt": self.created_at,
            "status": self.status,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "progress": self.progress,
            "message": self.message,
            "extra": self.extra,
        }


class JobManager:
    """Manage lifecycle of pipeline jobs and surface their logs."""

    def __init__(
        self,
        base_dir: Path,
        config_service: PipelineConfigService,
        max_workers: int = 3,
    ) -> None:
        self._base_dir = base_dir
        self._config_service = config_service
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def read_log(self, job_id: str, tail: int = 1000) -> str:
        job = self.get_job(job_id)
        if not job or not job.log_path or not job.log_path.exists():
            return ""
        if tail <= 0:
            return job.log_path.read_text(encoding="utf-8", errors="ignore")
        # Read the tail efficiently
        with job.log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = 1024
            data = b""
            while size > 0 and data.count(b"\n") <= tail:
                delta = min(block, size)
                size -= delta
                fh.seek(size)
                data = fh.read(delta) + data
                if size == 0:
                    break
                fh.seek(size)
            return data.decode("utf-8", errors="ignore")

    def _register_job(self, job: JobRecord) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)

    def start_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        build_steps: callable[[Path, Dict[str, Any]], Iterable[JobStep]],
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        workspace = self._base_dir / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "job.log"
        job = JobRecord(
            id=job_id,
            job_type=job_type,
            params=params,
            created_at=time.time(),
            log_path=log_path,
            workspace=workspace,
        )
        self._register_job(job)

        def runner() -> None:
            self._update_job(job_id, status=JobStatus.RUNNING, started_at=time.time())
            try:
                job_config = self._config_service.build_job_config(
                    workspace=workspace,
                    overrides=config_overrides,
                )
                config_path = workspace / "pipeline_config.yaml"
                with config_path.open("w", encoding="utf-8") as fh:
                    fh.write("# Generated per-job pipeline config\n")
                    yaml.safe_dump(job_config, fh, allow_unicode=True, sort_keys=False)

                steps = list(build_steps(workspace, job_config))
                total = max(len(steps), 1)
                for index, step in enumerate(steps, start=1):
                    self._run_step(step, log_path)
                    self._update_job(job_id, progress=index / total)
                self._update_job(job_id, status=JobStatus.SUCCEEDED, finished_at=time.time(), progress=1.0)
            except subprocess.CalledProcessError as exc:
                message = f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}"
                self._append_log(log_path, message + "\n")
                self._update_job(job_id, status=JobStatus.FAILED, finished_at=time.time(), message=message)
            except Exception as exc:  # noqa: BLE001 - surface unexpected failures
                self._append_log(log_path, f"Unexpected error: {exc}\n")
                self._update_job(job_id, status=JobStatus.FAILED, finished_at=time.time(), message=str(exc))

        self._executor.submit(runner)
        return job

    def _append_log(self, log_path: Path, text: str) -> None:
        with self._log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(text)

    def _run_step(self, step: JobStep, log_path: Path) -> None:
        self._append_log(log_path, f"\n>>> STEP: {step.name}\n")
        process = subprocess.Popen(
            step.command,
            cwd=str(step.cwd or Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            self._append_log(log_path, line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, step.command)


__all__ = [
    "JobManager",
    "JobRecord",
    "JobStatus",
    "JobStep",
]
