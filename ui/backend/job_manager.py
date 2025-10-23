"""Service for managing and executing long-running background jobs.

This module provides the `JobManager`, a service responsible for the entire
lifecycle of background tasks, such as running inference or training a model.
It is designed to be thread-safe and robust, handling job creation, queuing,
execution, cancellation, and state persistence.

Key components:
-   **JobManager**: The main class that orchestrates all jobs. It uses a
    `ThreadPoolExecutor` to run jobs concurrently and a semaphore to limit
    the number of active workers. It persists job metadata to disk, allowing
    the state to be recovered after a server restart.
-   **JobRecord**: A dataclass that represents the state of a single job,
    including its ID, status, parameters, and results. This object is
    serialized to a `metadata.json` file in the job's dedicated workspace.
-   **JobContext**: An object passed to each job runner function. It provides a
    safe and controlled interface for the running job to communicate back to
    the manager, allowing it to log messages, update its progress, check for
    cancellation requests, and access runtime configuration.
-   **Exceptions**: Custom exceptions like `JobCancelled` and `JobQueueFull`
    are defined to handle specific flow control and error conditions.
"""
from __future__ import annotations

import json
import shutil
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


class JobCancelled(Exception):
    """Raised when a running job should stop due to cancellation."""


class JobCancellationError(RuntimeError):
    """Raised when a cancellation request cannot be fulfilled."""


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

    def _ensure_not_cancelled(self) -> None:
        if self._manager.is_cancelled(self.record.id):
            raise JobCancelled()

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
        stdout = process.stdout

        def _terminate_due_to_cancel() -> None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                process.kill()
                process.wait()
            with self._log_lock:
                self._log_file.flush()

        import selectors

        selector = selectors.DefaultSelector()
        selector.register(stdout, selectors.EVENT_READ)
        try:
            while True:
                if self._manager.is_cancelled(self.record.id):
                    _terminate_due_to_cancel()
                    raise JobCancelled()

                events = selector.select(timeout=0.2)
                if not events:
                    if process.poll() is not None:
                        break
                    continue

                line = stdout.readline()
                if line:
                    with self._log_lock:
                        self._log_file.write(line)
                        self._log_file.flush()
                    continue

                if process.poll() is not None:
                    break

            # Drain any buffered output after the process exits.
            while True:
                remaining = stdout.readline()
                if not remaining:
                    break
                with self._log_lock:
                    self._log_file.write(remaining)
                    self._log_file.flush()
        finally:
            try:
                selector.unregister(stdout)
            except KeyError:  # pragma: no cover - defensive
                pass
            selector.close()

        stdout.close()
        process.wait()
        if self._manager.is_cancelled(self.record.id):
            raise JobCancelled()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {display}")
        with self._log_lock:
            self._log_file.flush()

    def update(self, *, progress: Optional[float] = None, message: Optional[str] = None) -> None:
        self._ensure_not_cancelled()
        self._manager.update_job(self.record.id, progress=progress, message=message)

    def effective_config(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._ensure_not_cancelled()
        return self._manager.prepare_runtime_config(self.record, overrides or {})

    def segmentation_config(self, overrides: Optional[Dict[str, Any]] = None) -> Path:
        self._ensure_not_cancelled()
        return self._manager.prepare_segmentation_config(self.record, overrides or {})

    def finalize(self, status: JobStatus, *, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        self._manager.finalize_job(self.record.id, status=status, result=result, error=error)

    def close(self) -> None:
        with self._log_lock:
            self._log_file.close()

    def __enter__(self) -> "JobContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class JobQueueFull(Exception):
    """Raised when the job queue is at capacity."""


class JobManager:
    """Coordinates the lifecycle of background pipeline jobs.

    This thread-safe class is the central component for managing jobs. It handles
    queuing, execution via a thread pool, state tracking, and persistence.
    It ensures that the number of concurrent jobs does not exceed a defined
    limit and manages a queue for pending jobs.

    Attributes:
        _storage_root: The root directory for all persistent job data.
        _config_service: The main `ConfigService` instance.
        _segmentation_service: The `ConfigService` for the segmentation model.
        _jobs: An in-memory dictionary of all known `JobRecord` objects.
        _futures: A dictionary mapping job IDs to `Future` objects from the executor.
        _lock: A re-entrant lock to protect shared state (`_jobs`, `_futures`).
        _executor: A `ThreadPoolExecutor` for running jobs in the background.
        _queue_limit: The maximum number of jobs allowed to be queued.
        _concurrency_guard: A semaphore to limit the number of active workers.
        _jobs_root: The specific directory where job workspaces are created.
        _cancel_events: A dictionary mapping job IDs to cancellation `Event` objects.
    """

    def __init__(
        self,
        storage_root: Path,
        config_service: ConfigService,
        *,
        segmentation_config_service: Optional[ConfigService] = None,
        max_workers: int = 3,
        queue_limit: Optional[int] = None,
    ) -> None:
        self._storage_root = storage_root
        self._config_service = config_service
        self._segmentation_service = segmentation_config_service
        self._jobs: Dict[str, JobRecord] = {}
        self._futures: Dict[str, Future[Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue_limit = queue_limit if queue_limit is not None else max_workers * 4
        if self._queue_limit < 1:
            raise ValueError("queue_limit must be at least 1")
        self._concurrency_guard = threading.BoundedSemaphore(max_workers)
        self._jobs_root = storage_root / "jobs"
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._cancel_events: Dict[str, threading.Event] = {}
        self._load_existing_jobs()

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------
    def create_job(self, job_type: str, params: Dict[str, Any], runner) -> JobRecord:
        """Creates, persists, and queues a new job for execution.

        This method handles the initial creation of a job. It generates a unique
        ID, creates a dedicated workspace on disk, writes the initial metadata,
        and submits the job to the thread pool for execution.

        Args:
            job_type: A string identifying the type of job (e.g., "inference").
            params: A dictionary of parameters for the job.
            runner: The function that will be executed to run the job.

        Returns:
            The newly created `JobRecord`.

        Raises:
            JobQueueFull: If the number of currently queued jobs has reached
                          the configured limit.
        """
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
        self._cancel_events[job_id] = threading.Event()
        with self._lock:
            self._jobs[job_id] = record
            active = sum(1 for future in self._futures.values() if not future.done())
            if active >= self._queue_limit:
                del self._jobs[job_id]
                shutil.rmtree(workspace, ignore_errors=True)
                del self._cancel_events[job_id]
                raise JobQueueFull("Job queue is at capacity; try again later.")
            future = self._executor.submit(self._run_job, record, runner)
            self._futures[job_id] = future
            future.add_done_callback(lambda _: self._cleanup_future(job_id))
        return record

    def _run_job(self, record: JobRecord, runner) -> None:
        self._concurrency_guard.acquire()
        try:
            with JobContext(self, record) as ctx:
                self.update_job(record.id, status="running", progress=0.02, message="Starting")
                try:
                    runner(ctx)
                    with self._lock:
                        current = self._jobs[record.id]
                        if current.status == "running":
                            ctx.finalize("succeeded")
                except JobCancelled:
                    self._append_log(record, "Job cancelled by user")
                    self.finalize_job(record.id, status="cancelled")
                    self.update_job(record.id, message="Cancelled")
                except Exception as exc:  # pragma: no cover - defensive
                    ctx.log(f"Job failed: {exc}")
                    ctx.finalize("failed", error=str(exc))
                    raise
        finally:
            self._concurrency_guard.release()

    def _cleanup_future(self, job_id: str) -> None:
        with self._lock:
            future = self._futures.get(job_id)
            if future and future.done():
                del self._futures[job_id]

    def _append_log(self, record: JobRecord, message: str) -> None:
        timestamp = datetime.utcnow().isoformat()
        record.log_path.parent.mkdir(parents=True, exist_ok=True)
        with record.log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        """Updates the state of a running job.

        This method is called by a `JobContext` to report progress back to the
        manager. It updates the in-memory `JobRecord` and writes the new state
        to disk.

        Args:
            job_id: The ID of the job to update.
            status: The new status of the job.
            progress: The new progress value (0.0 to 1.0).
            message: The new status message.
        """
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
        """Marks a job as complete, setting its final state.

        This is called by the `JobContext` when a job finishes, either by
        succeeding, failing, or being cancelled.

        Args:
            job_id: The ID of the job to finalize.
            status: The final status ('succeeded', 'failed', 'cancelled').
            result: A dictionary of final results, if successful.
            error: An error message, if failed.
        """
        with self._lock:
            record = self._jobs[job_id]
            record.status = status
            record.result = result
            record.error = error
            record.progress = 1.0 if status == "succeeded" else record.progress
            record.updated_at = datetime.utcnow()
            record.write_metadata()
            if status in {"succeeded", "failed", "cancelled"}:
                self._cancel_events.pop(job_id, None)

    def is_cancelled(self, job_id: str) -> bool:
        """Checks if a cancellation has been requested for a job.

        Args:
            job_id: The ID of the job to check.

        Returns:
            True if cancellation has been requested, False otherwise.
        """
        event = self._cancel_events.get(job_id)
        return event.is_set() if event else False

    def cancel_job(self, job_id: str) -> JobRecord:
        """Requests the cancellation of a job.

        This sets a cancellation event for the specified job, which the running
        job can check via its `JobContext`.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            The job record with an updated "Cancellation requested" message.

        Raises:
            KeyError: If the job ID does not exist.
            JobCancellationError: If the job has already completed.
        """
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            if record.status in {"succeeded", "failed", "cancelled"}:
                raise JobCancellationError(f"Job already {record.status}")
            cancel_event = self._cancel_events.setdefault(job_id, threading.Event())
            cancel_event.set()
            record.message = "Cancellation requested"
            record.updated_at = datetime.utcnow()
            record.write_metadata()
            self._append_log(record, "Cancellation requested by user")
            future = self._futures.get(job_id)
        if future and future.cancel():
            self.finalize_job(job_id, status="cancelled")
            self.update_job(job_id, message="Cancelled")
            return self.get_job(job_id)
        return self.get_job(job_id)

    def prepare_runtime_config(self, record: JobRecord, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the effective `pipeline_config.yaml` for a specific job run.

        This critical method constructs a job-specific configuration by layering
        global overrides, job-specific overrides, and injecting dynamic paths
        (like the job's workspace) into the configuration. The resulting config
        is written to the job's workspace and returned as a dictionary.

        Args:
            record: The `JobRecord` for the current job.
            overrides: A dictionary of overrides specific to this job run.

        Returns:
            The fully resolved, job-specific runtime configuration dictionary.
        """
        base = self._config_service.base_config()
        stored_overrides = self._config_service.stored_overrides()
        merged = _recursive_update(base, stored_overrides)
        resolved = self._config_service.resolve_paths(merged)

        resolved["project_root"] = str(self._storage_root.parent)
        pipeline_root = record.workspace / "pipeline"
        resolved["pipeline_root"] = str(pipeline_root)

        runtime = _recursive_update(resolved, overrides)
        runtime = self._config_service.resolve_paths(runtime)

        pipeline_root.mkdir(parents=True, exist_ok=True)
        self._write_runtime_config(record.workspace, runtime)
        runtime["__path__"] = str(record.workspace / "pipeline_config.runtime.yaml")

        if self._segmentation_service is not None:
            segmentation_path = self.prepare_segmentation_config(record)
            runtime["segmentation_config_path"] = str(segmentation_path)

        return runtime

    def _write_runtime_config(self, workspace: Path, config: Dict[str, Any]) -> None:
        config_path = workspace / "pipeline_config.runtime.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)

    def prepare_segmentation_config(
        self, record: JobRecord, overrides: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Prepares the effective `config.yaml` for a specific job run.

        Similar to `prepare_runtime_config`, but for the segmentation model's
        configuration. It writes a job-specific `segmentation_config.runtime.yaml`
        to the job's workspace.

        Args:
            record: The `JobRecord` for the current job.
            overrides: A dictionary of job-specific segmentation config overrides.

        Returns:
            The `Path` to the generated runtime segmentation config file.

        Raises:
            RuntimeError: If the segmentation config service is not configured.
        """
        if self._segmentation_service is None:
            raise RuntimeError("Segmentation config service is not configured")

        base = self._segmentation_service.base_config()
        stored_overrides = self._segmentation_service.stored_overrides()
        merged = _recursive_update(base, stored_overrides)
        runtime = _recursive_update(merged, overrides or {})
        runtime = self._segmentation_service.resolve_paths(runtime)

        config_path = record.workspace / "segmentation_config.runtime.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(runtime, fh, allow_unicode=True, sort_keys=False)
        return config_path

    # ------------------------------------------------------------------
    # Query APIs
    # ------------------------------------------------------------------
    def list_jobs(self) -> List[JobRecord]:
        """Returns a list of all known jobs, sorted by creation date.

        Returns:
            A list of `JobRecord` objects.
        """
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get_job(self, job_id: str) -> JobRecord:
        """Retrieves a single job by its ID.

        Args:
            job_id: The ID of the job to retrieve.

        Returns:
            The corresponding `JobRecord`.

        Raises:
            KeyError: If no job with that ID is found.
        """
        with self._lock:
            return self._jobs[job_id]

    def get_log_tail(self, job_id: str, limit: int = 4000) -> str:
        """Reads the last N characters of a job's log file.

        Args:
            job_id: The ID of the job whose log to read.
            limit: The maximum number of characters to read from the end.

        Returns:
            The tail of the log file as a string.
        """
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
            self._cancel_events.setdefault(record.id, threading.Event())


