from __future__ import annotations

import enum
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    params: Dict[str, Any]
    payload: Dict[str, Any] = field(default_factory=dict, repr=False)
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: float = 0.0
    stage: Optional[str] = None
    message: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[Path] = None
    workspace: Optional[Path] = None
    queue_position: Optional[int] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "id": self.job_id,
            "type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() + "Z",
            "started_at": self.started_at.isoformat() + "Z" if self.started_at else None,
            "finished_at": self.finished_at.isoformat() + "Z" if self.finished_at else None,
            "progress": round(self.progress, 4),
            "stage": self.stage,
            "message": self.message,
            "artifacts": self.artifacts,
            "params": self.params,
            "result": self.result,
        }
        if self.queue_position is not None:
            data["queue_position"] = self.queue_position
        data["workspace_path"] = str(self.workspace) if self.workspace else None
        return data

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "id": self.job_id,
            "type": self.job_type,
            "params": self.params,
            "payload": self.payload,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "artifacts": self.artifacts,
            "result": self.result,
        }

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "JobRecord":
        created_at = _parse_datetime(metadata.get("created_at"))
        started_at = _parse_datetime(metadata.get("started_at"))
        finished_at = _parse_datetime(metadata.get("finished_at"))
        record = cls(
            job_id=metadata["id"],
            job_type=metadata["type"],
            params=metadata.get("params", {}),
            payload=metadata.get("payload", {}),
            status=JobStatus(metadata.get("status", JobStatus.PENDING.value)),
            created_at=created_at or datetime.utcnow(),
            started_at=started_at,
            finished_at=finished_at,
            progress=float(metadata.get("progress", 0.0)),
            stage=metadata.get("stage"),
            message=metadata.get("message"),
            artifacts=list(metadata.get("artifacts", [])),
            result=dict(metadata.get("result", {})),
        )
        return record

    @property
    def metadata_path(self) -> Path:
        if not self.workspace:
            raise RuntimeError("JobRecord.workspace is not set")
        return self.workspace / "job.json"


class JobContext:
    def __init__(
        self,
        manager: "JobManager",
        record: JobRecord,
        log_handle,
    ) -> None:
        self._manager = manager
        self._record = record
        self._log_handle = log_handle
        self.workspace = record.workspace or manager.base_dir / record.job_id
        self.runtime_dir = self.workspace / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        self._log_handle.write(f"[{datetime.utcnow().isoformat()}Z] {message}\n")
        self._log_handle.flush()

    def write_output(self, text: str) -> None:
        self._log_handle.write(text)
        if not text.endswith("\n"):
            self._log_handle.write("\n")
        self._log_handle.flush()

    def set_stage(self, stage: str) -> None:
        self._manager._update_record(self._record.job_id, stage=stage)

    def set_progress(self, progress: float) -> None:
        self._manager._update_record(self._record.job_id, progress=max(0.0, min(progress, 1.0)))

    def add_artifact(self, name: str, path: Path) -> None:
        artifact = {"name": name, "path": str(path)}
        with self._manager._lock:
            self._record.artifacts.append(artifact)
            self._manager._persist_record(self._record)

    def set_result(self, **kwargs: Any) -> None:
        with self._manager._lock:
            self._record.result.update(kwargs)
            self._manager._persist_record(self._record)

    def run_command(
        self,
        command: Iterable[Any],
        *,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        stage: Optional[str] = None,
        progress_range: Optional[tuple[float, float]] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        import subprocess

        from shlex import quote

        if stage:
            self.set_stage(stage)
        if progress_range:
            self.set_progress(progress_range[0])

        command_list = [str(part) for part in command]
        self.log(f"$ {' '.join(quote(part) for part in command_list)}")

        process = subprocess.Popen(
            command_list,
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
        for line in process.stdout:
            self.write_output(line.rstrip("\n"))
            if on_output:
                on_output(line)

        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(f"Command {' '.join(command_list)} failed with exit code {exit_code}")

        if progress_range:
            self.set_progress(progress_range[1])


class JobManager:
    def __init__(
        self,
        base_dir: Path,
        *,
        max_workers: int = 2,
        queue_limit: int = 10,
    ) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._handlers: Dict[str, Callable[[JobContext, Dict[str, Any]], None]] = {}
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=queue_limit)
        self._pending_order: List[str] = []
        self._resume_candidates: set[str] = set()
        self._max_workers = max(1, max_workers)
        self._workers: List[threading.Thread] = []

        self._load_existing_jobs()
        self._start_workers()

    def register_handler(
        self, job_type: str, handler: Callable[[JobContext, Dict[str, Any]], None]
    ) -> None:
        with self._lock:
            self._handlers[job_type] = handler
            resume = [
                job_id
                for job_id in list(self._resume_candidates)
                if self._jobs[job_id].job_type == job_type
            ]
            for job_id in resume:
                self._enqueue(job_id)
                self._resume_candidates.discard(job_id)

    def start_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> JobRecord:
        with self._lock:
            if job_type not in self._handlers:
                raise ValueError(f"No handler registered for job type '{job_type}'")

        job_id = uuid.uuid4().hex
        workspace = self.base_dir / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "job.log"
        record = JobRecord(job_id=job_id, job_type=job_type, params=params, payload=payload)
        record.log_path = log_path
        record.workspace = workspace

        with self._lock:
            if self._queue.full():
                raise queue.Full("Job queue is full")
            self._jobs[job_id] = record
            self._persist_record(record)
            self._enqueue(job_id)
        return record

    def _start_workers(self) -> None:
        for _ in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    if job_id in self._pending_order:
                        self._pending_order.remove(job_id)
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            if record.status in {JobStatus.CANCELLED, JobStatus.COMPLETED, JobStatus.FAILED}:
                record.queue_position = None
                self._persist_record(record)
                return
            handler = self._handlers.get(record.job_type)
            if handler is None:
                record.status = JobStatus.FAILED
                record.message = "No task handler available"
                record.finished_at = datetime.utcnow()
                record.queue_position = None
                self._persist_record(record)
                return
            record.status = JobStatus.RUNNING
            record.started_at = datetime.utcnow()
            record.queue_position = None
            self._persist_record(record)

        if record.log_path is None:
            record.log_path = record.workspace / "job.log" if record.workspace else None

        log_path = record.log_path or (record.workspace / "job.log" if record.workspace else None)
        if log_path is None:
            raise RuntimeError("Job workspace is not configured")

        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as log_handle:
            context = JobContext(self, record, log_handle)
            try:
                handler(context, record.payload)
            except Exception as exc:  # noqa: BLE001
                context.log(f"[ERROR] {exc}")
                with self._lock:
                    record.status = JobStatus.FAILED
                    record.message = str(exc)
                    record.finished_at = datetime.utcnow()
                    record.queue_position = None
                    self._persist_record(record)
                return

        with self._lock:
            record.status = JobStatus.COMPLETED
            record.finished_at = datetime.utcnow()
            record.progress = 1.0
            record.queue_position = None
            self._persist_record(record)

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            if record.status != JobStatus.PENDING:
                raise ValueError("Job is no longer pending and cannot be cancelled")
            record.status = JobStatus.CANCELLED
            record.finished_at = datetime.utcnow()
            record.message = "Cancelled by user"
            record.queue_position = None
            if job_id in self._pending_order:
                self._pending_order.remove(job_id)
            self._persist_record(record)
            return record

    def list_jobs(self) -> List[JobRecord]:
        with self._lock:
            queue_positions = {
                job_id: index
                for index, job_id in enumerate(self._pending_order, start=1)
            }
            for record in self._jobs.values():
                record.queue_position = queue_positions.get(record.job_id)
            return list(self._jobs.values())

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            queue_positions = {
                queued_id: index
                for index, queued_id in enumerate(self._pending_order, start=1)
            }
            record.queue_position = queue_positions.get(job_id)
            return record

    def _update_record(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(record, key, value)
            self._persist_record(record)

    def read_log(self, job_id: str, offset: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
        record = self.get(job_id)
        if not record.log_path or not record.log_path.exists():
            return {"content": "", "offset": offset, "complete": record.status in {JobStatus.COMPLETED, JobStatus.FAILED}}

        path = record.log_path
        file_size = path.stat().st_size
        if offset < 0:
            offset = 0
        if offset > file_size:
            offset = file_size

        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(limit if limit is not None else -1)
        text = data.decode("utf-8", errors="ignore")
        new_offset = offset + len(data)
        complete = record.status in {JobStatus.COMPLETED, JobStatus.FAILED} and new_offset >= file_size
        return {"content": text, "offset": new_offset, "complete": complete}

    def _enqueue(self, job_id: str) -> None:
        with self._lock:
            if job_id not in self._pending_order:
                self._pending_order.append(job_id)
        self._queue.put(job_id)

    def _load_existing_jobs(self) -> None:
        for workspace in sorted(self.base_dir.glob("*/")):
            metadata_path = workspace / "job.json"
            if not metadata_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            record = JobRecord.from_metadata(metadata)
            record.workspace = workspace
            record.log_path = workspace / "job.log"
            if record.status == JobStatus.RUNNING:
                record.status = JobStatus.FAILED
                record.message = "Job interrupted during server restart"
                record.finished_at = datetime.utcnow()
            with self._lock:
                self._jobs[record.job_id] = record
                self._persist_record(record)
                if record.status == JobStatus.PENDING:
                    self._resume_candidates.add(record.job_id)

    def _persist_record(self, record: JobRecord) -> None:
        if record.workspace is None:
            return
        record.workspace.mkdir(parents=True, exist_ok=True)
        metadata_path = record.metadata_path
        metadata_path.write_text(
            json.dumps(record.to_metadata(), indent=2, sort_keys=False),
            encoding="utf-8",
        )

    def queue_size(self) -> int:
        return self._queue.qsize()


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
