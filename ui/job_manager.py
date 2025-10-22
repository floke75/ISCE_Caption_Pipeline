from __future__ import annotations

import enum
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

    def to_dict(self) -> Dict[str, Any]:
        return {
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

    def set_result(self, **kwargs: Any) -> None:
        with self._manager._lock:
            self._record.result.update(kwargs)

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
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def start_job(
        self,
        job_type: str,
        params: Dict[str, Any],
        task: Callable[[JobContext], None],
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        workspace = self.base_dir / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "job.log"
        record = JobRecord(job_id=job_id, job_type=job_type, params=params)
        record.log_path = log_path
        record.workspace = workspace
        with self._lock:
            self._jobs[job_id] = record

        thread = threading.Thread(target=self._run_job, args=(record, task), daemon=True)
        thread.start()
        return record

    def _run_job(self, record: JobRecord, task: Callable[[JobContext], None]) -> None:
        record.started_at = datetime.utcnow()
        record.status = JobStatus.RUNNING
        with record.log_path.open("a", encoding="utf-8") as log_handle:
            context = JobContext(self, record, log_handle)
            try:
                task(context)
            except Exception as exc:  # noqa: BLE001
                context.log(f"[ERROR] {exc}")
                with self._lock:
                    record.status = JobStatus.FAILED
                    record.message = str(exc)
                    record.finished_at = datetime.utcnow()
                return

        with self._lock:
            record.status = JobStatus.COMPLETED
            record.finished_at = datetime.utcnow()
            record.progress = 1.0

    def list_jobs(self) -> List[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def _update_record(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in kwargs.items():
                setattr(record, key, value)

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
