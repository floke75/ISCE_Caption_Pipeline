from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pytest

from ui.backend.app import _log_event_stream, job_manager
from ui.backend.job_manager import JobRecord


@pytest.fixture()
def temp_job_workspace(tmp_path: Path):
    job_id = f'test-log-stream-{tmp_path.name}'
    workspace = job_manager._jobs_root / job_id  # type: ignore[attr-defined]
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace / 'job.log'
    log_path.touch()
    record = JobRecord(
        id=job_id,
        job_type='inference',
        status='running',
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        params={},
        workspace=workspace,
        log_path=log_path,
    )
    record.write_metadata()
    with job_manager._lock:  # type: ignore[attr-defined]
        job_manager._jobs[job_id] = record  # type: ignore[attr-defined]
    try:
        yield job_id, workspace, log_path
    finally:
        with job_manager._lock:  # type: ignore[attr-defined]
            job_manager._jobs.pop(job_id, None)  # type: ignore[attr-defined]
        shutil.rmtree(workspace, ignore_errors=True)


def test_log_event_stream_yields_updates_and_completion(temp_job_workspace):
    job_id, _workspace, log_path = temp_job_workspace
    stream = _log_event_stream(job_id, poll_interval=0.01)

    first_event = next(stream)
    assert ': heartbeat' in first_event

    with log_path.open('a', encoding='utf-8') as fh:
        fh.write('line one\n')
        fh.flush()

    chunk_event = next(stream)
    assert 'data: line one' in chunk_event

    job_manager.finalize_job(job_id, status='succeeded')

    completion_event = next(stream)
    assert 'event: complete' in completion_event
    assert 'data: succeeded' in completion_event

    with pytest.raises(StopIteration):
        next(stream)
