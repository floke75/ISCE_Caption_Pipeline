import sys
import unittest
import shutil
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock
import os

# Add repo root
sys.path.append(os.getcwd())

from ui.backend.job_manager import JobManager, JobRecord
from ui.backend.config_service import ConfigService

class TestJobRecovery(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_root = Path(self.test_dir)
        self.jobs_root = self.storage_root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.mock_config_service = MagicMock(spec=ConfigService)
        self.mock_config_service.base_config.return_value = {}
        self.mock_config_service.stored_overrides.return_value = {}
        self.mock_config_service.resolve_paths.return_value = {}

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_job_metadata(self, job_id, status):
        job_dir = self.jobs_root / job_id
        job_dir.mkdir()
        metadata = {
            "id": job_id,
            "job_type": "test",
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "params": {},
            "progress": 0.5,
            "message": "Running..."
        }
        with open(job_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        return job_dir

    def test_recover_running_job(self):
        # Create a "running" job on disk
        self.create_job_metadata("job1", "running")

        # Initialize JobManager
        manager = JobManager(self.storage_root, self.mock_config_service)

        # Check status
        job = manager.get_job("job1")
        # Expect failure (will fail initially as I haven't fixed it yet)
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "System restarted while job was active")
        self.assertEqual(job.message, "Interrupted")

        # Verify it was persisted to disk
        with open(self.jobs_root / "job1" / "metadata.json", "r") as f:
            data = json.load(f)
            self.assertEqual(data["status"], "failed")

    def test_recover_pending_job(self):
        self.create_job_metadata("job2", "pending")
        manager = JobManager(self.storage_root, self.mock_config_service)
        job = manager.get_job("job2")
        self.assertEqual(job.status, "failed")
        self.assertEqual(job.error, "System restarted while job was active")

    def test_recover_completed_job(self):
        self.create_job_metadata("job3", "succeeded")
        manager = JobManager(self.storage_root, self.mock_config_service)
        job = manager.get_job("job3")
        self.assertEqual(job.status, "succeeded")
        # The stored job has no error field in create_job_metadata default?
        # create_job_metadata doesn't set error explicitly in dict (so None/missing).
        # JobRecord.from_file defaults error to None.
        self.assertIsNone(job.error)

if __name__ == '__main__':
    unittest.main()
