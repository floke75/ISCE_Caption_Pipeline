import asyncio
import json
import logging
import uuid
import sys
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

# Add project root to path to allow imports of pipeline scripts
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import pipeline functions
from run_pipeline import DEFAULT_SETTINGS, process_inference_file, process_training_file
from scripts.train_model import run_training
from pipeline_config import load_pipeline_config


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration and Paths ---
UI_DATA_ROOT = REPO_ROOT / "ui_data"
JOBS_DIR = UI_DATA_ROOT / "jobs"
JOBS_METADATA_FILE = UI_DATA_ROOT / "jobs.json"

# Ensure directories exist
UI_DATA_ROOT.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

# --- Enums and Models ---
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobType(str, Enum):
    INFERENCE = "inference"
    TRAINING_PAIR = "training_pair"
    MODEL_TRAINING = "model_training"

class Job(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None

# --- Request Models ---
class InferenceRequest(BaseModel):
    media_path: str
    transcript_path: Optional[str] = None

class TrainingPairRequest(BaseModel):
    media_path: str
    srt_path: str

class ModelTrainingRequest(BaseModel):
    corpus_dir: str
    constraints_path: str
    weights_path: str

# --- Job Manager ---
class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._execution_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._load_jobs_from_disk()

    def _load_pipeline_config(self) -> Dict[str, Any]:
        """Load the latest pipeline configuration for each job execution."""
        return load_pipeline_config(DEFAULT_SETTINGS, str(REPO_ROOT / "pipeline_config.yaml"))

    def _load_jobs_from_disk(self):
        if JOBS_METADATA_FILE.exists():
            with open(JOBS_METADATA_FILE, "r") as f:
                jobs_data = json.load(f)
                for job_id, job_data in jobs_data.items():
                    try:
                        self._jobs[job_id] = Job(**job_data)
                    except ValueError as exc:
                        logging.error("Failed to load job %s from disk: %s", job_id, exc)

    def _save_jobs_to_disk(self):
        serialized_jobs = {
            job_id: jsonable_encoder(job, by_alias=True, exclude_none=True)
            for job_id, job in self._jobs.items()
        }
        with open(JOBS_METADATA_FILE, "w") as f:
            json.dump(serialized_jobs, f, indent=4)

    def _update_job_status(self, job_id: str, status: JobStatus, error: Optional[str] = None, result: Optional[Dict[str, Any]] = None):
        job = self._jobs[job_id]
        job.status = status
        if status == JobStatus.RUNNING:
            job.started_at = datetime.utcnow()
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            job.completed_at = datetime.utcnow()
        if error:
            job.error = error
        if result:
            job.result = result
        self._save_jobs_to_disk()

    def _run_job_safely(self, job: Job, target_fn, *args, **kwargs):
        job_dir = JOBS_DIR / job.id
        job_dir.mkdir(exist_ok=True)
        log_path = self.get_job_log_path(job.id)

        with self._execution_lock:
            try:
                self._update_job_status(job.id, JobStatus.RUNNING)
                with open(log_path, "w") as log_file, redirect_stdout(log_file):
                    print(f"Starting job {job.id} of type {job.type.value}")
                    result = target_fn(*args, **kwargs)
                    print(f"Job {job.id} completed successfully.")
                self._update_job_status(job.id, JobStatus.COMPLETED, result=result)
            except Exception as e:
                logging.error(f"Job {job.id} failed: {e}", exc_info=True)
                with open(log_path, "a") as log_file:
                    log_file.write(f"\n\n--- JOB FAILED ---\n{e}")
                self._update_job_status(job.id, JobStatus.FAILED, error=str(e))


    def create_job(self, job_type: JobType, parameters: Dict[str, Any]) -> Job:
        job = Job(type=job_type, parameters=parameters)
        self._jobs[job.id] = job
        self._save_jobs_to_disk()

        target_fn = None
        args = []
        kwargs = {}

        if job.type == JobType.INFERENCE:
            target_fn = process_inference_file
            cfg = self._load_pipeline_config()
            media_path = Path(parameters["media_path"])
            transcript_path = parameters.get("transcript_path")
            if transcript_path:
                transcript_file = Path(transcript_path)
                if transcript_file.exists():
                    destination_dir = Path(cfg["txt_placement_folder"])
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    destination_path = destination_dir / f"{media_path.stem}.txt"
                    if transcript_file.resolve() != destination_path.resolve():
                        shutil.copy2(transcript_file, destination_path)
                else:
                    logging.warning("Transcript path %s does not exist; running inference without external transcript.", transcript_file)
            args = [media_path, cfg]

        elif job.type == JobType.TRAINING_PAIR:
            target_fn = process_training_file
            cfg = self._load_pipeline_config()
            args = [Path(parameters["media_path"]), Path(parameters["srt_path"]), cfg]

        elif job.type == JobType.MODEL_TRAINING:
            target_fn = run_training
            args = [parameters["corpus_dir"], parameters["constraints_path"], parameters["weights_path"]]

        if target_fn:
            self._executor.submit(self._run_job_safely, job, target_fn, *args, **kwargs)

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get_job_log_path(self, job_id: str) -> Path:
        return JOBS_DIR / job_id / "job.log"

job_manager = JobManager()
app = FastAPI(title="ISCE Caption Pipeline UI")

# --- API Endpoints ---
@app.post("/api/jobs/inference", response_model=Job, status_code=202)
async def create_inference_job(request: InferenceRequest):
    return job_manager.create_job(JobType.INFERENCE, request.dict())

@app.post("/api/jobs/training-pair", response_model=Job, status_code=202)
async def create_training_pair_job(request: TrainingPairRequest):
    return job_manager.create_job(JobType.TRAINING_PAIR, request.dict())

@app.post("/api/jobs/model-training", response_model=Job, status_code=202)
async def create_model_training_job(request: ModelTrainingRequest):
    return job_manager.create_job(JobType.MODEL_TRAINING, request.dict())

@app.get("/api/jobs", response_model=List[Job])
async def list_jobs():
    return job_manager.get_all_jobs()

@app.get("/api/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    log_path = job_manager.get_job_log_path(job_id)
    if not job_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    if not log_path.exists():
        return {"logs": ""}
    return {"logs": log_path.read_text()}

import yaml

@app.get("/api/config/pipeline")
async def get_pipeline_config():
    config_path = REPO_ROOT / "pipeline_config.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="pipeline_config.yaml not found")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@app.get("/api/config/model")
async def get_model_config():
    config_path = REPO_ROOT / "config.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config.yaml not found")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@app.post("/api/config/pipeline")
async def save_pipeline_config(config: Dict[str, Any]):
    config_path = REPO_ROOT / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return {"status": "success"}

@app.post("/api/config/model")
async def save_model_config(config: Dict[str, Any]):
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return {"status": "success"}

@app.get("/api/files")
async def browse_files(path: Optional[str] = None):
    base_path = (REPO_ROOT / (path or "")).resolve()
    if not base_path.is_relative_to(REPO_ROOT):
        raise HTTPException(status_code=403, detail="Access denied")

    if not base_path.exists() or not base_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid or inaccessible path")

    # ..for parent directory
    parent = {"name": "..", "path": str(base_path.parent), "is_dir": True} if base_path != REPO_ROOT else None

    items = []
    for item in sorted(base_path.iterdir()):
        try:
            items.append({"name": item.name, "path": str(item), "is_dir": item.is_dir()})
        except OSError:
            continue # Skip inaccessible files/links

    return {"path": str(base_path), "parent": parent, "items": items}

# --- Static Files ---
def _mount_frontend(app: FastAPI):
    dist_dir = REPO_ROOT / "ui" / "frontend" / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui-frontend")

_mount_frontend(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
