"""Web UI backend for managing ISCE pipeline runs.

This module exposes a FastAPI application that lets non-technical users
launch inference and training jobs, edit configuration files, and stream
job logs for progress feedback.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline_config import load_pipeline_config
from run_pipeline import (
    DEFAULT_SETTINGS as PIPELINE_DEFAULTS,
    process_inference_file,
    process_training_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RUNTIME_ROOT = PROJECT_ROOT / "ui" / "runtime"
LOG_ROOT = PROJECT_ROOT / "ui" / "logs"

RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Utility models
# ---------------------------------------------------------------------------
class FlexibleModel(BaseModel):
    """Base model that accepts unknown keys and preserves them."""

    class Config:
        extra = "allow"


class AlignMakeSettings(FlexibleModel):
    out_root: Optional[str] = None
    cache_dir: Optional[str] = None
    whisper_model_id: Optional[str] = None
    align_model_id: Optional[str] = None
    language: Optional[str] = None
    compute_type: Optional[str] = None
    batch_size: Optional[int] = None
    hf_token: Optional[str] = None
    do_diarization: Optional[bool] = None
    diar_min_spk: Optional[int] = Field(None, alias="diar_min_spk")
    diar_max_spk: Optional[int] = Field(None, alias="diar_max_spk")
    skip_if_asr_exists: Optional[bool] = None


class BuildPairSettings(FlexibleModel):
    out_training_dir: Optional[str] = None
    out_inference_dir: Optional[str] = None
    language: Optional[str] = None
    time_tolerance_s: Optional[float] = None
    round_seconds: Optional[int] = None
    spacy_enable: Optional[bool] = None
    spacy_model: Optional[str] = None
    spacy_add_dependencies: Optional[bool] = None
    emit_asr_style_training_copy: Optional[bool] = None
    txt_match_close: Optional[float] = None
    txt_match_weak: Optional[float] = None


class OrchestratorSettings(FlexibleModel):
    poll_interval_seconds: Optional[int] = None
    file_settle_delay_seconds: Optional[int] = None
    srt_wait_timeout_seconds: Optional[int] = None


class PipelineConfigModel(FlexibleModel):
    project_root: Optional[str] = None
    pipeline_root: Optional[str] = None
    drop_folder_inference: Optional[str] = None
    drop_folder_training: Optional[str] = None
    srt_placement_folder: Optional[str] = None
    txt_placement_folder: Optional[str] = None
    processed_dir: Optional[str] = None
    intermediate_dir: Optional[str] = None
    output_dir: Optional[str] = None
    align_make: Optional[AlignMakeSettings] = None
    build_pair: Optional[BuildPairSettings] = None
    orchestrator: Optional[OrchestratorSettings] = None


class ModelConstraints(FlexibleModel):
    min_block_duration_s: Optional[float] = None
    max_block_duration_s: Optional[float] = None
    line_length_soft_target: Optional[int] = None
    line_length_hard_limit: Optional[int] = None
    min_chars_for_single_word_block: Optional[int] = None


class ModelSliders(FlexibleModel):
    flow: Optional[float] = None
    density: Optional[float] = None
    balance: Optional[float] = None
    line_length_leniency: Optional[float] = None
    orphan_leniency: Optional[float] = None
    structure_boost: Optional[float] = None


class ModelPaths(FlexibleModel):
    model_weights: Optional[str] = None
    constraints: Optional[str] = None


class ModelConfigModel(FlexibleModel):
    beam_width: Optional[int] = None
    constraints: Optional[ModelConstraints] = None
    sliders: Optional[ModelSliders] = None
    paths: Optional[ModelPaths] = None


# ---------------------------------------------------------------------------
# Job handling primitives
# ---------------------------------------------------------------------------
class JobType(str, Enum):
    INFERENCE = "inference"
    TRAINING_PAIR = "training_pair"
    MODEL_TRAINING = "model_training"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class InferenceRequest(BaseModel):
    media_file: str
    txt_file: Optional[str] = None
    output_directory: Optional[str] = None
    intermediate_directory: Optional[str] = None
    copy_srt_to: Optional[str] = None


class TrainingPairRequest(BaseModel):
    media_file: str
    srt_file: str
    intermediate_directory: Optional[str] = None


class ModelTrainingRequest(BaseModel):
    corpus: str
    constraints: str
    weights: str
    config: str = "config.yaml"
    iterations: int = 3
    error_boost_factor: float = Field(1.0, alias="error_boost_factor")


class JobInfo(BaseModel):
    id: str
    type: JobType
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    params: Dict[str, Any]
    result: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    progress: Optional[float] = None


@dataclass
class JobRecord:
    info: JobInfo
    log_path: Path
    runner: Callable[["JobRecord"], Dict[str, Any]]
    future: Optional[Any] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


jobs: Dict[str, JobRecord] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError:
        # Surface malformed files as empty payloads so the UI stays responsive.
        return {}


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_pipeline_config_model() -> PipelineConfigModel:
    data = _load_yaml(PIPELINE_CONFIG_PATH)
    return PipelineConfigModel(**data)


def save_pipeline_config_model(model: PipelineConfigModel) -> None:
    payload = json.loads(model.json(exclude_none=True, by_alias=True))
    _dump_yaml(PIPELINE_CONFIG_PATH, payload)


def load_model_config_model() -> ModelConfigModel:
    data = _load_yaml(MODEL_CONFIG_PATH)
    return ModelConfigModel(**data)


def save_model_config_model(model: ModelConfigModel) -> None:
    payload = json.loads(model.json(exclude_none=True, by_alias=True))
    _dump_yaml(MODEL_CONFIG_PATH, payload)


def build_base_config() -> Dict[str, Any]:
    """Load the orchestrator defaults merged with YAML overrides."""
    cfg = load_pipeline_config(PIPELINE_DEFAULTS, str(PIPELINE_CONFIG_PATH))
    cfg["project_root"] = str(PROJECT_ROOT)
    return cfg


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def tee_streams(*streams: Iterable[io.TextIOBase]):
    class _Tee(io.TextIOBase):
        def write(self, data: str) -> int:  # type: ignore[override]
            for stream in streams:
                stream.write(data)
                stream.flush()
            return len(data)

        def flush(self) -> None:  # type: ignore[override]
            for stream in streams:
                stream.flush()

    return _Tee()


# ---------------------------------------------------------------------------
# Job execution helpers
# ---------------------------------------------------------------------------
def register_job(job_type: JobType, params: Dict[str, Any], runner: Callable[[JobRecord], Dict[str, Any]]) -> JobInfo:
    job_id = str(uuid.uuid4())
    log_path = LOG_ROOT / f"{job_id}.log"
    info = JobInfo(
        id=job_id,
        type=job_type,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        params=params,
    )
    record = JobRecord(info=info, log_path=log_path, runner=runner)
    with jobs_lock:
        jobs[job_id] = record
    record.future = executor.submit(execute_job, record)
    return info


def execute_job(record: JobRecord) -> None:
    update_job(record, status=JobStatus.RUNNING)
    ensure_path(record.log_path.parent)
    try:
        with record.lock, record.log_path.open("a", encoding="utf-8") as log_file:
            tee = tee_streams(log_file, sys.__stdout__)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                result = record.runner(record)
        update_job(record, status=JobStatus.SUCCEEDED, result=result)
    except Exception as exc:  # pylint: disable=broad-except
        update_job(record, status=JobStatus.FAILED, message=str(exc))


def update_job(
    record: JobRecord,
    *,
    status: Optional[JobStatus] = None,
    result: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
) -> None:
    with record.lock:
        if status:
            record.info.status = status
        if result:
            record.info.result.update(result)
        if message:
            record.info.message = message
        record.info.updated_at = datetime.utcnow()
        record.info.metrics = derive_metrics(record)
        record.info.progress = derive_progress(record)


# ---------------------------------------------------------------------------
# Metric extraction helpers
# ---------------------------------------------------------------------------
def read_log_tail(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="ignore")
    return data[-max_chars:]


def derive_progress(record: JobRecord) -> Optional[float]:
    text = read_log_tail(record.log_path)
    if not text:
        return None
    if record.info.type == JobType.INFERENCE:
        steps_complete = sum(f"[Step {i}/3]" in text for i in range(1, 4))
        if record.info.status == JobStatus.SUCCEEDED:
            return 1.0
        return min(steps_complete / 3.0, 0.99)
    if record.info.type == JobType.TRAINING_PAIR:
        steps_complete = sum(f"[Step {i}/2]" in text for i in (1, 2))
        if record.info.status == JobStatus.SUCCEEDED:
            return 1.0
        return min(steps_complete / 2.0, 0.99)
    if record.info.type == JobType.MODEL_TRAINING:
        iterations = text.count("--- Starting Training Iteration")
        if iterations and record.info.params.get("iterations"):
            total = int(record.info.params["iterations"])
            return min(iterations / max(total, 1), 1.0 if record.info.status == JobStatus.SUCCEEDED else 0.99)
    return None


def derive_metrics(record: JobRecord) -> Dict[str, Any]:
    text = read_log_tail(record.log_path)
    metrics: Dict[str, Any] = {}
    if record.info.type == JobType.MODEL_TRAINING:
        for line in text.splitlines():
            if "accuracy on training set" in line:
                parts = line.split(":", maxsplit=1)
                if len(parts) == 2:
                    metrics["last_accuracy"] = parts[1].strip()
        if "Successfully saved constraints to" in text:
            metrics["constraints_saved"] = True
        if "Successfully saved final model weights" in text:
            metrics["weights_saved"] = True
    if record.info.type == JobType.INFERENCE:
        if record.info.result.get("srt_path"):
            metrics["output_srt"] = record.info.result["srt_path"]
    if record.info.type == JobType.TRAINING_PAIR:
        if record.info.result.get("training_json"):
            metrics["training_json"] = record.info.result["training_json"]
    return metrics


# ---------------------------------------------------------------------------
# Job runners
# ---------------------------------------------------------------------------
def inference_runner_factory(request: InferenceRequest) -> Callable[[JobRecord], Dict[str, Any]]:
    media_path = Path(request.media_file).expanduser().resolve()
    if not media_path.exists():
        raise HTTPException(status_code=400, detail=f"Media file not found: {media_path}")
    txt_path = Path(request.txt_file).expanduser().resolve() if request.txt_file else None
    if txt_path and not txt_path.exists():
        raise HTTPException(status_code=400, detail=f"Transcript file not found: {txt_path}")

    def runner(record: JobRecord) -> Dict[str, Any]:
        job_root = ensure_path(RUNTIME_ROOT / record.info.id)
        cfg = build_base_config()
        cfg["pipeline_root"] = str(job_root)
        cfg["intermediate_dir"] = str(ensure_path(job_root / "intermediate"))
        cfg["output_dir"] = str(ensure_path(job_root / "output"))
        cfg["processed_dir"] = str(ensure_path(job_root / "processed"))
        cfg["txt_placement_folder"] = str(ensure_path(job_root / "txt"))
        cfg["drop_folder_inference"] = str(ensure_path(job_root / "drop_inference"))
        cfg["drop_folder_training"] = str(ensure_path(job_root / "drop_training"))
        cfg["srt_placement_folder"] = str(ensure_path(job_root / "srt"))

        if request.intermediate_directory:
            cfg["intermediate_dir"] = str(ensure_path(Path(request.intermediate_directory).expanduser().resolve()))
        if request.output_directory:
            cfg["output_dir"] = str(ensure_path(Path(request.output_directory).expanduser().resolve()))

        cfg_align = cfg.get("align_make", {})
        cfg_align["out_root"] = cfg["intermediate_dir"]
        cfg["align_make"] = cfg_align

        cfg_build = cfg.get("build_pair", {})
        cfg_build["out_inference_dir"] = str(Path(cfg["intermediate_dir"]) / "_inference_input")
        cfg_build["out_training_dir"] = str(Path(cfg["intermediate_dir"]) / "_training")
        cfg["build_pair"] = cfg_build

        if txt_path:
            destination = Path(cfg["txt_placement_folder"]) / f"{media_path.stem}.txt"
            if destination.resolve() != txt_path:
                destination.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

        process_inference_file(media_path, cfg)

        final_srt = Path(cfg["output_dir"]) / f"{media_path.stem}.srt"
        enriched = Path(cfg["intermediate_dir"]) / "_inference_input" / f"{media_path.stem}.enriched.json"

        result: Dict[str, Any] = {
            "srt_path": str(final_srt),
            "enriched_json": str(enriched),
        }
        if request.copy_srt_to:
            copy_target = Path(request.copy_srt_to).expanduser().resolve()
            ensure_path(copy_target.parent)
            import shutil

            shutil.copy2(final_srt, copy_target)
            result["copied_srt"] = str(copy_target)
        return result

    return runner


def training_pair_runner_factory(request: TrainingPairRequest) -> Callable[[JobRecord], Dict[str, Any]]:
    media_path = Path(request.media_file).expanduser().resolve()
    srt_path = Path(request.srt_file).expanduser().resolve()
    if not media_path.exists():
        raise HTTPException(status_code=400, detail=f"Media file not found: {media_path}")
    if not srt_path.exists():
        raise HTTPException(status_code=400, detail=f"SRT file not found: {srt_path}")

    def runner(record: JobRecord) -> Dict[str, Any]:
        job_root = ensure_path(RUNTIME_ROOT / record.info.id)
        cfg = build_base_config()
        cfg["pipeline_root"] = str(job_root)
        cfg["intermediate_dir"] = str(ensure_path(job_root / "intermediate"))
        cfg["processed_dir"] = str(ensure_path(job_root / "processed"))
        cfg["txt_placement_folder"] = str(ensure_path(job_root / "txt"))
        cfg["srt_placement_folder"] = str(ensure_path(job_root / "srt"))

        if request.intermediate_directory:
            cfg["intermediate_dir"] = str(ensure_path(Path(request.intermediate_directory).expanduser().resolve()))

        cfg_align = cfg.get("align_make", {})
        cfg_align["out_root"] = cfg["intermediate_dir"]
        cfg["align_make"] = cfg_align

        cfg_build = cfg.get("build_pair", {})
        cfg_build["out_inference_dir"] = str(Path(cfg["intermediate_dir"]) / "_inference_input")
        cfg_build["out_training_dir"] = str(Path(cfg["intermediate_dir"]) / "_training")
        cfg["build_pair"] = cfg_build

        txt_copy = Path(cfg["srt_placement_folder"]) / f"{media_path.stem}.srt"
        if txt_copy.resolve() != srt_path:
            txt_copy.write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")

        process_training_file(media_path, txt_copy, cfg)

        training_json = Path(cfg["intermediate_dir"]) / "_training" / f"{media_path.stem}.train.words.json"
        return {"training_json": str(training_json)}

    return runner


def model_training_runner_factory(request: ModelTrainingRequest) -> Callable[[JobRecord], Dict[str, Any]]:
    corpus_dir = Path(request.corpus).expanduser().resolve()
    if not corpus_dir.exists():
        raise HTTPException(status_code=400, detail=f"Corpus directory not found: {corpus_dir}")

    def runner(record: JobRecord) -> Dict[str, Any]:
        import subprocess

        args = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_model.py"),
            "--corpus",
            str(corpus_dir),
            "--constraints",
            str(Path(request.constraints).expanduser().resolve()),
            "--weights",
            str(Path(request.weights).expanduser().resolve()),
            "--config",
            str(Path(request.config).expanduser().resolve()),
            "--iterations",
            str(request.iterations),
            "--error-boost-factor",
            str(request.error_boost_factor),
        ]

        process = subprocess.Popen(
            args,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Model training failed with exit code {process.returncode}")
        return {
            "constraints_path": str(Path(request.constraints).expanduser().resolve()),
            "weights_path": str(Path(request.weights).expanduser().resolve()),
        }

    return runner


# ---------------------------------------------------------------------------
# API setup
# ---------------------------------------------------------------------------
app = FastAPI(title="ISCE Pipeline UI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


@api.get("/config/pipeline")
def get_pipeline_config() -> PipelineConfigModel:
    return load_pipeline_config_model()


@api.put("/config/pipeline")
def update_pipeline_config(config: PipelineConfigModel) -> JSONResponse:
    save_pipeline_config_model(config)
    return JSONResponse({"status": "ok"})


@api.get("/config/model")
def get_model_config() -> ModelConfigModel:
    return load_model_config_model()


@api.put("/config/model")
def update_model_config(config: ModelConfigModel) -> JSONResponse:
    save_model_config_model(config)
    return JSONResponse({"status": "ok"})


@api.get("/jobs")
def list_jobs() -> List[JobInfo]:
    with jobs_lock:
        return [record.info for record in jobs.values()]


@api.get("/jobs/{job_id}")
def get_job(job_id: str) -> JobInfo:
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return record.info


@api.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str) -> Dict[str, Any]:
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"log": read_log_tail(record.log_path, max_chars=20000)}


@api.post("/jobs/inference", response_model=JobInfo)
def create_inference_job(request: InferenceRequest) -> JobInfo:
    runner = inference_runner_factory(request)
    return register_job(JobType.INFERENCE, request.dict(), runner)


@api.post("/jobs/training-pair", response_model=JobInfo)
def create_training_pair_job(request: TrainingPairRequest) -> JobInfo:
    runner = training_pair_runner_factory(request)
    return register_job(JobType.TRAINING_PAIR, request.dict(), runner)


@api.post("/jobs/model-training", response_model=JobInfo)
def create_model_training_job(request: ModelTrainingRequest) -> JobInfo:
    runner = model_training_runner_factory(request)
    return register_job(JobType.MODEL_TRAINING, request.dict(by_alias=True), runner)


app.include_router(api)

frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


__all__ = ["app"]
