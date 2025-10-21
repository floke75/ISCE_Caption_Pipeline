"""Pydantic schemas for the UI API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator


class PipelineOverrides(BaseModel):
    """User supplied overrides for pipeline configuration keys."""

    pipeline_root: Optional[str] = Field(None, description="Root directory used for pipeline artifacts.")
    output_dir: Optional[str] = Field(None, description="Directory where inference SRT files will be written.")
    intermediate_dir: Optional[str] = Field(None, description="Directory for intermediate assets.")
    txt_placement_folder: Optional[str] = Field(None, description="Location searched for TXT transcripts.")
    drop_folder_inference: Optional[str] = Field(None, description="Drop folder for inference media.")
    drop_folder_training: Optional[str] = Field(None, description="Drop folder for training media.")
    srt_placement_folder: Optional[str] = Field(None, description="Location of manual SRTs for training runs.")


class InferenceRequest(BaseModel):
    media_path: str = Field(..., description="Path to the media file to process.")
    transcript_path: Optional[str] = Field(None, description="Optional path to a prepared TXT transcript.")
    pipeline_overrides: Optional[PipelineOverrides] = None


class TrainingPairRequest(BaseModel):
    media_path: str = Field(..., description="Path to the media file to align.")
    srt_path: str = Field(..., description="Ground-truth SRT used to label the training pair.")
    pipeline_overrides: Optional[PipelineOverrides] = None


class TrainModelRequest(BaseModel):
    corpus_dir: str = Field(..., description="Directory containing *.json training files.")
    constraints_path: str = Field(..., description="Output path for constraints.json")
    weights_path: str = Field(..., description="Output path for model_weights.json")
    iterations: int = Field(3, ge=1, le=20, description="Number of reweighting iterations to execute.")
    error_boost_factor: float = Field(1.0, ge=0.0, le=10.0, description="Weight increment applied to mistakes.")
    config_path: str = Field("config.yaml", description="Path to the captioning configuration YAML.")

    @validator("corpus_dir", "constraints_path", "weights_path", "config_path")
    def non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("Value must not be empty")
        return value


class ConfigEnvelope(BaseModel):
    defaults: Dict[str, Any]
    overrides: Dict[str, Any]
    resolved: Dict[str, Any]


class ConfigUpdate(BaseModel):
    content: Dict[str, Any]


class ConfigWriteResult(BaseModel):
    path: str
    updated: bool


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    metadata: Dict[str, Any]
    error: Optional[str]


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class JobLogsResponse(BaseModel):
    lines: list[str]
    next_index: int
    total: int


def job_to_schema(job: "Job") -> JobResponse:
    from .job_manager import Job

    data = job.to_dict()
    return JobResponse(**data)
