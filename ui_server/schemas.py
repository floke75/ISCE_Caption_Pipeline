from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceJobRequest(BaseModel):
    media_path: str = Field(..., description="Absolute path to the media file")
    transcript_path: Optional[str] = Field(None, description="Optional path to the transcript txt file")
    output_directory: Optional[str] = Field(None, description="Directory to copy the generated SRT into")
    name: Optional[str] = None
    pipeline_overrides: Dict[str, Any] = Field(default_factory=dict)
    model_overrides: Dict[str, Any] = Field(default_factory=dict)


class TrainingPairJobRequest(BaseModel):
    media_path: str
    srt_path: str
    output_directory: Optional[str] = None
    name: Optional[str] = None
    pipeline_overrides: Dict[str, Any] = Field(default_factory=dict)


class ModelTrainingJobRequest(BaseModel):
    corpus_dir: str
    constraints_path: Optional[str] = None
    weights_path: Optional[str] = None
    iterations: int = Field(3, ge=1, le=20)
    error_boost_factor: float = Field(1.0, ge=0.0)
    name: Optional[str] = None
    config_path: Optional[str] = None
    model_overrides: Dict[str, Any] = Field(default_factory=dict)


class JobSummary(BaseModel):
    id: str
    job_type: str
    name: str
    status: str
    progress: float
    stage: Optional[str]
    message: Optional[str]
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class JobDetail(JobSummary):
    parameters: Dict[str, Any]
    workspace: str

    @model_validator(mode="before")
    def _normalise_workspace(cls, values: Any) -> Any:
        if isinstance(values, dict):
            workspace = values.get("workspace")
            if isinstance(workspace, Path):
                values["workspace"] = str(workspace)
        return values


class LogChunk(BaseModel):
    content: str
    next_offset: int
