from __future__ import annotations

from datetime import datetime
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ui_server.path_validation import describe_allowlist


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
    workspace: str
    cancel_requested: bool = False

    model_config = ConfigDict(from_attributes=True)


class JobDetail(JobSummary):
    parameters: Dict[str, Any]

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


class PathValidationRequest(BaseModel):
    path: str = Field(..., description="Path supplied by the operator")
    kind: Literal["file", "directory", "any"] = Field(
        "file", description="Expected filesystem entry type"
    )
    must_exist: bool = Field(True, description="Require the entry to exist")
    allow_create: bool = Field(
        False, description="Allow creation of the entry if it does not exist"
    )
    purpose: Optional[str] = Field(
        None, description="Human-friendly label for error messages"
    )


class PathValidationResponse(BaseModel):
    valid: bool
    resolved_path: Optional[str] = None
    exists: bool
    is_file: bool
    is_dir: bool
    message: Optional[str] = None
    allowed_roots: list[str] = Field(default_factory=describe_allowlist)
    root: Optional[str] = Field(
        None, description="Allowlisted root directory that contains the path"
    )
