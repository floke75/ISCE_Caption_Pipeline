"""Pydantic models for the UI backend API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    pythonExecutable: Optional[str] = Field(
        None,
        description="Path to the Python interpreter that should execute the scripts.",
    )

    def runtime_params(self) -> Dict[str, Any]:
        payload = self.dict(exclude_none=True, exclude={"configOverrides"})
        return payload


class InferenceJobRequest(JobRequest):
    mediaPath: str = Field(..., description="Absolute path to the media file.")
    transcriptPath: str = Field(..., description="Absolute path to the transcript file.")
    configOverrides: Optional[Dict[str, Any]] = None


class TrainingPairJobRequest(JobRequest):
    transcriptPath: str = Field(..., description="Path to the transcript or SRT file.")
    asrReference: str = Field(..., description="Path to the enriched ASR JSON file.")
    mode: Optional[str] = Field("inference", regex="^(inference|training)$")
    outputBasename: Optional[str] = None
    configOverrides: Optional[Dict[str, Any]] = None


class TrainingJobRequest(JobRequest):
    corpusDir: str = Field(..., description="Directory containing *.json training data files.")
    iterations: Optional[int] = Field(3, ge=1, le=20)
    errorBoostFactor: Optional[float] = Field(1.0, ge=0.0)
    configOverrides: Optional[Dict[str, Any]] = None


class JobInfo(BaseModel):
    id: str
    jobType: str
    params: Dict[str, Any]
    createdAt: float
    status: str
    startedAt: Optional[float]
    finishedAt: Optional[float]
    progress: float
    message: Optional[str]
    extra: Dict[str, Any]


class ConfigPayload(BaseModel):
    data: Dict[str, Any]
