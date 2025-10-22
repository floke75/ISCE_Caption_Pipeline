from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    pipeline_overrides: Optional[Dict[str, Any]] = Field(default=None, alias="pipelineOverrides")


class InferenceRequest(BaseJobPayload):
    media_path: Path = Field(alias="mediaPath")
    transcript_path: Path = Field(alias="transcriptPath")
    output_path: Optional[Path] = Field(default=None, alias="outputPath")
    output_basename: Optional[str] = Field(default=None, alias="outputBasename")
    model_overrides: Optional[Dict[str, Any]] = Field(default=None, alias="modelOverrides")


class TrainingPairsRequest(BaseJobPayload):
    transcript_path: Path = Field(alias="transcriptPath")
    asr_reference_path: Path = Field(alias="asrReferencePath")
    output_basename: Optional[str] = Field(default=None, alias="outputBasename")
    asr_only_mode: bool = Field(default=False, alias="asrOnlyMode")


class ModelTrainingRequest(BaseJobPayload):
    corpus_dir: Path = Field(alias="corpusDir")
    constraints_output: Optional[Path] = Field(default=None, alias="constraintsOutput")
    weights_output: Optional[Path] = Field(default=None, alias="weightsOutput")
    iterations: int = 3
    error_boost_factor: float = Field(default=1.0, alias="errorBoostFactor")
    model_overrides: Optional[Dict[str, Any]] = Field(default=None, alias="modelOverrides")
