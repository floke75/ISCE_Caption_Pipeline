"""Configuration helpers for the UI backend.

This module centralizes loading and saving of the pipeline configuration so
that both the API layer and the job runners can reuse the same behaviour.
The implementation takes inspiration from the contest entries that offered
structured config editors while preserving unknown keys.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return the result."""
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


class PipelineConfigService:
    """Helper responsible for loading and persisting the pipeline config."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._lock = threading.Lock()

    @property
    def config_path(self) -> Path:
        return self._config_path

    def load(self) -> Dict[str, Any]:
        """Load the YAML configuration as a dictionary."""
        if not self._config_path.exists():
            return {}
        with self._config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data

    def save(self, data: Dict[str, Any]) -> None:
        """Persist ``data`` back to YAML."""
        with self._lock:
            with self._config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)

    def apply_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ``patch`` to the stored config and return the merged result."""
        with self._lock:
            current = self.load()
            merged = _deep_merge(dict(current), patch)
            self.save(merged)
        return merged

    def build_job_config(
        self,
        workspace: Path,
        overrides: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return a config dictionary tailored for a specific job workspace."""
        config = self.load()
        overrides = overrides or {}
        workspace_root = workspace / "pipeline"
        project_root = Path(__file__).resolve().parents[2]

        derived = {
            "project_root": str(project_root),
            "pipeline_root": str(workspace_root),
            "align_make": {
                "out_root": str(workspace_root / "_intermediate"),
                "cache_dir": str(workspace_root / "cache"),
            },
            "build_pair": {
                "out_training_dir": str(workspace_root / "_training"),
                "out_inference_dir": str(workspace_root / "_inference_input"),
            },
        }
        for path_key in ("_intermediate", "_training", "_inference_input", "cache"):
            (workspace_root / path_key).mkdir(parents=True, exist_ok=True)
        config = _deep_merge(config, derived)
        if overrides:
            config = _deep_merge(config, overrides)
        return config
