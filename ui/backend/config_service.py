"""Utilities for loading, describing, and updating pipeline configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

import yaml


@dataclass(frozen=True)
class ConfigField:
    """Metadata describing a configurable value for the UI form renderer."""

    path: List[str]
    label: str
    field_type: str
    description: Optional[str] = None
    section: str = "General"
    options: Optional[List[Any]] = None
    advanced: bool = False

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


def _recursive_update(base: MutableMapping[str, Any], update: Dict[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict):
            child = base.get(key)
            if not isinstance(child, MutableMapping):
                child = {}
            base[key] = _recursive_update(dict(child), value)
        else:
            base[key] = value
    return base


def _resolve_placeholders(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in list(config.items()):
        if isinstance(value, dict):
            config[key] = _resolve_placeholders(value, context)
        elif isinstance(value, str) and "{" in value and "}" in value:
            try:
                config[key] = value.format(**context)
            except KeyError:
                pass
    return config


def _prune_nulls(data: Any) -> Any:
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in data.items():
            pruned = _prune_nulls(value)
            if pruned is None:
                continue
            cleaned[key] = pruned
        return cleaned
    if isinstance(data, list):
        return [item for item in (_prune_nulls(v) for v in data) if item is not None]
    return data


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class ConfigService:
    """Loads the pipeline configuration and exposes metadata for the UI."""

    def __init__(self, repo_root: Path, storage_root: Path) -> None:
        self._repo_root = repo_root
        self._base_config_path = repo_root / "pipeline_config.yaml"
        self._overrides_path = storage_root / "config" / "pipeline_overrides.yaml"
        self._field_catalog = self._build_field_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def base_config(self) -> Dict[str, Any]:
        return self._load_yaml(self._base_config_path)

    def stored_overrides(self) -> Dict[str, Any]:
        if self._overrides_path.exists():
            return self._load_yaml(self._overrides_path)
        return {}

    def effective_config(self) -> Dict[str, Any]:
        base = self.base_config()
        overrides = self.stored_overrides()
        merged = _recursive_update(base, overrides)
        context = {k: v for k, v in merged.items() if isinstance(v, str)}
        return _resolve_placeholders(merged, context)

    def resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        context = {k: v for k, v in config.items() if isinstance(v, str)}
        return _resolve_placeholders(config, context)

    def describe_fields(self) -> List[ConfigField]:
        return list(self._field_catalog)

    def save_overrides(self, overrides: Dict[str, Any]) -> None:
        cleaned = _prune_nulls(overrides)
        _ensure_parent(self._overrides_path)
        with self._overrides_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cleaned, fh, allow_unicode=True, sort_keys=False)

    def apply_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        overrides = self.stored_overrides()
        merged = _recursive_update(overrides, patch)
        self.save_overrides(merged)
        return self.effective_config()

    def reset_overrides(self) -> None:
        if self._overrides_path.exists():
            self._overrides_path.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file {path} must contain a mapping")
        return data

    def _build_field_catalog(self) -> List[ConfigField]:
        return [
            ConfigField(
                path=["project_root"],
                section="Core paths",
                label="Project root",
                field_type="path",
                description="Root of the repository containing all scripts and assets.",
            ),
            ConfigField(
                path=["pipeline_root"],
                section="Core paths",
                label="Pipeline root",
                field_type="path",
                description="Base directory where drop folders, intermediates, and outputs are created.",
            ),
            ConfigField(
                path=["align_make", "whisper_model_id"],
                section="Audio alignment",
                label="Whisper model",
                field_type="string",
                description="Model identifier used for transcription in align_make.py.",
            ),
            ConfigField(
                path=["align_make", "align_model_id"],
                section="Audio alignment",
                label="Alignment model",
                field_type="string",
                description="Model identifier used for wav2vec2 alignment.",
            ),
            ConfigField(
                path=["align_make", "language"],
                section="Audio alignment",
                label="Language code",
                field_type="string",
                description="Two-letter language code passed to WhisperX.",
            ),
            ConfigField(
                path=["align_make", "compute_type"],
                section="Audio alignment",
                label="Compute type",
                field_type="select",
                options=["float16", "float32", "int8"],
                description="Torch compute precision for WhisperX (GPU vs CPU).",
            ),
            ConfigField(
                path=["align_make", "batch_size"],
                section="Audio alignment",
                label="Batch size",
                field_type="number",
                description="Batch size for WhisperX transcription.",
            ),
            ConfigField(
                path=["align_make", "do_diarization"],
                section="Audio alignment",
                label="Enable diarization",
                field_type="boolean",
            ),
            ConfigField(
                path=["build_pair", "language"],
                section="Enrichment",
                label="Language",
                field_type="string",
                description="spaCy model language used for linguistic features.",
            ),
            ConfigField(
                path=["build_pair", "spacy_enable"],
                section="Enrichment",
                label="Enable spaCy",
                field_type="boolean",
            ),
            ConfigField(
                path=["build_pair", "emit_asr_style_training_copy"],
                section="Enrichment",
                label="Emit ASR-style training copy",
                field_type="boolean",
                advanced=True,
            ),
            ConfigField(
                path=["orchestrator", "poll_interval_seconds"],
                section="Orchestrator",
                label="Poll interval (s)",
                field_type="number",
                description="Interval between hot-folder scans when using the CLI orchestrator.",
            ),
            ConfigField(
                path=["orchestrator", "audio_exts"],
                section="Orchestrator",
                label="Audio extensions",
                field_type="list",
                description="Recognised audio/video extensions for hot folder ingestion.",
                advanced=True,
            ),
        ]

    # Utilities used by the API layer
    def extract_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return just the fields defined in the catalog from a config mapping."""
        values: Dict[str, Any] = {}
        for field in self._field_catalog:
            target = config
            for key in field.path[:-1]:
                target = target.get(key, {}) if isinstance(target, dict) else {}
            last = field.path[-1]
            if isinstance(target, dict) and last in target:
                values[field.dotted_path] = target[last]
        return values

    def build_patch(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Transform dotted-path updates back into nested dictionaries."""
        patch: Dict[str, Any] = {}
        for dotted, value in updates.items():
            parts = dotted.split(".")
            cursor = patch
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
        return patch

