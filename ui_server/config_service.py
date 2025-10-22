from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from pipeline_config import load_pipeline_config
from run_pipeline import DEFAULT_SETTINGS


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep merge of two dictionaries without mutating the inputs."""
    result: Dict[str, Any] = copy.deepcopy(base)
    stack = [(result, overlay)]
    while stack:
        target, source = stack.pop()
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                stack.append((target[key], value))
            else:
                target[key] = copy.deepcopy(value)
    return result


def _ensure_project_root(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the project_root points to the current repository if missing."""
    project_root = Path(config.get("project_root", ""))
    repo_root = Path(__file__).resolve().parent.parent
    if not project_root.exists():
        config["project_root"] = str(repo_root)
    return config


def _normalise_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Path-like structures to plain Python types for YAML dumping."""
    if isinstance(config, dict):
        return {k: _normalise_types(v) for k, v in config.items()}
    if isinstance(config, (list, tuple)):
        return [_normalise_types(v) for v in config]
    if isinstance(config, Path):
        return str(config)
    return config


class YAMLConfigService:
    """Thread-safe helper for reading and writing YAML configuration files."""

    def __init__(self, path: Path, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._defaults = copy.deepcopy(defaults or {})

    @property
    def path(self) -> Path:
        return self._path

    def load(self, resolved: bool = True) -> Dict[str, Any]:
        """Load the YAML config merged with defaults, optionally resolving paths."""
        with self._lock:
            data: Dict[str, Any] = copy.deepcopy(self._defaults)
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"Configuration at {self._path} must contain a mapping")
                data = _deep_merge(data, raw)
            if resolved:
                data = load_pipeline_config(data, str(self._path))
            else:
                data = _deep_merge({}, data)
        return _normalise_types(data)

    def save(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the provided config to disk and return the stored copy."""
        with self._lock:
            normalised = _normalise_types(config)
            yaml_text = yaml.safe_dump(normalised, sort_keys=False, allow_unicode=True)
            self._path.write_text(yaml_text, encoding="utf-8")
        return normalised

    def update(self, partial: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge the partial update into the stored config."""
        with self._lock:
            current = yaml.safe_load(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
            if current is None:
                current = {}
            if not isinstance(current, dict):
                raise ValueError(f"Configuration at {self._path} must contain a mapping")
            merged = _deep_merge(current, partial)
            normalised = _normalise_types(merged)
            yaml_text = yaml.safe_dump(normalised, sort_keys=False, allow_unicode=True)
            self._path.write_text(yaml_text, encoding="utf-8")
        return self.load(resolved=True)


class PipelineConfigService(YAMLConfigService):
    """Specialised service that ensures the pipeline config is runnable."""

    def __init__(self, path: Path | None = None) -> None:
        target = Path(path) if path else Path("pipeline_config.yaml")
        super().__init__(target, defaults=DEFAULT_SETTINGS)

    def load(self, resolved: bool = True) -> Dict[str, Any]:  # type: ignore[override]
        config = super().load(resolved=resolved)
        return _ensure_project_root(config)


class ModelConfigService(YAMLConfigService):
    def __init__(self, path: Path | None = None) -> None:
        target = Path(path) if path else Path("config.yaml")
        defaults = yaml.safe_load(target.read_text(encoding="utf-8")) if target.exists() else {}
        super().__init__(target, defaults=defaults or {})
