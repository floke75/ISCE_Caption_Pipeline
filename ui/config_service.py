"""Utilities for loading and persisting pipeline configuration for the UI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from pipeline_config import load_pipeline_config
from run_pipeline import DEFAULT_SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "pipeline_config.yaml"
CORE_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _ensure_runtime_defaults() -> Dict[str, Any]:
    defaults = copy.deepcopy(DEFAULT_SETTINGS)
    defaults["project_root"] = str(PROJECT_ROOT)
    defaults.setdefault("pipeline_root", str(PROJECT_ROOT / "runtime_pipeline"))
    return defaults


def runtime_defaults() -> Dict[str, Any]:
    """Expose sanitized defaults for API consumers."""

    return _ensure_runtime_defaults()


def _merge_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_effective_pipeline_config(additional_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    defaults = _ensure_runtime_defaults()
    resolved = load_pipeline_config(defaults, str(PIPELINE_CONFIG_PATH))
    resolved["project_root"] = str(PROJECT_ROOT)
    resolved.setdefault("pipeline_root", defaults["pipeline_root"])
    if additional_overrides:
        resolved = _merge_overrides(resolved, additional_overrides)
    return resolved


def load_pipeline_overrides() -> Dict[str, Any]:
    if not PIPELINE_CONFIG_PATH.exists():
        return {}
    with open(PIPELINE_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def write_pipeline_overrides(data: Dict[str, Any]) -> None:
    PIPELINE_CONFIG_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_core_config() -> Dict[str, Any]:
    if not CORE_CONFIG_PATH.exists():
        return {}
    with open(CORE_CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def write_core_config(data: Dict[str, Any]) -> None:
    CORE_CONFIG_PATH.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def ensure_directories(cfg: Dict[str, Any]) -> None:
    from run_pipeline import setup_directories

    setup_directories(cfg)


def ensure_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
