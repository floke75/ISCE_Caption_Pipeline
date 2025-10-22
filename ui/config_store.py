from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges ``updates`` into ``base`` and returns a new dict."""
    merged = copy.deepcopy(base)
    stack: list[tuple[Dict[str, Any], Dict[str, Any]]] = [(merged, updates)]
    while stack:
        current, update = stack.pop()
        for key, value in update.items():
            if (
                isinstance(value, dict)
                and key in current
                and isinstance(current[key], dict)
            ):
                stack.append((current[key], value))
            else:
                current[key] = copy.deepcopy(value)
    return merged


def _resolve_templates(config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``str.format`` style placeholders within a configuration dict."""
    resolved: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, str) and "{" in value and "}" in value:
            try:
                resolved[key] = value.format(**context)
            except KeyError:
                resolved[key] = value
        elif isinstance(value, dict):
            resolved[key] = _resolve_templates(value, context)
        else:
            resolved[key] = value
    return resolved


class ConfigStore:
    """Thread-safe helper for loading and persisting YAML configuration files."""

    def __init__(self, path: Path, defaults: Optional[Dict[str, Any]] = None) -> None:
        self._path = Path(path)
        self._defaults = copy.deepcopy(defaults or {})
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> Dict[str, Any]:
        """Load the YAML configuration merged with the optional defaults."""
        with self._lock:
            data = copy.deepcopy(self._defaults)
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(
                        f"Configuration file {self._path} must contain a mapping."
                    )
                data = _deep_merge(data, loaded)
            return data

    def write(self, config: Dict[str, Any]) -> None:
        """Persist ``config`` to disk, ensuring parent directories exist."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ``updates`` into the stored configuration and persist it."""
        with self._lock:
            current = self.read()
            merged = _deep_merge(current, updates)
            self.write(merged)
            return merged

    def dump_yaml(self) -> str:
        """Return the current configuration as a YAML string."""
        config = self.read()
        return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)

    def prepare_runtime_config(
        self,
        overrides: Optional[Dict[str, Any]],
        workspace: Path,
        *,
        filename: Optional[str] = None,
        isolation: bool = False,
        repo_root: Optional[Path] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        """
        Materialise a configuration file tailored for a specific job workspace.

        Args:
            overrides: Optional dictionary of configuration overrides.
            workspace: Base path for the job's temporary assets.
            filename: Optional explicit filename for the runtime YAML file.
            isolation: If ``True``, key path entries tied to shared directories
                are rewritten to live underneath the workspace.
            repo_root: Project root used when enforcing isolation.
            extra_context: Additional placeholder values for template expansion.

        Returns:
            A tuple of ``(config_dict, runtime_yaml_path)``.
        """
        runtime_dir = workspace / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)

        config = self.read()
        if overrides:
            config = _deep_merge(config, overrides)

        if isolation:
            if repo_root is None:
                raise ValueError("repo_root must be provided when isolation=True")
            config = _apply_isolation(config, workspace, repo_root)

        context: Dict[str, Any] = {
            key: value for key, value in config.items() if isinstance(value, str)
        }
        if extra_context:
            context.update(extra_context)
        config = _resolve_templates(config, context)
        config = _absolutify_resource_paths(
            config,
            self._path.parent.resolve(),
        )

        runtime_path = runtime_dir / (filename or self._path.name)
        with runtime_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

        return config, runtime_path


def _apply_isolation(config: Dict[str, Any], workspace: Path, repo_root: Path) -> Dict[str, Any]:
    """Rewrite config paths so the job operates within an isolated sandbox."""
    sandbox = workspace / "pipeline"
    intermediate = sandbox / "_intermediate"
    processed = sandbox / "_processed"

    rewritten = copy.deepcopy(config)
    rewritten["project_root"] = str(repo_root)
    rewritten["pipeline_root"] = str(sandbox)
    rewritten["drop_folder_inference"] = str(sandbox / "drop_inference")
    rewritten["drop_folder_training"] = str(sandbox / "drop_training")
    rewritten["srt_placement_folder"] = str(sandbox / "manual_srt")
    rewritten["txt_placement_folder"] = str(sandbox / "manual_txt")
    rewritten["processed_dir"] = str(processed)
    rewritten["intermediate_dir"] = str(intermediate)
    rewritten["output_dir"] = str(sandbox / "_output")

    align_cfg = rewritten.setdefault("align_make", {})
    align_cfg["out_root"] = str(intermediate)
    align_cfg["cache_dir"] = str(workspace / "cache")

    build_cfg = rewritten.setdefault("build_pair", {})
    build_cfg["out_training_dir"] = str(intermediate / "_training")
    build_cfg["out_inference_dir"] = str(intermediate / "_inference_input")

    Path(rewritten["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(rewritten["intermediate_dir"]).mkdir(parents=True, exist_ok=True)
    Path(rewritten["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(build_cfg["out_training_dir"]).mkdir(parents=True, exist_ok=True)
    Path(build_cfg["out_inference_dir"]).mkdir(parents=True, exist_ok=True)
    Path(align_cfg["out_root"]).mkdir(parents=True, exist_ok=True)

    return rewritten


def _absolutify_resource_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Ensure resource paths remain valid when the config is relocated."""

    if not isinstance(config, dict):
        return config

    paths_section = config.get("paths")
    if not isinstance(paths_section, dict):
        return config

    for key, value in list(paths_section.items()):
        if isinstance(value, str):
            candidate = Path(value)
            if not candidate.is_absolute():
                paths_section[key] = str((base_dir / candidate).resolve())

    return config
