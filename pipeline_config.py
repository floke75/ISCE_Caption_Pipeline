"""Configuration loading utility for the ISCE pipeline.

This module provides a standardized function, `load_pipeline_config`, for
managing configuration across the various pipeline scripts. It allows scripts
to define in-code default settings, which can be overridden by a central
`pipeline_config.yaml` file.

The key features of this loader are:
-   **Default Fallback**: Ensures that the pipeline can run with sensible
    defaults even if the YAML file is missing.
-   **Recursive Merging**: Deeply merges the YAML configuration over the
    defaults, allowing users to override only specific nested values.
-   **Path Resolution**: Automatically resolves path placeholders (like
    `{project_root}` or `{pipeline_root}`) to create absolute, portable paths.
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any


def _recursive_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Merges the `update` dictionary into the `base` dictionary recursively.

    If a key exists in both dictionaries and its value is a dictionary in
    both, it recursively merges the nested dictionaries. Otherwise, the value
    from `update` overwrites the value in `base`.

    Args:
        base: The dictionary to be updated.
        update: The dictionary containing new values.

    Returns:
        The updated `base` dictionary.
    """
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_paths(config: Dict[str, Any], context: Dict[str, str]) -> Dict[str, Any]:
    """Resolves path placeholders like `{project_root}` using a given context.

    Recursively iterates through a configuration dictionary and formats any
    string values that contain `{placeholder}` style variables using the
    provided context dictionary.

    Args:
        config: The configuration dictionary with unresolved path strings.
        context: A dictionary mapping placeholder keys to their string values.

    Returns:
        The configuration dictionary with path placeholders resolved.
    """
    for k, v in config.items():
        if isinstance(v, str) and "{" in v and "}" in v:
            try:
                config[k] = v.format(**context)
            except KeyError:
                # Ignore if a placeholder key is not in the context.
                pass
        elif isinstance(v, dict):
            # Recurse into nested dictionaries.
            config[k] = _resolve_paths(v, context)
    return config


def load_pipeline_config(
    default_settings: Dict[str, Any],
    yaml_path: str = "pipeline_config.yaml"
) -> Dict[str, Any]:
    """Loads a central pipeline configuration with a robust fallback mechanism.

    This function orchestrates the configuration loading process:
    1.  Starts with the hard-coded `default_settings` provided by the calling
        script (e.g., `run_pipeline.py`).
    2.  Tries to load the YAML file specified by `yaml_path`.
    3.  If the YAML file exists, it recursively overrides the defaults with
        the values from the file.
    4.  Finally, it resolves any path placeholders (e.g., `{project_root}`)
        found in the string values of the merged configuration.

    Args:
        default_settings: A dictionary of default settings that serves as the
            base configuration.
        yaml_path: The path to the YAML configuration file that will override
            the defaults. Defaults to "pipeline_config.yaml".

    Returns:
        The final, resolved configuration dictionary.
    """
    config = default_settings.copy()

    try:
        p = Path(yaml_path)
        if p.exists():
            print(f"[CONFIG] Loading overrides from: {p.name}")
            with open(p, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config and isinstance(yaml_config, dict):
                config = _recursive_update(config, yaml_config)
        else:
            print(f"[CONFIG] No {p.name} found. Using default in-script settings.")
    except Exception as e:
        print(f"[CONFIG] WARNING: Could not load or parse {yaml_path}. Using defaults. Error: {e}")

    # Create the context for path resolution from the top-level string keys.
    path_context = {k: v for k, v in config.items() if isinstance(v, str)}
    config = _resolve_paths(config, path_context)

    return config