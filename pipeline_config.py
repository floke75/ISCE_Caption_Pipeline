# C:\dev\Captions_Formatter\Formatter_machine\pipeline_config.py

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any

def _recursive_update(base: Dict, update: Dict) -> Dict:
    """Merges the update dict into the base dict recursively."""
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base

def _resolve_paths(config: Dict, context: Dict) -> Dict:
    """Resolves path placeholders like {project_root} using a given context."""
    for k, v in config.items():
        if isinstance(v, str) and "{" in v and "}" in v:
            try:
                config[k] = v.format(**context)
            except KeyError:
                pass
        elif isinstance(v, dict):
            config[k] = _resolve_paths(v, context) # Recurse
    return config

def load_pipeline_config(
    default_settings: Dict[str, Any],
    yaml_path: str = "pipeline_config.yaml"
) -> Dict[str, Any]:
    """
    Loads a central pipeline configuration with a robust fallback mechanism.
    
    1. Starts with the hard-coded `default_settings` provided by the calling script.
    2. Tries to load the specified `yaml_path`.
    3. If the YAML file exists, it recursively overrides the defaults with its values.
    4. Resolves path placeholders (e.g., {project_root}) at the end.
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

    # Create the context for path resolution from the top-level keys
    path_context = {k: v for k, v in config.items() if isinstance(v, str)}
    config = _resolve_paths(config, path_context)
    
    return config