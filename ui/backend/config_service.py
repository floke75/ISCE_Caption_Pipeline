"""Utilities for loading, describing, and updating pipeline configuration."""
from __future__ import annotations

from copy import deepcopy
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
    read_only: bool = False

    @property
    def dotted_path(self) -> str:
        return ".".join(self.path)


def _recursive_update(base: MutableMapping[str, Any], update: Dict[str, Any]) -> MutableMapping[str, Any]:
    """Merge ``update`` into ``base`` recursively without mutating inputs."""
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
    """Expand ``str.format`` placeholders in ``config`` using ``context``."""
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
    """Remove ``None`` values from arbitrarily nested mappings and sequences."""
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
    """Create ``path.parent`` if it does not already exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


class ConfigService:
    """Loads a configuration document and exposes metadata for the UI."""

    def __init__(
        self,
        repo_root: Path,
        storage_root: Path,
        *,
        base_config_path: Optional[Path] = None,
        overrides_path: Optional[Path] = None,
        field_catalog: Optional[Iterable[ConfigField]] = None,
    ) -> None:
        self._repo_root = repo_root
        self._base_config_path = base_config_path or (repo_root / "pipeline_config.yaml")
        self._overrides_path = overrides_path or (storage_root / "config" / "pipeline_overrides.yaml")
        catalog = list(field_catalog) if field_catalog is not None else self._build_field_catalog()
        self._field_catalog = catalog
        self._field_map = {tuple(field.path): field for field in self._field_catalog}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def base_config(self) -> Dict[str, Any]:
        """Return the repository's baseline configuration document."""
        return self._load_yaml(self._base_config_path)

    def stored_overrides(self) -> Dict[str, Any]:
        """Return the persisted user overrides, if any exist on disk."""
        if self._overrides_path.exists():
            return self._load_yaml(self._overrides_path)
        return {}

    def effective_config(self) -> Dict[str, Any]:
        """Return the merged configuration after applying stored overrides."""
        base = self.base_config()
        overrides = self.stored_overrides()
        merged = _recursive_update(base, overrides)
        context = {k: v for k, v in merged.items() if isinstance(v, str)}
        return _resolve_placeholders(merged, context)

    def resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve placeholders in an arbitrary config document."""
        context = {k: v for k, v in config.items() if isinstance(v, str)}
        return _resolve_placeholders(config, context)

    def describe_fields(self) -> List[ConfigField]:
        """Expose the catalog used to render editable fields in the UI."""
        return list(self._field_catalog)

    def describe_tree(self) -> List[Dict[str, Any]]:
        """Return a hierarchical view of base values, overrides, and effective output."""
        base = self.base_config()
        effective = self.effective_config()
        overrides = self.stored_overrides()
        keys = sorted(set(base.keys()) | set(effective.keys()) | set(overrides.keys()))
        return [
            self._build_node([key], base.get(key), effective.get(key), overrides)
            for key in keys
        ]

    def save_overrides(self, overrides: Dict[str, Any]) -> None:
        """Persist overrides to disk after removing empty/null values."""
        cleaned = _prune_nulls(overrides)
        _ensure_parent(self._overrides_path)
        with self._overrides_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cleaned, fh, allow_unicode=True, sort_keys=False)

    def apply_patch(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Merge ``patch`` into the stored overrides and return the new effective config."""
        overrides = self.stored_overrides()
        merged = _recursive_update(overrides, patch)
        self.save_overrides(merged)
        return self.effective_config()

    def reset_overrides(self) -> None:
        """Delete the overrides file if one currently exists."""
        if self._overrides_path.exists():
            self._overrides_path.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML document from ``path`` and ensure it is a mapping."""
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file {path} must contain a mapping")
        return data

    def _build_field_catalog(self) -> List[ConfigField]:
        """Generate the default field metadata consumed by the SPA."""
        return [
            ConfigField(
                path=["project_root"],
                section="Core paths",
                label="Project root",
                field_type="path",
                description=(
                    "Root of the repository containing all scripts and assets. "
                    "For UI-launched jobs this value is managed automatically and "
                    "cannot be overridden."
                ),
                read_only=True,
            ),
            ConfigField(
                path=["pipeline_root"],
                section="Core paths",
                label="Pipeline root",
                field_type="path",
                description=(
                    "Base directory where drop folders, intermediates, and outputs "
                    "are created. UI jobs always use a per-run workspace and ignore "
                    "manual overrides."
                ),
                read_only=True,
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

    def _build_node(
        self,
        path: List[str],
        default_value: Any,
        current_value: Any,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        field = self._field_map.get(tuple(path))
        value_type = self._determine_value_type(field, default_value, current_value)
        description = field.description if field else None
        options = field.options if field else None
        advanced = field.advanced if field else False
        children: List[Dict[str, Any]] = []

        # If this is a mapping node, recursively describe children.
        if value_type == "object":
            default_mapping = default_value if isinstance(default_value, dict) else {}
            current_mapping = current_value if isinstance(current_value, dict) else {}
            override_mapping = self._value_at(overrides, path)
            child_keys = sorted(
                set(default_mapping.keys())
                | set(current_mapping.keys())
                | (set(override_mapping.keys()) if isinstance(override_mapping, dict) else set())
            )
            for child_key in child_keys:
                children.append(
                    self._build_node(
                        path + [child_key],
                        default_mapping.get(child_key),
                        current_mapping.get(child_key),
                        overrides,
                    )
                )

        return {
            "key": path[-1],
            "path": path,
            "label": field.label if field else self._humanize_label(path[-1]),
            "valueType": value_type,
            "description": description,
            "default": deepcopy(default_value),
            "current": deepcopy(current_value),
            "options": options,
            "advanced": advanced,
            "overridden": self._has_override(overrides, path),
            "children": children,
        }

    def _determine_value_type(
        self, field: Optional[ConfigField], default_value: Any, current_value: Any
    ) -> str:
        if field:
            mapping = {
                "string": "string",
                "number": "number",
                "boolean": "boolean",
                "path": "path",
                "list": "list",
                "select": "select",
            }
            if field.field_type in mapping:
                return mapping[field.field_type]
        candidate = current_value if current_value is not None else default_value
        if isinstance(candidate, dict):
            return "object"
        if isinstance(candidate, bool):
            return "boolean"
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return "number"
        if isinstance(candidate, list):
            return "list"
        return "string"

    def _humanize_label(self, key: str) -> str:
        cleaned = key.replace("_", " ").strip()
        if not cleaned:
            return key
        return cleaned[:1].upper() + cleaned[1:]

    def _has_override(self, overrides: Dict[str, Any], path: List[str]) -> bool:
        target: Any = overrides
        for segment in path:
            if not isinstance(target, dict) or segment not in target:
                return False
            target = target[segment]
        return True

    def _value_at(self, source: Dict[str, Any], path: List[str]) -> Any:
        target: Any = source
        for segment in path:
            if not isinstance(target, dict) or segment not in target:
                return {}
            target = target[segment]
        return target

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


def build_segmentation_field_catalog() -> List[ConfigField]:
    """Field metadata for the segmentation (model) configuration."""

    return [
        ConfigField(
            path=["beam_width"],
            section="Beam search",
            label="Beam width",
            field_type="number",
            description="Number of parallel hypotheses explored during segmentation.",
        ),
        ConfigField(
            path=["constraints", "min_block_duration_s"],
            section="Constraints",
            label="Minimum block duration (s)",
            field_type="number",
        ),
        ConfigField(
            path=["constraints", "max_block_duration_s"],
            section="Constraints",
            label="Maximum block duration (s)",
            field_type="number",
        ),
        ConfigField(
            path=["constraints", "line_length_soft_target"],
            section="Constraints",
            label="Line length soft target",
            field_type="number",
        ),
        ConfigField(
            path=["constraints", "line_length_hard_limit"],
            section="Constraints",
            label="Line length hard limit",
            field_type="number",
        ),
        ConfigField(
            path=["constraints", "min_chars_for_single_word_block"],
            section="Constraints",
            label="Min chars for single-word block",
            field_type="number",
            description="Shortest caption length allowed when a block contains only one word.",
        ),
        ConfigField(
            path=["allowed_single_word_proper_nouns"],
            section="Constraints",
            label="Allowed single-word proper nouns",
            field_type="list",
            description="Proper nouns permitted as standalone captions without triggering orphan penalties.",
            advanced=True,
        ),
        ConfigField(
            path=["sliders", "flow"],
            section="Stylistic sliders",
            label="Flow weight",
            field_type="number",
            description="Multiplier for rhythm-focused statistical weights.",
        ),
        ConfigField(
            path=["sliders", "density"],
            section="Stylistic sliders",
            label="Density weight",
            field_type="number",
        ),
        ConfigField(
            path=["sliders", "balance"],
            section="Stylistic sliders",
            label="Balance weight",
            field_type="number",
        ),
        ConfigField(
            path=["sliders", "line_length_leniency"],
            section="Stylistic sliders",
            label="Line length leniency",
            field_type="number",
            description=">1.0 allows longer lines before applying penalties.",
        ),
        ConfigField(
            path=["sliders", "orphan_leniency"],
            section="Stylistic sliders",
            label="Orphan leniency",
            field_type="number",
            description=">1.0 strengthens penalties for orphan words.",
        ),
        ConfigField(
            path=["sliders", "single_word_line_penalty"],
            section="Stylistic sliders",
            label="Single-word line penalty",
            field_type="number",
            description="Penalty applied when a cue would end with a single word or sub-minimal line.",
        ),
        ConfigField(
            path=["sliders", "extreme_balance_penalty"],
            section="Stylistic sliders",
            label="Extreme balance penalty",
            field_type="number",
            description="Penalty applied when line character counts are dramatically imbalanced.",
        ),
        ConfigField(
            path=["sliders", "extreme_balance_threshold"],
            section="Stylistic sliders",
            label="Extreme balance threshold",
            field_type="number",
            description="Character ratio beyond which the extreme balance penalty activates.",
        ),
        ConfigField(
            path=["sliders", "structure_boost"],
            section="Stylistic sliders",
            label="Structure boost",
            field_type="number",
            description="Additive boost for structural cues like speaker changes.",
        ),
        ConfigField(
            path=["paths", "model_weights"],
            section="Model assets",
            label="Model weights path",
            field_type="path",
            description="Relative path to the segmentation model weights JSON.",
        ),
        ConfigField(
            path=["paths", "constraints"],
            section="Model assets",
            label="Constraints path",
            field_type="path",
            description="Relative path to the fallback constraints JSON.",
        ),
    ]

