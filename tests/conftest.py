"""Test configuration helpers for ensuring local imports resolve."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _stub_yaml_if_missing() -> None:
    if importlib.util.find_spec("yaml") is not None:
        return

    module = ModuleType("yaml")

    class YAMLError(Exception):
        """Fallback error matching PyYAML's exception hierarchy."""

    def safe_load(stream):  # type: ignore[override]
        data = stream.read() if hasattr(stream, "read") else stream
        if not data:
            return None
        return json.loads(data)

    module.YAMLError = YAMLError
    module.safe_load = safe_load
    sys.modules["yaml"] = module


def _stub_tqdm_if_missing() -> None:
    if importlib.util.find_spec("tqdm") is not None:
        return

    module = ModuleType("tqdm")

    def _identity(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    module.tqdm = _identity
    sys.modules["tqdm"] = module


def _stub_numpy_if_missing() -> None:
    if importlib.util.find_spec("numpy") is not None:
        return

    module = ModuleType("numpy")

    def percentile(data, q):
        if not data:
            raise ValueError("percentile() requires a non-empty dataset")
        sorted_data = sorted(data)
        if not 0 <= q <= 100:
            raise ValueError("q must be between 0 and 100")
        if len(sorted_data) == 1:
            return float(sorted_data[0])
        # Simple linear interpolation akin to NumPy's default behaviour.
        position = (len(sorted_data) - 1) * (q / 100)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        lower = float(sorted_data[lower_index])
        upper = float(sorted_data[upper_index])
        fraction = position - lower_index
        return lower + (upper - lower) * fraction

    module.percentile = percentile
    sys.modules["numpy"] = module


def _stub_pandas_if_missing() -> None:
    if importlib.util.find_spec("pandas") is not None:
        return

    module = ModuleType("pandas")
    module.DataFrame = object  # type: ignore[attr-defined]
    module.Series = object  # type: ignore[attr-defined]
    sys.modules["pandas"] = module


_ensure_project_root_on_path()
_stub_yaml_if_missing()
_stub_tqdm_if_missing()
_stub_numpy_if_missing()
_stub_pandas_if_missing()
