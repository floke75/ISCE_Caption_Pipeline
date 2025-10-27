"""Utility helpers for normalising token payloads.

This module centralises routines that coerce raw token dictionaries or
dataclass instances into the normalised dictionary representation consumed by
both the scorer and the model builder.  The helpers defensively convert mixed
string/number types that appear in JSON exports, ensure ``token_index`` is
stable across scoring paths, and avoid mutating the original objects so callers
can safely reuse cached token structures.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used only for typing
    from .types import Token


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort conversion of ``value`` to ``float``."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> Optional[int]:
    """Best-effort conversion of ``value`` to ``int``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value == value:  # NaN
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None
    return None


def _coerce_bool(value: Any) -> Optional[bool]:
    """Best-effort conversion of ``value`` to ``bool``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "1", "yes", "y"}:
            return True
        if text in {"false", "f", "0", "no", "n", ""}:
            return False
    return bool(value)


def _ensure_text(value: Any, default: Optional[str] = None) -> Optional[str]:
    """Return a textual representation while preserving ``None`` when desired."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def normalize_token_payload(token: Optional[Any], idx: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Return a scorer-ready dictionary copy of ``token``.

    Parameters
    ----------
    token:
        Either a :class:`~isce.types.Token` instance, a raw dictionary loaded
        from JSON, or ``None``.
    idx:
        Optional fallback token index applied when ``token`` does not already
        advertise one.  The index is stored under ``token_index`` to keep
        dependency-aware features stable across scoring passes.
    """

    if token is None:
        return None

    if isinstance(token, dict):
        payload: dict[str, Any] = deepcopy(token)
    elif is_dataclass(token):  # works for Token dataclass and friends
        payload = asdict(token)
    elif hasattr(token, "__dict__"):
        payload = deepcopy(token.__dict__)
    else:
        raise TypeError(f"Unsupported token payload type: {type(token)!r}")

    # --- Numeric coercion ---
    token_index = _coerce_int(payload.get("token_index"))
    fallback_index = _coerce_int(idx)
    payload["token_index"] = token_index if token_index is not None else fallback_index

    for field in ("cue_id", "head_idx", "cue_line_index"):
        if field in payload:
            payload[field] = _coerce_int(payload.get(field))

    for field in ("pause_before_ms", "pause_after_ms"):
        if field in payload:
            coerced = _coerce_int(payload.get(field))
            payload[field] = coerced if coerced is not None else 0

    for field in ("start", "end", "pause_z", "relative_position"):
        if field in payload:
            coerced_float = _coerce_float(payload.get(field))
            payload[field] = coerced_float

    # --- Text coercion ---
    payload["w"] = _ensure_text(payload.get("w"), "")
    for field in ("lemma", "tag", "morph", "dep", "speaker", "asr_source_word"):
        if field in payload and payload[field] is not None:
            payload[field] = _ensure_text(payload[field])

    # --- Boolean coercion ---
    for field in (
        "speaker_change",
        "starts_with_dialogue_dash",
        "num_unit_glue",
        "is_llm_structural_break",
        "is_dangling_eos",
        "line_break_after",
        "is_last_in_cue",
    ):
        if field in payload:
            coerced_bool = _coerce_bool(payload.get(field))
            if coerced_bool is not None:
                payload[field] = coerced_bool

    return payload


__all__ = ["normalize_token_payload"]
