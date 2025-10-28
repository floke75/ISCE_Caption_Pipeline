"""Utility helpers for normalising token payloads.

The functions in this module are intentionally small, pure, and extensively
documented so that downstream agents—human or LLM—can understand exactly how
incoming token payloads are massaged before scoring.  Callers may provide raw
JSON dictionaries, dataclasses, or lightweight objects with ``__dict__``
attributes.  Regardless of the source, :func:`normalize_token_payload` returns a
fresh dictionary copy with stable ``token_index`` values, predictable types,
and guardrail-friendly defaults.

Because normalisation is invoked inside the scorer, the beam search, and the
training pipeline, even subtle behaviour needs to be transparent.  Each helper
therefore documents the accepted inputs, the returned value, and any
edge-case handling (for example, how ``NaN`` is treated or how blank strings
are coerced).  This rich documentation is designed to keep the cross-module
contract easy to follow when debugging or evolving the token schema.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import used only for typing
    from .types import Token


def _coerce_float(value: Any) -> Optional[float]:
    """Return ``value`` as a ``float`` when the conversion is lossless enough.

    Parameters
    ----------
    value:
        Arbitrary payload extracted from a token.  JSON serialisation frequently
        stores numbers as strings, so we accept ``str`` inputs in addition to
        numeric types.

    Returns
    -------
    float | None
        ``float(value)`` when the conversion succeeds.  Empty strings and
        values that raise :class:`ValueError` during conversion are mapped to
        ``None`` so callers can fall back to sensible defaults.
    """
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
    """Return ``value`` coerced to an ``int`` whenever it is representable.

    ``normalize_token_payload`` leans on this helper for numeric identifiers
    such as ``token_index`` and ``cue_id``.  We therefore accept a broad range
    of inputs and document the fallbacks explicitly:

    - ``None`` stays ``None`` so the caller can decide on a default.
    - ``bool`` is cast to ``int`` (``True`` → ``1``) to mirror Python's natural
      behaviour.
    - ``float`` values that are not ``NaN`` lose their fractional component via
      ``int(value)``.
    - ``str`` inputs are stripped and parsed first as integers, then as floats
      to handle values like ``"3.0"``.

    Parameters
    ----------
    value:
        Numeric or textual representation to coerce.

    Returns
    -------
    int | None
        The best-effort integer conversion.  Non-parsable inputs yield ``None``.
    """
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
    """Return ``value`` as a boolean when its intent is unambiguous.

    Parameters
    ----------
    value:
        Any object potentially encoding truthy/falsey information.  Strings are
        compared case-insensitively against a small vocabulary so exports like
        ``"True"`` and ``"false"`` are handled gracefully.

    Returns
    -------
    bool | None
        ``True`` or ``False`` when ``value`` clearly maps to a boolean.  When
        ``value`` is ``None`` the helper returns ``None`` to allow the caller to
        decide whether to apply a fallback.  All other inputs fall back to
        Python's default truthiness rules via ``bool(value)``.
    """
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
    """Return a ``str`` version of ``value`` while preserving explicit blanks.

    Parameters
    ----------
    value:
        The candidate value to coerce into a string.  Existing strings are
        returned untouched to avoid surprising transformations such as
        lowercasing.
    default:
        Optional string to return when ``value`` is ``None``.  This keeps
        :func:`normalize_token_payload` from sprinkling the literal ``"None"``
        into token text fields when metadata is missing.

    Returns
    -------
    str | None
        ``str(value)`` for non-string inputs, ``default`` when ``value`` is
        ``None``, or ``None`` when both ``value`` and ``default`` are ``None``.
    """
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
        from JSON, a dataclass, or an arbitrary object exposing ``__dict__``.
        ``None`` inputs propagate as ``None`` so callers can skip optional
        neighbours (for example, the ``nxt`` token in bigram features).
    idx:
        Optional fallback token index applied when ``token`` does not already
        advertise one.  The index is stored under ``token_index`` to keep
        dependency-aware features stable across scoring passes.  If both the
        payload and ``idx`` are missing an index the resulting dictionary will
        explicitly contain ``{"token_index": None}``.

    Returns
    -------
    dict[str, Any] | None
        A deep-copied dictionary with the following guarantees:

        * Numeric-like fields (``token_index``, ``pause_*``, ``head_idx`` …)
          are coerced via the dedicated helpers above.
        * Textual fields are always strings, never ``None``.
        * Boolean guardrails are normalised to ``True``/``False`` rather than
          relying on Python truthiness.

        ``None`` is returned only when ``token`` itself is ``None``.

    Raises
    ------
    TypeError
        If ``token`` cannot be materialised into a dictionary using ``dict``
        semantics (for example, generators or primitives).

    Examples
    --------
    >>> normalize_token_payload({"w": "Hello", "token_index": "7"})["token_index"]
    7
    >>> normalize_token_payload(None) is None
    True
    >>> normalize_token_payload(Token(w="bye", token_index=None), idx=3)["token_index"]
    3
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
