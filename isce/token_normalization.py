"""Helpers for normalising token payloads across the pipeline."""
from __future__ import annotations

from typing import Any, Optional

from .types import Token


def normalize_token_payload(
    token: Optional[Token | dict[str, Any]], idx: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """Return a scorer-friendly dictionary with stable indexes and strings.

    This helper accepts either :class:`~isce.types.Token` instances or the raw
    dictionaries saved in JSON artifacts.  It produces a shallow copy so callers
    can freely mutate the payload without affecting the original object.

    ``token_index`` is normalised to an ``int`` whenever present; when absent or
    not coercible we fall back to the optional ``idx`` provided by the caller so
    dependency-aware feature helpers can derive repeatable keys (for example
    ``head_position_key`` and ``dependency_link_key``).

    The ``w`` field is coerced to ``str`` when a legacy payload stores it as a
    non-string type (for example numbers coming from pandas CSV exports).
    """

    if token is None:
        return None

    if isinstance(token, dict):
        payload: dict[str, Any] = {k: v for k, v in token.items()}
    else:
        payload = dict(token.__dict__)

    token_index = payload.get("token_index")
    if token_index is not None:
        try:
            payload["token_index"] = int(token_index)
        except (TypeError, ValueError):
            payload["token_index"] = None

    if payload.get("token_index") is None and idx is not None:
        payload["token_index"] = int(idx)

    if "w" in payload and payload["w"] is not None and not isinstance(payload["w"], str):
        payload["w"] = str(payload["w"])

    return payload
