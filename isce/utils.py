from __future__ import annotations
from typing import List
from .types import Token, TokenRow
from .config import Config

def _token_to_row_dict(token: Token, reverse: bool = False) -> dict:
    """Converts a Token object to a dictionary for use in a TokenRow."""
    return token.__dict__

def _build_token_rows(tokens: List[Token], cfg: Config) -> List[TokenRow]:
    """Materialize :class:`TokenRow` entries for every decision point."""

    rows: List[TokenRow] = []
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        lookahead = _get_lookahead_slice(tokens, i + 1, cfg.lookahead_width)
        rows.append(
            TokenRow(
                token=_token_to_row_dict(token),
                nxt=_token_to_row_dict(nxt) if nxt else None,
                lookahead=lookahead,
            )
        )
    return rows

def _get_lookahead_slice(tokens: List[Token], start_idx: int, width: int) -> tuple[dict, ...]:
    """Extracts a slice of future tokens for lookahead heuristics."""
    if width == 0:
        return ()

    slice_end = start_idx
    while slice_end < len(tokens) and tokens[slice_end].break_type != "SB":
        slice_end += 1
        if slice_end - start_idx >= width:
            break

    if slice_end <= start_idx:
        return ()

    return tuple(_token_to_row_dict(t) for t in tokens[start_idx:slice_end])
