from __future__ import annotations
from typing import List, Dict
from .types import Token, TokenRow, BreakType
from .scorer import Scorer
from .config import Config

def _token_to_row_dict(token: Token, reverse: bool = False) -> dict:
    """Converts a Token object to a dictionary for use in a TokenRow."""
    return token.__dict__

def _compute_transition_scores(tokens: List[Token], scorer: Scorer, cfg: Config) -> Dict[int, Dict[str, float]]:
    """Pre-computes and caches transition scores for all tokens."""
    scores = {}
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        row = TokenRow(
            token=_token_to_row_dict(token),
            nxt=_token_to_row_dict(nxt) if nxt else None,
            lookahead=_get_lookahead_slice(tokens, i + 1, cfg.lookahead_width),
        )
        scores[i] = scorer.score_transition(row)
    return scores

def _get_lookahead_slice(tokens: List[Token], start_idx: int, width: int) -> List[dict]:
    """Extracts a slice of future tokens for lookahead heuristics."""
    if width == 0:
        return []

    slice_end = start_idx
    while slice_end < len(tokens) and tokens[slice_end].break_type != "SB":
        slice_end += 1
        if slice_end - start_idx >= width:
            break

    return [_token_to_row_dict(t) for t in tokens[start_idx:slice_end]]
