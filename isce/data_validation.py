# C:\dev\Captions_Formatter\Formatter_machine\isce\data_validation.py

from __future__ import annotations
from typing import List, Iterator, Dict, Any, Tuple
from .types import Token
from .config import Config

def iter_blocks(tokens: List[Token]) -> Iterator[Tuple[int, int]]:
    """
    Yields (start_idx, end_idx_inclusive) for each block in a token list.
    Blocks are defined by the 'SB' break_type or the end of the list.
    """
    if not tokens:
        return
    
    start_idx = 0
    for i, token in enumerate(tokens):
        if token.break_type == "SB":
            yield (start_idx, i)
            start_idx = i + 1
            
    if start_idx < len(tokens):
        yield (start_idx, len(tokens) - 1)

def validate(tokens: List[Token], cfg: Config) -> Dict[str, Any]:
    """
    Performs sanity and logic checks on a list of labeled tokens.
    This version is updated for the new Token data structure.
    """
    issues = []

    # 1. Per-token time sanity checks
    for i, t in enumerate(tokens):
        if t.end < t.start:
            issues.append({
                "type": "time_order_error",
                "idx": i,
                "message": f"Token '{t.w}' has end time {t.end} before start time {t.start}."
            })
        if t.pause_after_ms is not None and t.pause_after_ms < 0:
            issues.append({
                "type": "negative_pause_error",
                "idx": i,
                "message": f"Token '{t.w}' has negative pause_after_ms: {t.pause_after_ms}."
            })

    # 2. Per-block structural and duration checks
    for start_idx, end_idx in iter_blocks(tokens):
        block_slice = tokens[start_idx : end_idx + 1]
        
        duration = block_slice[-1].end - block_slice[0].start
        if duration < cfg.min_block_duration_s:
            issues.append({
                "type": "short_block_warning",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration": round(duration, 3),
                "message": f"Block from index {start_idx} to {end_idx} is shorter ({duration:.3f}s) than min_duration ({cfg.min_block_duration_s}s)."
            })

        lb_count = sum(1 for token in block_slice if token.break_type == "LB")
        if lb_count > 1:
            issues.append({
                "type": "too_many_line_breaks_error",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "lb_count": lb_count,
                "message": f"Block from index {start_idx} to {end_idx} has {lb_count} line breaks (max is 1)."
            })

    return {"issue_count": len(issues), "issues": issues}