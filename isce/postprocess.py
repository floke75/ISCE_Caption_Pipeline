"""Post-processing utilities for segmented caption tokens."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

from .scorer import Scorer
from .types import BreakType, Token


def postprocess(tokens: Sequence[Token], scorer: Scorer) -> List[Token]:
    """Clean up segmentation artifacts after the beam search.

    The beam search occasionally emits blocks that are technically valid but
    awkward for human readers.  This helper inspects each block and applies a
    handful of corrective transformations:

    *   Move a line break when it produces an orphaned single-word line or when
        the two lines are heavily imbalanced.
    *   Merge an ultra-short, single-word block into the following block when
        doing so improves the holistic block scores according to
        :meth:`Scorer.score_block`.

    Args:
        tokens: The segmented token sequence returned by ``segment``.
        scorer: The instantiated scorer used during segmentation.  Only the
            ``score_block`` method is required here.

    Returns:
        A new list of :class:`Token` instances with any adjustments applied.
    """

    adjusted = _rebalance_line_breaks(list(tokens), scorer)
    merged = _merge_short_blocks(adjusted, scorer)
    return _rebalance_line_breaks(merged, scorer)


def _rebalance_line_breaks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    output = list(tokens)
    for start, end in _block_ranges(output):
        block_tokens = output[start : end + 1]
        block_breaks = _block_breaks(block_tokens)

        lb_indices = [idx for idx, br in enumerate(block_breaks[:-1]) if br == "LB"]
        if not lb_indices:
            continue

        stats = _line_stats(block_tokens, block_breaks)
        if not _needs_rebalance(stats):
            continue

        best_breaks = block_breaks
        best_rank, _ = _candidate_rank(block_tokens, block_breaks, scorer)

        candidate_indices = list(range(len(block_tokens) - 1))
        for candidate_lb in candidate_indices:
            if candidate_lb == len(block_tokens) - 1:
                continue
            candidate_breaks = _build_breaks(len(block_tokens), candidate_lb)
            rank, _score = _candidate_rank(block_tokens, candidate_breaks, scorer)
            if rank > best_rank + 1e-6:
                best_rank = rank
                best_breaks = candidate_breaks

        if best_breaks != block_breaks:
            for offset, br in enumerate(best_breaks):
                output[start + offset] = replace(output[start + offset], break_type=br)

    return output


def _merge_short_blocks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    output = list(tokens)
    changed = True
    while changed:
        changed = False
        block_ranges = list(_block_ranges(output))
        for idx in range(len(block_ranges) - 1):
            start1, end1 = block_ranges[idx]
            start2, end2 = block_ranges[idx + 1]

            block1 = output[start1 : end1 + 1]
            block2 = output[start2 : end2 + 1]

            if len(block1) != 1:
                continue

            word = block1[0].w.rstrip(".,?!¡¿…")
            if len(word) > 6:
                continue

            current_score = (
                scorer.score_block(_as_dicts(block1, _block_breaks(block1)), _block_breaks(block1))
                + scorer.score_block(_as_dicts(block2, _block_breaks(block2)), _block_breaks(block2))
            )

            combined_tokens = block1 + block2
            combined_breaks: List[BreakType] = []
            for offset, token in enumerate(combined_tokens):
                if offset == len(combined_tokens) - 1:
                    combined_breaks.append("SB")
                else:
                    br = token.break_type or "O"
                    combined_breaks.append("O" if br == "SB" else br)

            combined_score = scorer.score_block(
                _as_dicts(combined_tokens, combined_breaks), combined_breaks
            )

            if combined_score >= current_score - 1e-6:
                for offset, br in enumerate(combined_breaks):
                    output[start1 + offset] = replace(output[start1 + offset], break_type=br)
                changed = True
                break
        # Loop again if we modified any block boundaries.
    return output


def _block_ranges(tokens: Sequence[Token]) -> Iterable[Tuple[int, int]]:
    start = 0
    n = len(tokens)
    while start < n:
        end = start
        while end < n - 1 and tokens[end].break_type != "SB":
            end += 1
        yield (start, end)
        start = end + 1


def _block_breaks(block_tokens: Sequence[Token]) -> List[BreakType]:
    breaks: List[BreakType] = []
    for idx, token in enumerate(block_tokens):
        br: BreakType = token.break_type or ("SB" if idx == len(block_tokens) - 1 else "O")
        if idx == len(block_tokens) - 1:
            br = "SB"
        breaks.append(br)
    return breaks


def _build_breaks(length: int, lb_index: Optional[int]) -> List[BreakType]:
    breaks: List[BreakType] = []
    for idx in range(length):
        if idx == length - 1:
            breaks.append("SB")
        elif lb_index is not None and idx == lb_index:
            breaks.append("LB")
        else:
            breaks.append("O")
    return breaks


def _line_stats(block_tokens: Sequence[Token], block_breaks: Sequence[BreakType]) -> Tuple[List[int], List[int]]:
    words_per_line: List[int] = []
    chars_per_line: List[int] = []
    current_line: List[Token] = []

    for token, br in zip(block_tokens, block_breaks):
        current_line.append(token)
        if br in {"LB", "SB"}:
            words_per_line.append(len(current_line))
            chars_per_line.append(_count_chars(current_line))
            current_line = []

    if current_line:
        words_per_line.append(len(current_line))
        chars_per_line.append(_count_chars(current_line))

    return words_per_line, chars_per_line


def _needs_rebalance(stats: Tuple[List[int], List[int]]) -> bool:
    words, chars = stats
    if len(words) < 2:
        return False
    if min(words) <= 1:
        return True
    if min(chars) == 0:
        return True
    ratio = max(chars) / max(1, min(chars))
    return ratio > 2.0 or abs(chars[0] - chars[1]) > 15


def _candidate_rank(
    block_tokens: Sequence[Token], block_breaks: Sequence[BreakType], scorer: Scorer
) -> Tuple[float, float]:
    block_dicts = _as_dicts(block_tokens, block_breaks)
    score = scorer.score_block(block_dicts, list(block_breaks))
    words, chars = _line_stats(block_tokens, block_breaks)

    imbalance_penalty = 0.0
    if len(chars) >= 2:
        if min(chars) == 0:
            imbalance_penalty = 5.0
        else:
            imbalance_penalty = max(0.0, (max(chars) / min(chars)) - 1.0)

    orphan_penalty = 0.0
    if len(words) >= 2 and min(words) <= 1:
        orphan_penalty = 1.0

    rank = score - (0.2 * imbalance_penalty) - orphan_penalty
    return rank, score


def _as_dicts(tokens: Sequence[Token], breaks: Sequence[BreakType]) -> List[dict]:
    out: List[dict] = []
    for token, br in zip(tokens, breaks):
        token_dict = dict(token.__dict__)
        token_dict["break_type"] = br
        out.append(token_dict)
    return out


def _count_chars(token_slice: Sequence[Token]) -> int:
    if not token_slice:
        return 0
    return sum(len(t.w) for t in token_slice) + max(0, len(token_slice) - 1)

