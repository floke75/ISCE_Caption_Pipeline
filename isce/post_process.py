"""Post-processing helpers for refining segmentation output.

The functions in this module implement inexpensive, local heuristics that can be
applied after the primary beam-search segmentation has produced block
boundaries. They avoid re-running expensive search passes while still letting us
clean up artifacts such as very short cues or lopsided multi-line blocks.

At the time of writing, :func:`reflow_tokens` is intentionally conservative: it
only ever merges with the immediate next block or shuffles an existing line
break within a block. These changes are designed to be reversible and
interpretable, making them safe to run as an optional post-processing step.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import Config
from .scorer import Scorer
from .types import BreakType, Token

__all__ = ["reflow_tokens"]


def reflow_tokens(tokens: Sequence[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Apply lightweight refinements to segmented tokens.

    The routine scans each block in order and evaluates two adjustments:

    * **Merge short blocks.** When a block is unusually short (few tokens,
      characters, or seconds of duration) we evaluate whether gluing it to the
      following block increases the scorer's block-level score. If so, we
      rewrite the break type at the boundary to ``"O"`` or ``"LB"``.
    * **Rebalance line breaks.** If a block already contains a manual line
      break, we probe nearby positions to see whether moving the break yields a
      higher score—especially helpful when one line is lopsidedly short.

    Parameters
    ----------
    tokens:
        Segmented tokens emitted by the beam search.
    scorer:
        A ``Scorer`` instance that provides block-level preferences.
    cfg:
        The loaded :class:`~isce.config.Config`. We use ``min_block_duration_s``
        to detect suspiciously short cues.

    Returns
    -------
    list[Token]
        A defensive copy of ``tokens`` with any accepted refinements applied.
    """

    if not tokens:
        return []

    refined: List[Token] = list(tokens)
    refined = _merge_short_blocks(refined, scorer, cfg)
    refined = _rebalance_line_breaks(refined, scorer)
    return refined


def _merge_short_blocks(tokens: List[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    refined = list(tokens)
    epsilon = 1e-4

    changed = True
    while changed:
        changed = False
        block_ranges = list(_block_ranges(refined))
        for idx, (start1, end1) in enumerate(block_ranges[:-1]):
            start2, end2 = block_ranges[idx + 1]

            block1 = refined[start1 : end1 + 1]
            block2 = refined[start2 : end2 + 1]

            if not _should_merge_block(block1, cfg):
                continue

            boundary_token = block1[-1]
            next_token = block2[0]
            speakers_differ = (
                boundary_token.speaker is not None
                and next_token.speaker is not None
                and boundary_token.speaker != next_token.speaker
            )
            if getattr(boundary_token, "speaker_change", False) or speakers_differ:
                continue

            block1_breaks = _block_breaks(block1)
            block2_breaks = _block_breaks(block2)
            base_score = scorer.score_block(_as_dicts(block1, block1_breaks), block1_breaks)
            base_score += scorer.score_block(_as_dicts(block2, block2_breaks), block2_breaks)

            best_bridge: Optional[BreakType] = None
            best_score = base_score

            for bridge in ("O", "LB"):
                if bridge == "LB" and "LB" in block1_breaks:
                    continue

                combined_breaks = block1_breaks[:-1] + [bridge] + block2_breaks
                combined_tokens = block1 + block2
                combined_score = scorer.score_block(
                    _as_dicts(combined_tokens, combined_breaks), combined_breaks
                )

                if combined_score > best_score + epsilon:
                    best_score = combined_score
                    best_bridge = bridge

            if best_bridge:
                combined_breaks = block1_breaks[:-1] + [best_bridge] + block2_breaks
                for offset, br in enumerate(combined_breaks):
                    refined[start1 + offset] = replace(refined[start1 + offset], break_type=br)
                changed = True
                break

    return refined


def _rebalance_line_breaks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    refined = list(tokens)
    epsilon = 1e-4

    for start, end in _block_ranges(refined):
        block_tokens = refined[start : end + 1]
        block_breaks = _block_breaks(block_tokens)

        if "LB" not in block_breaks:
            continue

        stats = _line_stats(block_tokens, block_breaks)
        if not _needs_rebalance(stats):
            continue

        best_breaks = block_breaks
        best_rank, _ = _candidate_rank(block_tokens, block_breaks, scorer)

        for candidate_lb in range(len(block_tokens) - 1):
            candidate_breaks = _build_breaks(len(block_tokens), candidate_lb)
            rank, _ = _candidate_rank(block_tokens, candidate_breaks, scorer)
            if rank > best_rank + epsilon:
                best_rank = rank
                best_breaks = candidate_breaks

        if best_breaks != block_breaks:
            for offset, br in enumerate(best_breaks):
                refined[start + offset] = replace(refined[start + offset], break_type=br)

    return refined


def _should_merge_block(block_tokens: Sequence[Token], cfg: Config) -> bool:
    if not block_tokens:
        return False

    total_chars = _count_chars(block_tokens)
    duration = max(1e-6, block_tokens[-1].end - block_tokens[0].start)

    word = block_tokens[0].w.rstrip(".,?!¡¿…")
    is_single_word = len(block_tokens) == 1 and len(word) <= 6

    return (
        len(block_tokens) <= 2
        or total_chars <= 10
        or duration < max(cfg.min_block_duration_s * 1.25, 1.5)
        or is_single_word
    )


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
        if idx == len(block_tokens) - 1:
            breaks.append("SB")
        else:
            br: BreakType = token.break_type or "O"
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
    return ratio >= 1.8


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


def _count_chars(tokens: Sequence[Token]) -> int:
    if not tokens:
        return 0
    return sum(len(token.w) for token in tokens) + max(0, len(tokens) - 1)
