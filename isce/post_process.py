"""Provides post-processing utilities for refining segmented captions.

This module contains helpers that run after the main beam search to improve
the quality and readability of the final output. These functions are designed
to correct common segmentation issues, such as awkward line breaks or overly
short subtitle cues, by applying a series of heuristic-driven adjustments.
"""
from __future__ import annotations
from dataclasses import replace
from typing import List, Tuple
from .types import Token, BreakType
from .scorer import Scorer


def _block_ranges(tokens: List[Token]) -> List[Tuple[int, int]]:
    """Identifies the start and end indices of each subtitle block."""
    ranges = []
    start_idx = 0
    for i, token in enumerate(tokens):
        if token.break_type == "SB":
            ranges.append((start_idx, i))
            start_idx = i + 1
    return ranges


def _as_dicts(tokens: List[Token], breaks: List[BreakType]) -> List[dict]:
    """Converts a list of Token objects to dictionaries for the scorer."""
    return [replace(t, break_type=b).__dict__ for t, b in zip(tokens, breaks)]


def _block_breaks(block: List[Token]) -> List[BreakType]:
    """Extracts the break types from a list of tokens."""
    return [t.break_type for t in block]


def _rebalance_line_breaks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    """Adjusts line breaks within blocks to improve line balance."""
    output = list(tokens)
    for start, end in _block_ranges(output):
        block = output[start : end + 1]
        breaks = _block_breaks(block)
        if "LB" not in breaks:
            continue

        best_lb_idx = breaks.index("LB")
        current_score = scorer.score_block(_as_dicts(block, breaks), breaks)

        for i in range(len(block) - 1):
            if i == best_lb_idx:
                continue

            new_breaks = ["O"] * len(block)
            new_breaks[i] = "LB"
            new_breaks[-1] = "SB"

            new_score = scorer.score_block(_as_dicts(block, new_breaks), new_breaks)
            if new_score > current_score:
                current_score = new_score
                best_lb_idx = i

        final_breaks = ["O"] * len(block)
        final_breaks[best_lb_idx] = "LB"
        final_breaks[-1] = "SB"

        for i, br in enumerate(final_breaks):
            output[start + i] = replace(output[start + i], break_type=br)

    return output


def _merge_short_blocks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    """Merges short, single-word blocks into adjacent blocks."""
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

            # Prevent merging across speaker changes
            if block1[-1].speaker != block2[0].speaker:
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
    return output

def reflow_tokens(tokens: List[Token], scorer: Scorer) -> List[Token]:
    """
    Applies a series of post-processing steps to refine segmentation.

    This function chains together multiple refinement heuristics to improve
    the final output quality. It first attempts to merge short, isolated
    blocks and then rebalances line breaks within the updated blocks.

    Args:
        tokens: The list of `Token` objects with initial segmentation.
        scorer: The `Scorer` instance to use for evaluating changes.

    Returns:
        A new list of `Token` objects with refined segmentation.
    """
    merged = _merge_short_blocks(tokens, scorer)
    rebalanced = _rebalance_line_breaks(merged, scorer)
    return rebalanced
