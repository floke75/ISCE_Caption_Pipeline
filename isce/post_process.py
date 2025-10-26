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
    ranges: List[Tuple[int, int]] = []
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


def _line_balance_cost(block: List[Token], lb_idx: int) -> float:
    """Returns the absolute character difference between the two lines."""
    if lb_idx < 0 or lb_idx >= len(block) - 1:
        return float("inf")

    def _count_chars(slice_tokens: List[Token]) -> int:
        if not slice_tokens:
            return 0
        return sum(len(t.w) for t in slice_tokens) + max(0, len(slice_tokens) - 1)

    line1 = block[: lb_idx + 1]
    line2 = block[lb_idx + 1 :]
    return abs(_count_chars(line1) - _count_chars(line2))


def _rebalance_line_breaks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    """Adjusts line breaks within blocks to improve line balance."""
    output = list(tokens)
    for start, end in _block_ranges(output):
        block = output[start : end + 1]
        breaks = _block_breaks(block)
        if "LB" not in breaks:
            continue

        best_lb_idx = breaks.index("LB")
        original_lb_idx = best_lb_idx
        current_score = scorer.score_block(_as_dicts(block, breaks), breaks)
        best_balance = _line_balance_cost(block, best_lb_idx)
        improved = False
        balance_threshold = 2.0
        early_move_min_gain = 1.25

        for i in range(len(block) - 1):
            if i == best_lb_idx:
                continue

            new_breaks = ["O"] * len(block)
            new_breaks[i] = "LB"
            new_breaks[-1] = "SB"

            new_score = scorer.score_block(_as_dicts(block, new_breaks), new_breaks)
            gain = new_score - current_score
            if gain > 1e-6:
                if i < original_lb_idx and gain <= early_move_min_gain:
                    continue
                current_score = new_score
                best_lb_idx = i
                best_balance = _line_balance_cost(block, best_lb_idx)
                improved = True
            elif not improved and abs(new_score - current_score) <= 1e-6:
                balance = _line_balance_cost(block, i)
                if (best_balance - balance) > balance_threshold:
                    best_lb_idx = i
                    best_balance = balance

        final_breaks = ["O"] * len(block)
        final_breaks[best_lb_idx] = "LB"
        final_breaks[-1] = "SB"

        for i, br in enumerate(final_breaks):
            output[start + i] = replace(output[start + i], break_type=br)

    return output


def _merge_short_blocks(tokens: List[Token], scorer: Scorer) -> List[Token]:
    """Merges short, single-word blocks into adjacent blocks."""
    output = list(tokens)

    def _score_block(tokens_slice: List[Token]) -> float:
        breaks = _block_breaks(tokens_slice)
        return scorer.score_block(_as_dicts(tokens_slice, breaks), breaks)

    def _combined_breaks(block_a: List[Token], block_b: List[Token]) -> List[BreakType]:
        combined: List[BreakType] = []
        for offset, token in enumerate(block_a + block_b):
            if offset == len(block_a + block_b) - 1:
                combined.append("SB")
            else:
                br = token.break_type or "O"
                combined.append("O" if br == "SB" else br)
        return combined

    changed = True
    while changed:
        changed = False
        block_ranges = list(_block_ranges(output))

        for idx, (start, end) in enumerate(block_ranges):
            block = output[start : end + 1]

            if len(block) != 1:
                continue

            word = block[0].w.rstrip(".,?!¡¿…")
            if len(word) > 6:
                continue

            neighbors: List[Tuple[str, Tuple[int, int], List[Token]]] = []
            if idx > 0:
                prev_range = block_ranges[idx - 1]
                prev_block = output[prev_range[0] : prev_range[1] + 1]
                if prev_block[-1].speaker == block[0].speaker:
                    neighbors.append(("prev", prev_range, prev_block))
            if idx + 1 < len(block_ranges):
                next_range = block_ranges[idx + 1]
                next_block = output[next_range[0] : next_range[1] + 1]
                if block[-1].speaker == next_block[0].speaker:
                    neighbors.append(("next", next_range, next_block))

            if not neighbors:
                continue

            block_score = _score_block(block)
            best_choice: Tuple[str, Tuple[int, int], List[BreakType], float] | None = None

            for direction, rng, neighbor in neighbors:
                neighbor_score = _score_block(neighbor)
                if direction == "prev":
                    combined_tokens = neighbor + block
                    start_idx = rng[0]
                else:
                    combined_tokens = block + neighbor
                    start_idx = start

                combined_breaks = _combined_breaks(neighbor if direction == "prev" else block,
                                                   block if direction == "prev" else neighbor)
                combined_score = scorer.score_block(
                    _as_dicts(combined_tokens, combined_breaks), combined_breaks
                )
                baseline = neighbor_score + block_score
                if combined_score >= baseline - 1e-6:
                    if not best_choice or combined_score > best_choice[3] + 1e-6:
                        best_choice = (direction, (start_idx, start_idx + len(combined_tokens) - 1), combined_breaks, combined_score)

            if not best_choice:
                continue

            direction, (merged_start, merged_end), final_breaks, _ = best_choice
            for offset, br in enumerate(final_breaks):
                output[merged_start + offset] = replace(output[merged_start + offset], break_type=br)
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
