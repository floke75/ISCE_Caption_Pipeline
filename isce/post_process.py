"""Post-processing helpers for refining segmentation output."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Sequence

from .config import Config
from .scorer import Scorer
from .types import BreakType, Token

__all__ = ["reflow_tokens"]


def _count_chars(tokens: Sequence[Token]) -> int:
    """Approximate the number of visible characters for a token span."""
    if not tokens:
        return 0
    # Include a space between tokens when estimating the rendered width.
    return sum(len(token.w) for token in tokens) + max(0, len(tokens) - 1)


def _find_block_end(tokens: Sequence[Token], start: int) -> int:
    """Return the index of the final token in the block that starts at ``start``."""
    end = start
    while end < len(tokens) - 1 and tokens[end].break_type != "SB":
        end += 1
    return end


def _block_breaks(tokens: Sequence[Token]) -> List[BreakType]:
    """Return the break types for the provided tokens, defaulting missing values."""
    return [(token.break_type or "O") for token in tokens]


def _tokens_to_dicts(tokens: Iterable[Token]) -> List[dict]:
    """Convert dataclass tokens into dictionaries understood by ``Scorer``."""
    return [dict(token.__dict__) for token in tokens]


def _make_breaks_with_lb(length: int, lb_index: int | None) -> List[BreakType]:
    """Construct a break sequence with an optional line break at ``lb_index``."""
    breaks: List[BreakType] = ["O"] * length
    breaks[-1] = "SB"
    if lb_index is not None and 0 <= lb_index < length - 1:
        breaks[lb_index] = "LB"
    return breaks


def reflow_tokens(tokens: Sequence[Token], scorer: Scorer, cfg: Config) -> List[Token]:
    """Apply lightweight refinements to segmented tokens."""
    if not tokens:
        return []

    refined: List[Token] = list(tokens)
    epsilon = 1e-4
    start = 0

    while start < len(refined):
        end = _find_block_end(refined, start)
        block_tokens = refined[start : end + 1]
        if not block_tokens:
            break

        block_breaks = _block_breaks(block_tokens)
        block_dicts = _tokens_to_dicts(block_tokens)
        block_score = scorer.score_block(block_dicts, block_breaks)

        total_chars = _count_chars(block_tokens)
        duration = max(1e-6, block_tokens[-1].end - block_tokens[0].start)
        is_short = (
            len(block_tokens) <= 2
            or total_chars <= 10
            or duration < max(cfg.min_block_duration_s * 1.25, 1.5)
        )

        lb_idx = next((i for i, br in enumerate(block_breaks) if br == "LB"), -1)
        is_imbalanced = False
        if lb_idx != -1:
            first_line = _count_chars(block_tokens[: lb_idx + 1])
            second_line = _count_chars(block_tokens[lb_idx + 1 :])
            shorter = min(first_line, second_line)
            longer = max(first_line, second_line)
            if shorter == 0:
                is_imbalanced = True
            else:
                ratio = longer / max(1, shorter)
                is_imbalanced = ratio >= 2.0 or shorter <= 5

        # Candidate 1: merge with the next block when the current one feels weak.
        merged = False
        if (is_short or is_imbalanced) and end < len(refined) - 1:
            next_start = end + 1
            next_end = _find_block_end(refined, next_start)
            next_tokens = refined[next_start : next_end + 1]
            if next_tokens:
                next_breaks = _block_breaks(next_tokens)
                next_dicts = _tokens_to_dicts(next_tokens)
                base_score = block_score + scorer.score_block(next_dicts, next_breaks)

                best_bridge: BreakType | None = None
                best_score = base_score

                for bridge in ("O", "LB"):
                    if bridge == "LB" and lb_idx != -1:
                        # Avoid emitting two line breaks within the same block.
                        continue
                    combined_breaks = block_breaks[:-1] + [bridge] + next_breaks
                    combined_tokens = block_tokens + next_tokens
                    combined_dicts = _tokens_to_dicts(combined_tokens)
                    combined_score = scorer.score_block(combined_dicts, combined_breaks)
                    if combined_score > best_score + epsilon:
                        best_score = combined_score
                        best_bridge = bridge

                if best_bridge:
                    refined[end] = replace(refined[end], break_type=best_bridge)
                    block_tokens = refined[start : _find_block_end(refined, start) + 1]
                    block_breaks = _block_breaks(block_tokens)
                    block_dicts = _tokens_to_dicts(block_tokens)
                    block_score = scorer.score_block(block_dicts, block_breaks)
                    merged = True
                    # Re-run the loop for the expanded block.
                    continue

        if not merged and is_imbalanced and lb_idx != -1:
            candidate_positions: List[int] = []
            if lb_idx > 0:
                candidate_positions.append(lb_idx - 1)
            if lb_idx < len(block_tokens) - 2:
                candidate_positions.append(lb_idx + 1)

            best_breaks = block_breaks
            best_score = block_score

            for candidate in candidate_positions:
                new_breaks = _make_breaks_with_lb(len(block_tokens), candidate)
                new_score = scorer.score_block(block_dicts, new_breaks)
                if new_score > best_score + epsilon:
                    best_score = new_score
                    best_breaks = new_breaks

            if best_breaks is not block_breaks and best_breaks != block_breaks:
                for offset, br in enumerate(best_breaks):
                    refined[start + offset] = replace(refined[start + offset], break_type=br)

        start = _find_block_end(refined, start) + 1

    return refined

