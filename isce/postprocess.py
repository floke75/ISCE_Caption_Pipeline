"""Compatibility wrapper for legacy post-processing helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence

from .config import Config
from .post_process import reflow_tokens
from .scorer import Scorer
from .types import Token

__all__ = ["postprocess"]


def _default_config() -> Config:
    """Materialize a lightweight default config for standalone post-processing."""

    return Config(
        beam_width=7,
        min_block_duration_s=1.0,
        max_block_duration_s=8.0,
        line_length_constraints={
            "line1": {"soft_target": 37, "hard_limit": 42},
            "line2": {"soft_target": 37, "hard_limit": 42},
        },
        min_chars_for_single_word_block=10,
        sliders={},
        paths={},
    )


def postprocess(tokens: Sequence[Token], scorer: Scorer, cfg: Optional[Config] = None) -> List[Token]:
    """Run the modern reflow pass while honoring the legacy helper signature."""

    config = cfg or _default_config()
    return reflow_tokens(tokens, scorer, config)
