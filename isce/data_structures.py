
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Engineered:
    """
    A container for engineered features.

    In the refactored design, this class is a placeholder, as features are
    now read directly from the token dictionaries. It is kept for structural
    compatibility with the `TokenRow` object.
    """
    pass # In the new design, features are read directly from the token dict.

@dataclass
class TokenRow:
    """
    Represents a single decision point between two tokens.

    This container holds the dictionary representation of the current token and
    the next token. This pair represents the boundary or "decision point" where
    a break (`O`, `LB`, or `SB`) could be placed. It is the primary data
    structure passed to the feature engineering and scoring functions during
    training.

    Attributes:
        token: The dictionary for the current token.
        nxt: The dictionary for the subsequent token.
        feats: A placeholder `Engineered` object.
    """
    token: dict
    nxt: Optional[dict]
    feats: Engineered
