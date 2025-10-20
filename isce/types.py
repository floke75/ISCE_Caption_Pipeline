# C:\dev\Captions_Formatter\Formatter_machine\isce\types.py

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Literal, Optional

BreakType = Literal["O", "LB", "SB"]

@dataclass(frozen=True)
class Token:
    """
    Represents a single token with all its associated metadata from the
    consolidated enrichment pipeline. This is the central data contract.
    """
    w: str
    start: float
    end: float
    speaker: Optional[str]
    
    # Features from the enrichment script
    cue_id: Optional[int] = None
    is_sentence_initial: bool = False
    is_sentence_final: bool = False
    pause_after_ms: int = 0
    pause_before_ms: int = 0
    pause_z: float = 0.0
    pos: Optional[str] = None
    lemma: Optional[str] = None
    tag: Optional[str] = None
    morph: Optional[str] = None
    dep: Optional[str] = None
    head_idx: Optional[int] = None
    
    # Heuristic flags
    starts_with_dialogue_dash: bool = False
    speaker_change: bool = False
    num_unit_glue: bool = False
    is_llm_structural_break: bool = False
    
    # The final decision label, assigned by the segmenter
    break_type: Optional[BreakType] = None

    @classmethod
    def get_field_names(cls) -> set[str]:
        """Returns a set of the field names for this dataclass."""
        return {f.name for f in fields(cls)}

# The TokenRow and Engineered classes are no longer defined here.
# They have been retired or are defined locally in the scripts that need them.