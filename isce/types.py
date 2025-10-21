# C:\dev\Captions_Formatter\Formatter_machine\isce\types.py

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any, Literal, Optional

__all__ = ["BreakType", "Token", "TokenRow", "Engineered"]

BreakType = Literal["O", "LB", "SB"]

@dataclass(frozen=True)
class Token:
    """
    Represents a single word token with its associated linguistic and timing data.

    This dataclass is the central data structure used throughout the segmentation
    process. It holds not only the word itself but also a rich set of features
    engineered by the `build_training_pair_standalone.py` script.

    Attributes:
        w: The word text itself.
        start: The start time of the word in seconds.
        end: The end time of the word in seconds.
        speaker: The speaker label assigned by the diarization process.
        cue_id: (Training only) The ID of the ground-truth SRT cue this token belongs to.
        is_sentence_initial: True if the token is likely the start of a sentence.
        is_sentence_final: True if the token is likely the end of a sentence.
        pause_after_ms: The duration of the pause following the token in milliseconds.
        pause_before_ms: The duration of the pause preceding the token in milliseconds.
        pause_z: The z-score of the pause duration, for statistical normalization.
        pos: The part-of-speech tag (e.g., 'NOUN', 'VERB').
        lemma: The base form of the word.
        tag: A fine-grained part-of-speech tag.
        morph: Detailed morphological features.
        dep: The syntactic dependency relation.
        head_idx: The index of the syntactic head of this token.
        starts_with_dialogue_dash: True if the next token starts with a dialogue dash.
        speaker_change: True if a speaker change occurs immediately after this token.
        num_unit_glue: True if the token is a number followed by a unit (e.g., '5 kg').
        is_llm_structural_break: True if an LLM or structural hint suggested a break.
        is_dangling_eos: True if the token ends a sentence but the following token
            does not look like a fresh sentence (short pause, no capitalization).
        relative_position: Normalised index of the token within its sentence (0-1).
        asr_source_word: The raw word text originating from the ASR alignment.
        cue_line_index: (Training) Zero-based index of the line inside the cue.
        line_break_after: (Training) True when the cue inserts a newline after this token.
        is_last_in_cue: (Training) True when the token closes the cue.
        break_type: The final segmentation decision ('O', 'LB', 'SB'), assigned by
                    the beam search algorithm.
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
    is_dangling_eos: bool = False
    relative_position: float = 0.0
    asr_source_word: Optional[str] = None
    cue_line_index: Optional[int] = None
    line_break_after: bool = False
    is_last_in_cue: bool = False

    # The final decision label, assigned by the segmenter
    break_type: Optional[BreakType] = None

    @classmethod
    def get_field_names(cls) -> set[str]:
        """
        Returns a set of all field names for the Token dataclass.

        This method is a utility for conveniently accessing the defined
        attributes of the class, which can be useful for validation or
        serialization purposes.

        Returns:
            A set of strings, where each string is a field name.
        """
        return {f.name for f in fields(cls)}

@dataclass(frozen=True)
class TokenRow:
    """A lightweight container describing a token boundary."""

    token: dict[str, Any]
    nxt: Optional[dict[str, Any]]
    feats: Any = None


@dataclass(frozen=True)
class Engineered:
    """Compatibility shim for pre-engineered feature payloads."""

    # In the refactored design, engineered features are stored directly on the
    # token dictionaries.  The scorer still expects a distinct object to exist
    # for backwards compatibility, so we provide a minimal, typed placeholder
    # here to keep the data model centralized.
    pass

