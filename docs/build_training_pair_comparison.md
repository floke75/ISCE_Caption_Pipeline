# `build_training_pair_standalone.py` Variants — Structural Review

## Overview
Two different versions of `build_training_pair_standalone.py` are under consideration:

1. **Repository version** – the refactored script currently living in this repository (`repo variant`).
2. **Submitted version** – the "self-contained training pair builder" provided for evaluation (`submitted variant`).

This document compares the two implementations with respect to completeness, flexibility, maintainability, and risk.

## High-level Summary
| Aspect | Repo variant | Submitted variant |
| --- | --- | --- |
| Primary goal | End-to-end alignment, feature engineering, and labeling for both inference (TXT) and training (SRT) inputs via a CLI entry point. | Focused training data builder that expects pre-aligned "visual words" JSON and a ground-truth SRT; no support for TXT inference data. |
| Configuration | Layered defaults + YAML overrides + CLI flags. | Hard-coded settings dictionary; edit-in-code workflow. |
| Alignment responsibilities | Includes Needleman–Wunsch alignment to marry edited text with ASR timestamps. | Assumes timestamps already exist in visual-words JSON; no alignment logic. |
| Speaker handling | Applies "sole winner" speaker correction plus inference hint wiring. | Copies majority speaker per cue from alignment JSON; no corrective post-processing. |
| Output formats | `.train.words.json` (training) or `.enriched.json` (inference) with optional LLM hint propagation. | Always emits paired `.train.words.json` + `.train.labels.json`; includes prosody statistics in metadata. |
| NLP features | Optional spaCy enrichment without vectors. | Optional spaCy enrichment with vector sidecar writer (numpy). |

**Verdict:** the repo variant remains the most complete and production-ready option because it handles the entire alignment and enrichment workflow, supports both training and inference pipelines, and is externally configurable. The submitted variant can be useful as a specialized training-label builder when alignment has already been solved elsewhere, but it omits key stages that the repo relies on.

## Detailed Comparison

### Scope & Responsibilities
- **Repo variant**: Handles token alignment from raw TXT/SRT inputs, performs feature engineering, applies speaker correction, and serializes outputs for both training and inference paths, all orchestrated inside `process_file` with CLI argument parsing for integrations.【F:build_training_pair_standalone.py†L499-L639】
- **Submitted variant**: Starts from an already aligned visual-words dataset, applies cue-aware labeling/feature steps, and writes training outputs without any alignment or inference pathway; configuration is edited directly in the script and is executed via a plain `main()` call.【F:docs/alt_build_training_pair_standalone.py†L193-L614】【F:docs/alt_build_training_pair_standalone.py†L695-L741】

**Impact:** The repo variant covers the full pipeline expected by the rest of the project. The submitted variant would require external tooling to produce compatible enriched inference data, limiting its usefulness as a drop-in replacement.

### Configuration & Extensibility
- **Repo variant**: Supports YAML overlays and CLI overrides via `_recursive_update`, `_resolve_paths`, and the argparse interface, making it easy to adapt to different environments or experiments.【F:build_training_pair_standalone.py†L33-L116】【F:build_training_pair_standalone.py†L600-L639】
- **Submitted variant**: Offers a single `SETTINGS` dictionary with absolute Windows paths that must be edited before every run; no CLI or external config hooks exist.【F:docs/alt_build_training_pair_standalone.py†L33-L108】【F:docs/alt_build_training_pair_standalone.py†L695-L741】

**Impact:** The repo variant is substantially more maintainable in multi-environment deployments. The submitted variant creates churn for path changes and complicates automation.

### Alignment & Token Integrity
- **Repo variant**: Implements global alignment (`_global_align` + `align_text_to_asr`) so that TXT or SRT edits inherit precise timestamps from the ASR reference.【F:build_training_pair_standalone.py†L131-L263】 This is critical for both inference and training parity.
- **Submitted variant**: Assumes timestamps are already perfect in the visual-words JSON (`load_visual_words`) and only reassigns cue IDs based on time windows; there is no mechanism to align edited text.【F:docs/alt_build_training_pair_standalone.py†L223-L371】【F:docs/alt_build_training_pair_standalone.py†L475-L614】

**Impact:** Without alignment, the submitted variant cannot ingest manual TXT edits, so inference parity with the model pipeline is lost.

### Speaker Handling
- **Repo variant**: Provides the "sole winner" speaker correction algorithm to clean diarization mistakes sentence-by-sentence.【F:build_training_pair_standalone.py†L312-L349】
- **Submitted variant**: Propagates majority speakers per cue based on pre-existing diarization labels and offers no correction step.【F:docs/alt_build_training_pair_standalone.py†L550-L592】

**Impact:** The repo variant actively mitigates diarization noise; the submitted variant assumes upstream data is already correct.

### Feature Engineering & Labeling
- **Repo variant**: Adds prosody (pause_after_ms), spaCy fields, guardrail heuristics, and cue-derived labels, covering both training and inference features.【F:build_training_pair_standalone.py†L354-L495】
- **Submitted variant**: Adds both pause_before/after, pause_z, optional spaCy with dependency heads, dialogue dash detection, numeric-unit glue, and LB hardening logic that can emit `LB_HARD` in labels and collapse to `break_type` per token.【F:docs/alt_build_training_pair_standalone.py†L402-L687】

**Impact:** Feature sets overlap but differ in emphasis. The submitted variant introduces richer prosody metadata (pause_before_ms, pause_z) and explicit `LB_HARD` handling, which could be valuable if integrated, yet this comes at the expense of missing inference support.

### Outputs & Sidecars
- **Repo variant**: Emits a single JSON per run (training or inference) and does not manage vector sidecars.【F:build_training_pair_standalone.py†L567-L590】
- **Submitted variant**: Always writes both tokens and labels files and can optionally persist spaCy vectors to `.npy` files for downstream consumption.【F:docs/alt_build_training_pair_standalone.py†L614-L687】

**Impact:** The submitted variant's vector export is a nice-to-have but requires numpy and additional storage management; the repo variant keeps I/O minimal and aligned with existing consumers.

## Pros & Cons

### Repo Variant
**Pros**
- Full pipeline coverage (alignment → correction → features → output) for both training and inference modes.【F:build_training_pair_standalone.py†L499-L590】
- Robust configuration strategy through CLI and YAML, easing deployment automation.【F:build_training_pair_standalone.py†L33-L116】【F:build_training_pair_standalone.py†L600-L639】
- Built-in speaker correction and inference hint propagation safeguard model quality.【F:build_training_pair_standalone.py†L312-L349】【F:build_training_pair_standalone.py†L525-L555】

**Cons**
- No support for spaCy vector sidecars or pause_z metrics; only pause_after_ms is emitted.【F:build_training_pair_standalone.py†L354-L580】
- Alignment routine can be heavy for long transcripts (Needleman–Wunsch), potentially impacting runtime.【F:build_training_pair_standalone.py†L131-L262】

### Submitted Variant
**Pros**
- Rich prosody analytics (pause_before_ms, pause_after_ms, pause_z) and LB hardening rules generate nuanced training signals.【F:docs/alt_build_training_pair_standalone.py†L402-L532】【F:docs/alt_build_training_pair_standalone.py†L632-L687】
- Optional spaCy dependency export and vector sidecar writer enhance downstream ML workflows.【F:docs/alt_build_training_pair_standalone.py†L592-L687】
- Conservative transformation toggles (hyphen split/merge, dash stripping) allow experimentation without touching raw alignment by default.【F:docs/alt_build_training_pair_standalone.py†L331-L474】

**Cons**
- Hard-coded paths and lack of CLI/config make automation and cross-environment use fragile.【F:docs/alt_build_training_pair_standalone.py†L33-L108】【F:docs/alt_build_training_pair_standalone.py†L695-L741】
- No mechanism to align raw TXT/SRT edits with ASR; assumes pre-aligned data and thus cannot replace the repo tool for inference.【F:docs/alt_build_training_pair_standalone.py†L223-L371】【F:docs/alt_build_training_pair_standalone.py†L695-L741】
- Speaker correction is limited to majority-vote propagation, offering no remediation for diarization errors.【F:docs/alt_build_training_pair_standalone.py†L550-L592】

## Recommendation
Retain the repo variant as the canonical implementation for production workflows. Cherry-pick ideas from the submitted variant (e.g., pause_z, LB hardening, optional vector dumps) by porting them into the repo codebase if those features are desired. Integrating them directly will preserve configuration flexibility and alignment support while gaining the richer annotations.
