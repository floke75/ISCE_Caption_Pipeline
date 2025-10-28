# Beam Search Walkthrough

This note complements the docstrings in `isce/beam_search.py` by mapping the
call graph into newcomer-friendly stages. It is intentionally concise so teams
can skim it before diving into the code.

## 1. Normalising scorer payloads
* `_token_to_row_dict` coerces every token into a dictionary and stabilises the
  `token_index` field. Dependency-driven helpers in `isce.scorer` rely on the
  index to compute repeatable keys, so the function backfills or normalises the
  value whenever the payload originates from JSON or refinement slices.
  【F:isce/beam_search.py†L41-L115】
* `_get_lookahead_slice` calls the same helper across the lookahead window.  It
  accepts an `offset` so refinement slices and the backward beam keep absolute
  indexes despite operating on sub-spans. 【F:isce/beam_search.py†L104-L141】

## 2. Transition scoring via `Segmenter`
* `Segmenter._build_transition_context` packages the partially written block and
  forwards optional lookahead projections to the scorer.  The docstring explains
  how second-line estimates are computed, which matches the heuristics called
  out in the README summary. 【F:isce/beam_search.py†L284-L347】
* `Segmenter.run` is the hot loop.  It materialises a `TokenRow` for each token,
  queries the scorer, and keeps only the top `beam_width` hypotheses. The
  docstring covers fallback penalties and the preservation of `last_path_score`,
  which downstream refinement reuses. 【F:isce/beam_search.py†L315-L489】

## 3. Reconciliation, scoring, and refinement
* `_score_segmentation` rebuilds scorer payloads for any arbitrary break
  sequence.  Bidirectional reconciliation and refinement rely on it to evaluate
  alternative decisions without hand-crafting the dictionaries themselves.
  【F:isce/beam_search.py†L601-L682】
* `_reconcile_bidirectional_breaks` mixes forward/backward beams by re-scoring
  candidate swaps.  The helper intentionally keeps the trailing `SB` anchored so
  diagnostics and SRT writers see the same block endings they would expect from
  the forward pass. 【F:isce/beam_search.py†L763-L816】
* `refine_blocks` revisits weak captions with a wider beam and honours a
  `start_offset` parameter so the regenerated payloads keep absolute indexes.
  The docstring summarises the heuristics it uses to decide when refinement is
  worth the extra scoring. 【F:isce/beam_search.py†L825-L918】

## 4. Entry point
* `segment` orchestrates the passes based on configuration toggles. The function
  pairs with the README section "Segmentation safeguards at a glance" so readers
  can connect UI switches to the actual code paths. 【F:isce/beam_search.py†L929-L968】

---
For deeper motivation behind the dependency-aware features and the cost/benefit
trade-offs of keeping `token_index`, refer to `docs/spacy_feature_impact.md`.
