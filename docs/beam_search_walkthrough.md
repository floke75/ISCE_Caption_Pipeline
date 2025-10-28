# Beam Search Walkthrough

This note complements the docstrings in `isce/beam_search.py` by mapping the
call graph into newcomer-friendly stages. It is intentionally concise so teams
can skim it before diving into the code. For executable examples of these
paths, follow the companion assertions in
`tests/test_beam_search.py::test_token_index_propagates_through_all_scoring_paths`
and neighbouring cases.

## 1. Normalising scorer payloads
* `_token_to_row_dict` coerces every token into a dictionary and stabilises the
  `token_index` field. Dependency-driven helpers in `isce.scorer` rely on the
  index to compute repeatable keys, so the function backfills or normalises the
  value whenever the payload originates from JSON or refinement slices. See the
  guard assertions in `tests/test_beam_search.py::test_token_to_row_dict_*`.
  【F:isce/beam_search.py†L41-L115】【F:tests/test_beam_search.py†L94-L122】
* `_get_lookahead_slice` calls the same helper across the lookahead window.  It
  accepts an `offset` so refinement slices and the backward beam keep absolute
  indexes despite operating on sub-spans. The lookahead contract is exercised by
  `tests/test_beam_search.py::test_score_path_exposes_lookahead`.【F:isce/beam_search.py†L104-L141】【F:tests/test_beam_search.py†L221-L246】

## 2. Transition scoring via `Segmenter`
* `Segmenter._build_transition_context` packages the partially written block and
  forwards optional lookahead projections to the scorer.  The docstring explains
  how second-line estimates are computed, which matches the heuristics called
  out in the README summary and the checks inside
  `tests/test_beam_search.py::test_transition_context_projects_second_line_from_first_line`.【F:isce/beam_search.py†L284-L347】【F:tests/test_beam_search.py†L274-L319】
* `Segmenter.run` is the hot loop.  It materialises a `TokenRow` for each token,
  queries the scorer, and keeps only the top `beam_width` hypotheses. The
  docstring covers fallback penalties and the preservation of `last_path_score`,
  which downstream refinement reuses. The behaviour is asserted in
  `tests/test_beam_search.py::test_segmenter_records_last_path_score`.【F:isce/beam_search.py†L315-L489】【F:tests/test_beam_search.py†L246-L273】

## 3. Reconciliation, scoring, and refinement
* `_score_segmentation` rebuilds scorer payloads for any arbitrary break
  sequence.  Bidirectional reconciliation and refinement rely on it to evaluate
  alternative decisions without hand-crafting the dictionaries themselves. See
  `tests/test_beam_search.py::test_score_path_assigns_block_indices_with_offset`
  for the start-offset contract.【F:isce/beam_search.py†L601-L682】【F:tests/test_beam_search.py†L247-L270】
* `_reconcile_bidirectional_breaks` mixes forward/backward beams by re-scoring
  candidate swaps.  The helper intentionally keeps the trailing `SB` anchored so
  diagnostics and SRT writers see the same block endings they would expect from
  the forward pass. Its invariants are codified in
  `tests/test_beam_search.py::test_bidirectional_reconciliation_prefers_higher_score`.
  【F:isce/beam_search.py†L763-L816】【F:tests/test_beam_search.py†L451-L501】
* `refine_blocks` revisits weak captions with a wider beam and honours a
  `start_offset` parameter so the regenerated payloads keep absolute indexes.
  The docstring summarises the heuristics it uses to decide when refinement is
  worth the extra scoring; practical coverage lives in
  `tests/test_beam_search.py::test_token_index_propagates_through_all_scoring_paths`
  and `::test_refine_blocks_clamps_single_word_window`.【F:isce/beam_search.py†L825-L918】【F:tests/test_beam_search.py†L328-L408】【F:tests/test_beam_search.py†L520-L546】

## 4. Entry point
* `segment` orchestrates the passes based on configuration toggles. The function
  pairs with the README section "Segmentation safeguards at a glance" so readers
  can connect UI switches to the actual code paths. 【F:isce/beam_search.py†L929-L968】

---
For deeper motivation behind the dependency-aware features and the cost/benefit
trade-offs of keeping `token_index`, refer to `docs/spacy_feature_impact.md`.
