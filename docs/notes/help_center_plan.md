# Embedded help center implementation (S15b)

## Overview
- Added a modal-style Help Center with quickstart checklists, glossary entries, and a guided tour for the ISCE frontend.
- The Help Center opens from the main header via the "Help center" button and can drive navigation between tabs.
- Guided tour steps follow the critical workflows: inference → training pairs → model training → configuration → monitoring.

## Components
- `HelpCenter.tsx`: Renders the overlay, quickstarts, glossary, and guided tour controls.
- `help-center.css`: Styles for the overlay, cards, and controls.
- `App.tsx`: Wires the Help Center state, exposes tab metadata to the guided tour, and provides the trigger button.

## Content sources
- Quickstart steps align with the workflows documented in `FRONTEND.md`.
- Glossary definitions reference terminology from `README.md` and `FRONTEND.md` (workspace, overrides, diarization, alignment).
- Tour steps automatically switch tabs to match the highlighted workflow.

## Usage notes
- The tour can be started/stopped from the Help Center; navigation updates the active tab when a step references one.
- Quickstart cards deep-link to the associated tab and include documentation shortcuts.
- Button styles reuse the existing link aesthetic with a ghost variant for header placement.
