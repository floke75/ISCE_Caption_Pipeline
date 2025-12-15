# Training Alignment Visualization Design (S09)

## Purpose
The alignment visualization aims to build trust in the automated pipeline by surfacing the relationship between the human-edited subtitle cues (the "Ground Truth" for breaks) and the raw ASR words (which provide the precise timestamps).

It allows operators to verify that the alignment algorithm (Needleman-Wunsch) has correctly matched the text streams, ensuring that line breaks are learned from the correct audio segments.

## Design Decisions

### 1. Verification vs. Debugging
The visualization focuses on **verification**. It uses a time-bucket approach rather than explicitly visualizing the algorithm's traceback path (which would require deep backend changes).
- **Left Column:** Edited text, grouped by Cue ID.
- **Right Column:** ASR words, filtered to show only those falling within the Cue's start/end timestamps.
- **Sync Line:** A visual connector (or "Sync" column) to show the relationship.

### 2. Data Sources
- **Edited Stream:** Loaded from `*.train.words.json` (specifically tokens grouped by `cue_id`).
- **ASR Stream:** Loaded from `*.asr.visual.words.diar.json` (or similar raw ASR artifact).
- **Audio:** "Best effort" playback using the existing `/api/files/download` endpoint.

### 3. Layout (Mockup Analysis)
See `docs/screenshots/S09/training_alignment_mockup.png`.

- **Row-based:** Each row corresponds to one Edited Cue.
- **Time Axis:** Vertical.
- **Visuals:**
  - **Timestamps:** Clearly visible on the left.
  - **Cue Text:** Highlighted in a distinct box (e.g., Blue) to represent the "Target".
  - **ASR Words:** Displayed as "Chips" on the right. Gaps in ASR words (silence) or excessive ASR words (hallucinations) become immediately visible when compared to the fixed Cue duration.

### 4. Implementation Plan (S09b)
1. **Component:** Create `AlignmentViewer.tsx`.
2. **Data Fetching:**
   - Use `useQuery` to fetch the JSON content of the artifacts.
   - Requires the backend `/api/files/content` endpoint (already exists).
3. **Logic:**
   - Parse `train.words.json` to extract Cues (group tokens by `cue_id`).
   - Parse raw ASR JSON.
   - Map ASR words to Cues based on `word.start >= cue.start` and `word.end <= cue.end`.
   - Handle "orphaned" words (words that fall between cues) by displaying them in a separate "Gap" row or attached to the previous cue with a visual warning.

## Future Improvements
- **Media Player:** A dedicated `/media` endpoint with Range support would allow seeking to the exact timestamp of a clicked word.
- **Diff View:** Highlight text differences (Edited vs. ASR) to show corrections made by the human editor.
