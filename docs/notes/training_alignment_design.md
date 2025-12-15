# Training Alignment Visualization Design (S09 - REVISED v2)

## Purpose
The alignment visualization builds trust in the pipeline by allowing operators to "watch" the training process. It verifies that the automated alignment (Needleman-Wunsch) has correctly matched the high-quality edited subtitle cues to the raw, timestamped ASR words.

## Design Decisions: "Teleprompter Style"

### 1. Verification Goal
The user should be able to verify that the **Edited Text** (Ground Truth) is in sync with the **Audio/ASR** (Time Reference). A split-screen, continuous scrolling view allows this by placing both streams on a shared vertical time axis.

### 2. Layout (Two-Column Teleprompter)
See `docs/screenshots/S09/teleprompter_alignment_mockup_v2.png`.

-   **Shared Vertical Time Axis:** The central spine of the UI represents time, moving downwards.
-   **Left Column (Edited Cues):**
    -   Displays the final, human-edited subtitle blocks.
    -   Vertical position and height are determined by the cue's *start* and *end* times.
    -   Visual Style: Clean, formatted blocks (like teleprompter cards) aligned to the right edge of the column.
-   **Right Column (Raw ASR Words):**
    -   Displays the individual words recognized by WhisperX.
    -   Vertical position corresponds exactly to each word's timestamp.
    -   Visual Style: Small "chips" or a continuous stream of text that flows down the timeline, aligned to the left edge of the column.
-   **Interaction:**
    -   **Synchronized Scrolling:** The user scrolls the timeline, and both columns move together.
    -   **Playback:** A "Play" button auto-scrolls the view (teleprompter mode), keeping the current timestamp centered.

### 3. Data Sources
-   **Edited Stream:** `*.train.words.json` (grouped by `cue_id`).
-   **ASR Stream:** `*.asr.visual.words.diar.json`.

### 4. Implementation Plan (S09b)
1.  **Component:** `AlignmentViewer.tsx`.
2.  **Virtualization:** Use a virtual list library (or simple CSS absolute positioning inside a scrolling container) to handle long audio files efficiently.
    -   *Approach:* Map 1 second of audio to `X` pixels (e.g., 60px/sec). Calculate `top` position for every element: `top = timestamp * 60`.
    -   *Layout:* Use `display: flex` container with `flex: 1` columns to ensure strict 50/50 split regardless of screen size.
3.  **Playback:** Use a standard HTML5 `<audio>` element (hidden or minimal). Update the scroll position on `timeupdate` events.
4.  **Sync Logic:**
    -   No complex "diffing" algorithm is needed for the view itself; the *visual alignment* on the time axis reveals the diffs naturally.

## Visual Cues for Errors
-   **Drift:** If the ASR words are consistently "ahead" (higher) or "behind" (lower) than the Edited Cues, there is an offset issue.
-   **Hallucinations:** A dense cluster of ASR words next to empty space in the Edited column indicates ASR hallucinations that were removed.
-   **Missing Audio:** A gap in ASR words next to an Edited Cue indicates the editor added text not present in the audio.
