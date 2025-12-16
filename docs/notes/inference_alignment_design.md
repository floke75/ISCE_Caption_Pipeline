# Inference Alignment Visualization Design (S10)

## Purpose
Enable operators to verify the **Stage 2 Alignment** quality for inference jobs. This ensures that the primary input text (e.g., LLM-refined transcript) has been correctly mapped to the audio timeline (via ASR timestamps) *before* the final segmentation engine processes it.

## Design Decisions

### 1. Unified "Teleprompter" Layout
We will reuse the existing `AlignmentViewer` component. The core interaction—scrolling a shared vertical time axis to compare two text streams—remains identical.

-   **Left Column (Input Text):** Displays the aligned tokens from `enriched.json`.
-   **Right Column (ASR Reference):** Displays the raw words from `*.asr.visual.words.diar.json`.
-   **Time Axis:** Central spine with seconds/minutes.

### 2. Grouping Strategy (Toggle)
Unlike Training jobs (which have fixed "Cues"), Inference jobs start with a flat list of tokens. We will provide a **Display Mode** toggle to structure this data:

*   **Mode A: Input Lines (Default)**
    *   **Logic:** Group tokens based on the `is_llm_structural_break` flag.
    *   **Purpose:** Shows how the model "sees" the input structure (e.g., original line breaks from the text file). This allows the user to verify if the model is respecting intended phrasing.
    *   **Visual:** Each block represents one line from the input file.

*   **Mode B: Sentences**
    *   **Logic:** Group tokens based on the `is_sentence_final` flag.
    *   **Purpose:** Provides a linguistically natural view, useful for checking if sentences are aligned continuously in time or if there are large gaps.

### 3. Visualizing Hints
The `is_llm_structural_break` flag is a critical input feature (a hint to the segmenter).
-   **In "Input Lines" Mode:** The hint effectively creates the visual block boundary, so it is implicitly visualized.
-   **In "Sentences" Mode:** If a structural break occurs *within* a sentence block, it will be visualized as a subtle return icon (`↵`) or divider to show where the input text had a newline.

### 4. Data Sources
-   **Inference Stream:** `_intermediate/_inference_input/*.enriched.json` (via `?inference=...`).
-   **ASR Stream:** `_intermediate/_align/*.asr.visual.words.diar.json` (via `?asr=...`).

### 5. Exclusions
-   **Confidence Scores:** Per user feedback, ASR confidence scores will **not** be visualized to keep the interface clean.

## Implementation Plan (S10b)

1.  **Refactor `AlignmentViewer.tsx`:**
    -   Extract the row generation logic into an adapter pattern.
    -   `useTrainingAdapter`: Groups by `cue_id`.
    -   `useInferenceAdapter`: Groups by `is_llm_structural_break` or `is_sentence_final` based on local state.
2.  **Add Controls:**
    -   Add a toggle button group in the header: `[ Group by: Lines | Sentences ]` (visible only in Inference mode).
3.  **Update `JobBoard.tsx`:**
    -   Render the "Visualise Alignment" action for Inference jobs (previously only for Training).
