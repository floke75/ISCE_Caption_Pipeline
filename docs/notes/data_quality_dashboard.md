# Data Quality Dashboard Design

## Overview
The Data Quality Dashboard provides deep insights into the artifacts generated during the pipeline's execution. It focuses on the "enriched" token streams used for both training and inference, allowing operators to verify the quality of the data *before* or *after* segmentation.

## Goals
1.  **Verify Feature Integrity:** Ensure that critical features (pauses, speaker changes) are correctly detected and populated.
2.  **Analyze Corpus Stats:** For training jobs, understand the distribution of breaks and pauses in the ground truth.
3.  **Diagnose Inference Issues:** For inference jobs, visualize the input features that drive the segmentation model.

## Data Sources
-   **Inference:** `*.enriched.json` (Token-level features including timing, syntax, and pauses).
-   **Training:** `*.train.words.json` (Ground truth tokens with labels).

## Proposed Metrics & Visualizations

### 1. Summary Cards
High-level statistics to give an at-a-glance view of the file.
-   **Total Duration:** Total duration of the media/transcript.
-   **Token Count:** Total number of words.
-   **Avg CPS:** Average characters per second across all tokens.
-   **Avg Pause:** Average duration of inter-word pauses.
-   **Speaker Changes:** Count of detected speaker changes.

### 2. Feature Distributions (Histograms)
Visualizing the spread of key features.
-   **Pause Duration Distribution:** A histogram of `pause_after_ms`.
    -   *Why:* Critical for segmentation. Long pauses should correlate with breaks.
    -   *Thresholds:* Highlight pauses > 500ms (strong break signal).
-   **Characters Per Second (CPS) Distribution:** A histogram of token-level CPS.
    -   *Why:* Detects ASR hallucination (infinite speed) or alignment errors.
-   **Token Duration:** Histogram of individual word durations.

### 3. Structural Events
-   **Speaker Change Timeline:** A simple timeline indicating where speaker changes occur.
-   **Sentence Boundaries:** (If available from spaCy) Distribution of sentence lengths.

### 4. Training-Specific Metrics
-   **Break Type Distribution:** Pie chart of `O` (None), `LB` (Line Break), `SB` (Sentence Break).
    -   *Why:* Ensure class balance (usually mostly `O`).

## UI Layout
The dashboard will be a tab or section within the **Job Details** view, alongside "Artifacts" and "Alignment".

```
+-----------------------------------------------------------------------+
|  [Summary Cards]                                                      |
|  Total Tokens: 12,405   |   Avg CPS: 14.2   |   Speaker Changes: 15   |
+-----------------------------------------------------------------------+
|                                                                       |
|  [Pause Distribution Chart]            [Break Type Distribution]      |
|  |       *                             |          (Pie)               |
|  |     * * *                           |      O: 80%                  |
|  |   * * * * *                         |      LB: 12%                 |
|  | * * * * * * *                       |      SB: 8%                  |
|  +---------------------                |                              |
|                                                                       |
+-----------------------------------------------------------------------+
|                                                                       |
|  [Timeline / Heatmap]                                                 |
|  [ |||  |  ||   ||||   |      |||   ||   ] Speaker Changes            |
|                                                                       |
+-----------------------------------------------------------------------+
```

## Implementation Plan
-   **Component:** `DataQualityDashboard`
-   **Library:** `recharts` (for histograms/pies) or simple CSS bars if lightweight is preferred.
-   **Data Loading:** Fetch JSON artifacts via `/files/content`.
