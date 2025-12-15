# Job Monitoring Analysis & Design (S07)

**Date:** 2025-07-05
**Status:** Analysis Completed

## 1. Overview
This document analyzes the current state of the Job Monitor (Job Board) in the ISCE pipeline UI, based on mocked data representing diverse job states.

## 2. Current State Analysis

### 2.1 Job List
- **Layout:** Vertical list of job cards.
- **Information:** Title, Status Pill, Message/Progress, Relative Timestamp, Actions (Copy, Cancel).
- **Differentiation:** Jobs are distinguished only by the text title ("Inference run", "Training pair").
- **Status:** Status is color-coded (Yellow/Blue/Green/Red).
- **Timestamps:** Relative time (e.g., "5m ago") is shown.

### 2.2 Job Details
- **Structure:** Sections for Details, Runtime Parameters, Results, Logs, and Errors.
- **Error Handling:** Failed jobs display a dedicated "Error" card with a red border and the full error trace in a preformatted block. This is effective.
- **Logs:** Live streaming (SSE) with auto-scroll support.

### 2.3 Gaps & Observations
- **Visual Scanability:** It takes a moment to read the text to differentiate an Inference job from a Training job. Distinct icons would improve this.
- **Timestamp Precision:** The list shows relative time, but hovering doesn't reveal the absolute timestamp (based on code review).
- **Status Scanability:** Status pills are text-based. Icons (Check/X) would be faster to parse.

## 3. Design Proposal (S07b)

### 3.1 Job Type Icons
Introduce SVG icons to the left of the job title in the list view:
- **Inference:** `Play` or `Film` icon.
- **Training Pair:** `Database` or `FileText` icon.
- **Model Training:** `TrendingUp` or `Brain` icon.

### 3.2 Enhanced Timestamps
- Wrap the relative timestamp in a `title` attribute to show the absolute ISO date on hover.
- Example: `<span title={job.createdAt}>{relativeTime(job.createdAt)}</span>`

### 3.3 Status Icons
- Add an icon inside the status pill or alongside it:
  - **Success:** Checkmark
  - **Failed:** Exclamation/X
  - **Running:** Spinner (animated)
  - **Pending:** Clock

### 3.4 Implementation Details
- **Icons:** Use simple SVG paths inline or import from a lightweight library (e.g., `lucide-react` if available, or just inline SVGs to avoid deps). *Current codebase seems to rely on custom CSS/SVGs or no icons yet.*
- **Component:** Update `JobBoard.tsx` to include `JobIcon` and `StatusIcon` helpers.

## 4. Artifacts
Screenshots captured with mocked data:
- `docs/screenshots/S07/job_board_list.png` (Full list)
- `docs/screenshots/S07/details_running.png` (Inference Running)
- `docs/screenshots/S07/details_fail.png` (Model Training Failed)
