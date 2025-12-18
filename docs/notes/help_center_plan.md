# Embedded Help Center Design

## Overview
The Embedded Help Center aims to reduce the "time to first successful job" for new users and provide quick reference for advanced operators. It will be a slide-over panel accessible from the global header, distinct from the specific form tooltips.

## Goals
1.  **Onboarding:** Provide a "Quickstart Checklist" for the first run.
2.  **Glossary:** Define domain-specific terms (Beam Search, Diarization, Pause Z) in one place.
3.  **Troubleshooting:** Link common error messages to solutions.
4.  **Context:** Explain *why* certain parameters matter (e.g., Error Boost).

## Content Strategy
We will reuse existing documentation from `README.md` and `docs/` but format it for quick consumption.

### 1. Quickstart Checklist
- [ ] Upload media file
- [ ] Choose "Inference" workflow
- [ ] Select "Standard" preset
- [ ] Run Job
- [ ] Inspect "Alignment"

### 2. Glossary Terms
- **Beam Width:** How many alternative subtitle segmentations the model explores. Higher = slower but better.
- **Diarization:** The process of identifying "who spoke when" to assign speaker labels.
- **Pause Z-Score:** A normalized measure of silence duration between words.
- **Token:** A single word or punctuation mark used as the basic unit of processing.

### 3. FAQ / Troubleshooting
- *Job stuck in Pending?* Check if the backend worker is running.
- *"File not found"?* Ensure paths are relative to the project root or absolute.

## UI Design

### Entry Point
A "?" icon button in the top-right header (next to System Status).

### Panel Layout (Slide-over)
```
+---------------------------------------------------------------+
|                                            |  [Help Center] X |
|                                            |                  |
|                                            |  [Search...]     |
|                                            |                  |
|                                            |  > Quickstart    |
|                                            |  > Glossary      |
|                                            |  > FAQ           |
|                                            |                  |
|                                            |  [Content Area]  |
|                                            |  **Beam Width**  |
|                                            |  Controls the... |
|                                            |                  |
|                                            |  [Tour Button]   |
+---------------------------------------------------------------+
```

### Guided Tour (Future S15b)
A lightweight overlay that highlights:
1.  **Workflow Tabs:** "Start here to choose your task"
2.  **Job Board:** "Monitor progress here"
3.  **Artifacts:** "Download results here"

## Technical Implementation
- **Component:** `HelpCenter.tsx` (Sidebar/Drawer).
- **State:** Global visibility state (Zustand or Context).
- **Content:** Markdown or structured JSON imported at build time.
