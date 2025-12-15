## Successful Implementation: Training Alignment Viewer

**Objective:**
Allow users to visually verify that their ground-truth training data (edited subtitles) is correctly aligned with the raw timestamped ASR words.

**Solution:**
Implemented a split-screen 'Teleprompter' style view (`AlignmentViewer.tsx`).
- **Layout:** Two fixed-width columns (Edited Cues vs. ASR Words) with a central vertical time axis.
- **Synchronization:** Elements are positioned absolutely based on their timestamps (e.g., `top = time * 60px`). This naturally aligns related content without complex diffing logic.
- **Integration:** Added a 'Visualise Alignment' button to the `JobBoard` for completed training jobs, which passes the file paths as URL parameters.

**Key Learnings:**
1.  **Strict Layouts:** For side-by-side verification tools, using CSS `flex` with `flex: 1` and explicit `position: relative` containers is crucial to prevent columns from collapsing or wrapping on smaller screens.
2.  **Playwright Testing:** When verifying wide layouts, the Playwright context must be initialized with a desktop viewport (e.g., `viewport={'width': 1920, 'height': 1080}`) to ensure elements are visible and positioned as expected.
3.  **Mocking API:** Mocking the backend `/api/files/content` endpoint with realistic JSON data allowed for rapid iteration on the frontend component without needing to run the full Python pipeline.
