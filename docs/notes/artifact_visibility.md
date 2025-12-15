# Artifact Visibility Analysis (S08)

## 1. Overview
The current Job Monitor exposes job results as file paths but lacks mechanisms to:
- Download artifacts directly.
- Preview text-based artifacts (SRT, JSON, Logs) in-browser.
- Visualise alignment data.

## 2. Current State
- **Job Details:** Shows paths (e.g., `/data/workspaces/job-inf-completed/movie.srt`) in `<code>` blocks.
- **Interactivity:** Paths can be copied to clipboard, but not clicked.
- **Backend:** `ui/backend/api/routes/files.py` provides browsing (`/list`, `/validate`) but no direct file serving (e.g., `/download`, `/view`).

## 3. Gap Analysis
| Artifact Type | Current Exposure | Desired Exposure | Feasibility |
| :--- | :--- | :--- | :--- |
| **SRT Subtitles** | Path text only | Download Link + In-browser Text Viewer | High (Text file) |
| **JSON Metadata** | Path text only | Download Link + JSON Tree Viewer | High (Text file) |
| **Logs** | Path text + Stream | Stream (Partial) + Download Full Log | High |
| **Media (Video/Audio)** | Path text only | N/A (Browser playback requires correct codecs/hosting) | Low (Out of scope for now) |

## 4. Proposed Solution (S08b)
### 4.1. Backend Updates
- Add a new endpoint to `ui/backend/api/routes/files.py` (or a new router):
  - `GET /api/files/content?path=...`: Returns file content (text/plain or application/json) with safety checks.
  - `GET /api/files/download?path=...`: Triggers file download (attachment).

### 4.2. UI Components
- **`ArtifactLink` Component:** Replaces raw path text for known artifact extensions (.srt, .json, .log, .txt).
  - Renders: `[Icon] Filename (Download | View)`
- **`FileViewer` Modal:**
  - Fetches content via `/api/files/content`.
  - Displays text in a syntax-highlighted editor (e.g., simple `<pre>` or Monaco).
- **Integration:** Update `JobBoard.tsx` to detect artifact paths and render `ArtifactLink`.

## 5. Verification Plan
- **Mock:** Use Playwright to intercept `/api/files/content` and verify the viewer opens with correct content.
- **Visual:** Verify the "View" button appears for SRT/JSON files.
