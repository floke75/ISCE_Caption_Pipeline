# Artifact Visibility Design (S08)

**Status:** Design Phase
**Plan ID:** `isce_caption_pipeline_frontend_update_v1`
**Date:** 2025-07-05

## Problem Statement

Users currently run jobs (Inference, Training) but cannot verify the results without SSHing into the server or browsing the filesystem manually. The Job Board displays `srt_path` or `workspace` paths as static text, which is friction-heavy.

## Goals

1.  **Direct Access:** Allow users to view and download key artifacts (SRT, JSON, Logs) directly from the Job Details pane.
2.  **Safety:** Respect the filesystem sandbox; only serve files from within allowed roots (Workspace, Output Dir).
3.  **Usability:** Provide a dedicated viewer page for artifacts to avoid cluttering the dashboard, with support for large files (virtual scrolling) and raw downloads.

## Technical Architecture

### 1. Backend API Updates (`ui/backend/api/routes/files.py`)

The current `FileBrowser` is limited to directory listing and existence validation. We need to add:

-   **`GET /api/files/download`**
    -   **Params:** `path` (absolute path).
    -   **Validation:** Must be within `allowed_roots`.
    -   **Response:** `FileResponse` (attachment).

-   **`GET /api/files/content`**
    -   **Params:** `path` (absolute path), `head_bytes` (optional limit).
    -   **Validation:** Must be within `allowed_roots`.
    -   **Response:** `{"content": "...", "size": 12345, "mime": "text/plain"}`.
    -   **Constraint:** Only serve text/json content inline. Binary files should force download.

### 2. Frontend Routing (`ui/frontend/src/App.tsx`)

A new route is required for the dedicated viewer:

-   **Route:** `/jobs/:jobId/artifacts/view`
-   **Query Params:** `?path=/abs/path/to/file.srt`
-   **Component:** `ArtifactViewer`

### 3. UI Components

#### A. Job Details Update (`JobBoard.tsx`)
In the "Results" or "Details" section, detect file paths (ending in `.srt`, `.json`, `.txt`, `.log`) and render them as links/buttons instead of plain text.

-   **Label:** "View" (opens Viewer) | "Download" (direct link).
-   **Icon:** Based on extension.

#### B. Artifact Viewer Page (`ArtifactViewer.tsx`)
A standalone page with:
-   **Header:** Filename, "Back to Job", "Download Raw".
-   **Body:**
    -   **Code/Text View:** Using `react-window` or a lightweight `List` for virtual scrolling if lines > 1000.
    -   **Syntax Highlighting:** Basic support (SRT, JSON).

## User Experience Flow

1.  User clicks a job in **Job Monitor**.
2.  In **Details > Results**, they see `output.srt`.
3.  They click **View**.
4.  Navigation takes them to `/jobs/123/artifacts/view?path=...`.
5.  They scroll through the SRT to verify timing.
6.  They click **Back** to return to the job list.

## Large File Strategy

-   **Backend:** The `/content` endpoint should support a `limit` param to avoid OOM on 1GB logs.
    -   *Initial Strategy:* Load first 500KB. If file is larger, show "File too large for preview, please download" or "Load more" (chunked loading).
    -   *User Preference:* Client-side virtual scrolling "if not too brittle". We will implement basic virtual scrolling for the *loaded* content. Full-file streaming for 100MB+ logs is out of scope for S08b (requires WebSocket/range-requests). We will stick to **Head Preview + Download Fallback** for massive files.

## Artifact Scope

| Type | Extension | Viewer |
| :--- | :--- | :--- |
| Subtitles | `.srt` | Text (Monospace) |
| Metadata | `.json` | JSON Pretty Print |
| Logs | `.log` | Text (Monospace) |
| Transcript | `.txt` | Text (Prose) |

## Implementation Steps (S08b)

1.  Implement backend endpoints in `ui/backend/api/routes/files.py`.
2.  Create `ArtifactViewer` component in frontend.
3.  Add Route in `App.tsx`.
4.  Update `JobBoard` to linkify paths.
