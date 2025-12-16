# System Health Monitoring - Analysis & Design

**Step:** S11
**Date:** 2025-07-05
**Status:** Approved

## 1. Overview
The goal is to provide users with visibility into the system's operational health, including storage capacity, memory usage, GPU availability, and job queue status. This ensures users can proactively manage resources (e.g., clearing old jobs when disk is low) and understand performance constraints (e.g., CPU-only mode).

## 2. API Design

### New Endpoint: `GET /api/health`
Extends the existing simple health check to return detailed metrics.

**Response Schema:**
```json
{
  "status": "ok",
  "system": {
    "disk": {
      "free_bytes": 123456789,
      "total_bytes": 500000000,
      "percent_used": 75.3
    },
    "memory": {
      "available_bytes": 8000000000,
      "total_bytes": 16000000000,
      "percent_used": 50.0
    },
    "gpu": {
      "available": false,
      "name": null,
      "device_count": 0
    }
  },
  "queue": {
    "pending": 2,
    "active": 1,
    "slots_total": 3
  }
}
```

**Implementation Details:**
- **Disk:** Monitor `STORAGE_ROOT` (`ui_data` volume).
- **Memory:** Use `psutil.virtual_memory()`.
- **GPU:** Check `torch.cuda.is_available()` (if torch is installed) or fallback to `nvidia-smi` check, or default to `available: false` gracefully.
- **Queue:** Query `JobManager` for job counts.

## 3. UI Design

### Component: `SystemStatus`
Located in the global `AppHeader` (top right, inside `header-meta`).

**Visual States:**
1.  **Healthy:** Green dot + "System: OK"
2.  **Warning:** Yellow dot + "System: Warning" (e.g., Low Disk, High Memory)
3.  **Error:** Red dot + "System: Error" (e.g., Disk Full)
4.  **CPU Mode:** "System: OK (CPU)" or specific icon if GPU is missing.

**Interaction:**
- **Click/Hover:** Opens a `Popover` (or dropdown) showing detailed metrics.

**Popover Content:**
- **Storage:** Progress bar showing usage. Text: "45GB free of 100GB".
- **Memory:** Progress bar. Text: "8GB / 16GB (50%)".
- **Compute:** "GPU: NVIDIA T4" or "CPU Only".
- **Queue:** "Active: 1 | Pending: 2".

**Alert Thresholds:**
- **Disk:**
    - Warning: < 2 GB free.
    - Error: < 500 MB free (Shows prominent global banner).
- **Memory:**
    - Warning: > 90% used.
- **GPU:**
    - Info: Not available (CPU Mode).

## 4. Implementation Plan (S11b)
1.  **Backend:** Add `psutil` dependency. Implement `/api/health` logic in `ui/backend/app.py`.
2.  **Frontend:** Create `SystemStatus.tsx` using existing styling patterns. Add to `App.tsx`.
3.  **Verification:** Mock backend responses to verify all visual states (Healthy, Warning, Error, CPU Mode).
