# Job Templates Design

## Overview
Job Templates allow users to save and reuse configuration sets for Inference and Training jobs. This reduces repetitive data entry and ensures consistency across runs (e.g., always using "High Precision" beam settings for Client X).

## Schema
Templates are stored as JSON objects in the browser's `localStorage` under the key `isce_job_templates`.

```typescript
type JobType = 'inference' | 'training_pair' | 'model_training';

interface JobTemplate {
  id: string; // UUID
  name: string; // User-defined name
  type: JobType;
  createdAt: string; // ISO date
  data: TemplateData;
}

interface TemplateData {
  // Shared
  notes?: string;
  overrides?: {
    pipeline?: Record<string, unknown>;
    segmentation?: Record<string, unknown>;
  };

  // Inference Form
  outputDir?: string; // Optional default output
  modelConfigPath?: string;
  preset?: string; // e.g., 'standard'
  beamWidth?: number;
  diarization?: boolean;

  // Model Training Form
  iterations?: number;
  errorBoost?: number;

  // Note: File paths (media, transcript, corpus) are generally NOT templated
  // as they change per job, but output directories often remain constant.
}
```

## UI Components

### 1. `TemplateSelector` (New Component)
Placed at the top of the form.
- **Dropdown:** "Load template..."
- **Actions:**
  - Select a template -> Populates form state.
  - "Save current settings as..." -> Opens modal/prompt for name.
  - "Manage templates" -> Simple list to delete old templates.

### 2. Form Integration
- **InferenceForm:**
  - Adds `TemplateSelector` at top.
  - `handleLoadTemplate(data)`: Sets `outputDir`, `notes`, `preset`, `beamWidth`, `diarization`, `edits`.
  - `handleSaveTemplate()`: Gathers current state and saves.
- **ModelTrainingForm:**
  - Adds `TemplateSelector`.
  - `handleLoadTemplate(data)`: Sets `iterations`, `errorBoost`, `notes`, `edits`.
- **TrainingPairForm:**
  - Adds `TemplateSelector`.
  - `handleLoadTemplate(data)`: Sets `notes`, `edits`. (Least useful here, but consistent).

## Storage Strategy
- **Persistence:** `localStorage` (Client-side only).
- **Key:** `isce_job_templates` (Array of `JobTemplate`).
- **Migration:** No complex migration needed for v1.

## Interaction Flow
1. **Saving:**
   - User fills out form (e.g., sets Beam Width = 10, Diarization = Off, Notes = "Draft").
   - Clicks "Save Template".
   - Enters name "Fast Draft No-Diar".
   - Template is saved to list.
2. **Loading:**
   - User opens form (default state).
   - Selects "Fast Draft No-Diar" from dropdown.
   - Form updates: Beam Width -> 10, Diarization -> Off, Notes -> "Draft".
   - User selects distinct Media File.
   - User clicks Submit.

## Mockups
See `docs/screenshots/S13/` for visual reference.
