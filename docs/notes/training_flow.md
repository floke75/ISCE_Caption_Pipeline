# Training Flow Input Clarity and Validation (S04)

**Date:** 2025-07-05
**Status:** Audit Complete

## Observations

### 1. Training Pair Form (`TrainingPairForm.tsx`)
- **Required Inputs:** Media path, SRT path.
- **Validation:**
  - Client-side validation relies on `FilePathPicker` which debounces requests to `/files/validate`.
  - Empty submissions trigger a Toast error: "Provide valid media and SRT paths".
  - Invalid paths display "Path does not exist" or "Path is outside the allowed directories" inline below the input.
- **Friction Points:**
  - The toast error is generic ("Provide valid media and SRT paths") and doesn't highlight *which* field is invalid if one is valid and the other isn't.
  - The submit button remains enabled even if fields are invalid (though it blocks submission logic), which can be misleading. (Actually, code shows `disabled={mutation.isPending}`, not disabled on invalid validity state).
  - Help text is minimal ("Absolute media file path on the host").

### 2. Model Training Form (`ModelTrainingForm.tsx`)
- **Required Inputs:** Training corpus directory.
- **Validation:**
  - Client-side validation via `FilePathPicker` (directory mode).
  - Empty submission triggers Toast: "Select a valid corpus directory before submitting".
- **Friction Points:**
  - "Iterations" and "Error boost factor" are raw number inputs without range guidance or defaults shown in the UI (though placeholders exist).
  - "Operator notes" is free text.

### 3. General UX
- **Error Feedback:** Relies heavily on Toasts (`react-hot-toast`). These disappear automatically, which might make it hard for users to read long error messages.
- **Visual Cues:** Invalid fields show a red error message below the input (via `file-picker-status invalid` class), which is good.
- **Missing:**
  - No "Success" indication on the field itself (green checkmark) other than text "Path verified".
  - No info tooltips explaining *what* a "Training corpus" is or expected structure.

## Proposed Improvements
1.  **Disable Submit Button:** explicit disable state when form is invalid, with a tooltip explaining why.
2.  **Field-Level Error Highlighting:** Add a red border to the input itself, not just the status text below.
3.  **Presets:** Add a "Load Example" button to pre-fill paths for learning.
4.  **Inline Help:** Add `(?)` icons with tooltips for "Error boost factor" and "Iterations".

## Screenshots
- `docs/screenshots/S04/training_pair_initial.png` - Empty form.
- `docs/screenshots/S04/training_pair_validation_error.png` - Toast error on empty submit.
- `docs/screenshots/S04/training_pair_invalid_paths.png` - Inline validation errors for bad paths.
- `docs/screenshots/S04/model_training_initial.png` - Model training form.
