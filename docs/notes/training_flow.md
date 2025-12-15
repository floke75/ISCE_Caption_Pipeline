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

## Implemented Improvements
1.  **Disable Submit Button:** The submit button is now explicitly disabled until all required fields (paths and overrides) are valid. A `title` tooltip explains the reason (e.g., "Please provide valid paths to continue").
2.  **Specific Error Toasts:** Updated `handleSubmit` to check validity before mutation and show specific error messages (e.g., "Invalid media file path selected") instead of a generic catch-all.
3.  **Inline Help:** Added precise technical explanations for hyperparameters:
    - **Iterations:** "Rounds of Expectation-Maximization reweighting to refine the model on hard examples."
    - **Error boost factor:** "Weight multiplier added to misclassified samples in each iteration (standard range 0.5–2.0)."
4.  **Placeholders:** Updated placeholders to reflect realistic defaults found in `train_model.py` (3 iterations, 1.0 boost) rather than arbitrary guesses.
5.  **Notes Clarity:** Updated "Operator notes" placeholder to explain its purpose: "Optional metadata stored in the job history for reproducibility."

## Screenshots (Improved State)
- `docs/screenshots/S04/training_pair_improved_initial.png` - Training form with disabled button.
- `docs/screenshots/S04/training_pair_improved_invalid.png` - Validation errors blocking submission.
- `docs/screenshots/S04/model_training_improved.png` - Model training form with new help text and disabled button.
