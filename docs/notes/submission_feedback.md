# Submission Feedback Analysis & Design (S06)

**Date:** 2025-07-05
**Status:** Analysis Completed

## 1. Overview
This document analyzes the current state of form submission feedback in the ISCE pipeline UI and outlines the design for improvements to ensure consistency and usability.

## 2. Current State Analysis

### 2.1 Inference Form (`InferenceForm.tsx`)
- **Behavior:** The "Launch inference run" button remains enabled even when required fields (Media file path) are empty or invalid.
- **Feedback:** Validation checks occur *on click*. If invalid, a toast notification (`react-hot-toast`) appears (e.g., "Select a valid media file path before submitting").
- **Backend Errors:** Displayed via toast.
- **Observation:** This pattern allows users to click a button that will essentially "fail" immediately, which is reactive rather than proactive.

### 2.2 Training Pair Form (`TrainingPairForm.tsx`)
- **Behavior:** The "Launch training-pair job" button is **disabled** until all required fields (Media, SRT) are valid.
- **Feedback:** The button has a `title` attribute explaining why it is disabled (e.g., "Please provide valid paths to continue").
- **Backend Errors:** Displayed via toast.
- **Observation:** This pattern prevents errors before they happen and provides a clear visual cue that the form is incomplete.

### 2.3 Model Training Form (`ModelTrainingForm.tsx`)
- **Behavior:** Follows the same "Disabled Button" pattern as the Training Pair Form.

### 2.4 Inconsistency
There is a clear UX inconsistency where the Inference form allows submission attempts on invalid data, while the Training forms do not.

### 2.5 Validation Components
- **`FilePathPicker`:** Provides inline status text ("Validating...", "Path verified", "Path does not exist").
- **Error States:** Uses CSS classes (`invalid`, `valid`) to style the status text.

## 3. Design Proposal (S06b)

### 3.1 Unify Submission UX
We will adopt the **Disabled Button** pattern across all forms, bringing `InferenceForm` in line with the others.

**Changes required:**
- Update `InferenceForm.tsx` to track overall form validity (`formInvalid` state).
- Disable the submit button when `formInvalid` is true.
- Add a `title` tooltip to the disabled button explaining the missing requirements.

### 3.2 Improve Inline Validation
- Ensure `FilePathPicker` error messages are distinct (e.g., ensure `invalid` class renders text in red/warning color).
- Ensure the "Validating..." state is clearly visible (spinner or distinct text).

### 3.3 Backend Error Handling
- Continue using `react-hot-toast` for backend errors (400/500), as this works well.
- Ensure error messages returned by the backend (e.g., "File not found on host") are propagated to the toast. (Verified as working in current implementation).

## 4. Implementation Plan (Concrete Fixes)

1.  **Refactor `InferenceForm.tsx`:**
    - Introduce `const formInvalid = !mediaValid || !transcriptValid || !outputDirValid || !configPathValid || overrideInvalid;`.
    - Update `<button>`: `disabled={mutation.isPending || formInvalid}`.
    - Update `<button>`: `title={formInvalid ? 'Please resolve validation errors' : 'Launch inference run'}`.
    - Remove `handleSubmit` validation checks that trigger toasts (except for race conditions, but UI should prevent click).

2.  **Verify `FilePathPicker.tsx` Styling:**
    - Check `forms.css` to ensure `.file-picker-status.invalid` has appropriate styling (e.g., `color: var(--color-error)`).

## 5. Artifacts
Screenshots captured during analysis:
- `docs/screenshots/S06/inference_invalid_toast.png` (Current behavior: Toast on click)
- `docs/screenshots/S06/training_invalid_disabled.png` (Target behavior: Disabled button)
- `docs/screenshots/S06/inference_backend_error.png` (Backend error handling)
- `docs/screenshots/S06/training_backend_error.png` (Backend error handling)
