# Embedded Help Center Design (S15)

## Objective
Provide users with contextual help, terminology definitions, and onboarding guidance directly within the application, reducing the need to consult external `README.md` files.

## Features

### 1. Help Sidebar (Slide-out)
A persistent "?" icon in the header toggles a right-side panel.
**Content Sections:**
*   **Quick Start:** "How to run your first job" checklist.
*   **Glossary:** Definitions for "Beam Search", "Diarization", "Enriched JSON", "Training Pair".
*   **Troubleshooting:** Common errors (e.g., "FFMPEG missing", "Path not found").
*   **Keyboard Shortcuts:** If any.

### 2. Contextual "What is this?" Triggers
Small `(i)` icons next to complex section headers (e.g., "Segmentation Overrides") that open the relevant Help Center section.

### 3. Guided Tour (Onboarding)
A lightweight overlay that highlights key UI elements for first-time users.
*   **Step 1:** Point to "Inference" tab -> "Start here to process media."
*   **Step 2:** Point to "Job Board" -> "Track progress here."
*   **Step 3:** Point to "Configuration" -> "Advanced settings."

## UI Design

**Sidebar Component:**
*   Fixed position `right: 0`, `top: 0`, `height: 100vh`.
*   Width: `350px`.
*   White background, shadow, z-index `1000`.
*   Tabs: [Guide] [Glossary] [FAQ].

**Tour Implementation:**
*   Since `react-joyride` is not in `package.json` and I should avoid new deps if possible, I will implement a **simple custom tour overlay**.
*   **Overlay:** A `div` with a "spotlight" hole (using `box-shadow`) and a tooltip.
*   **State:** Managed by `HelpContext` (current step, active status).

## Data Source
Content will be hardcoded in a `helpContent.ts` file, derived from the repository's `README.md` and `AGENTS.md` (sanitized for end-users).

## Implementation Plan (S15b)

1.  **Content Module (`ui/frontend/src/help/content.ts`):**
    *   Export `GLOSSARY`, `FAQ`, `QUICKSTART`.
2.  **Context (`HelpContext.tsx`):**
    *   `isOpen`, `toggle()`, `tourStep`, `nextTourStep()`.
3.  **Component (`HelpSidebar.tsx`):**
    *   The slide-out panel.
4.  **Component (`TourOverlay.tsx`):**
    *   The spotlight UI.
5.  **Integration:**
    *   Add `HelpProvider` to `App.tsx`.
    *   Add Trigger Button to `Header`.
