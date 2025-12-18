import { useMemo } from 'react';
import '../styles/help-center.css';
import type { TabId } from '../types';

export type QuickstartChecklist<TabType> = {
  id: string;
  title: string;
  tab: TabType;
  description: string;
  steps: string[];
  docLink?: string;
};

export type HelpTourStep<TabType> = {
  id: string;
  title: string;
  description: string;
  tab?: TabType;
};

const GLOSSARY = [
  {
    term: 'Workspace',
    definition:
      'A job-specific folder under ui_data/jobs/<id>/ that stores inputs, intermediate artifacts, logs, and final outputs.',
  },
  {
    term: 'Overrides',
    definition:
      'JSON patches that temporarily adjust pipeline_config.yaml or config.yaml values without editing files on disk.',
  },
  {
    term: 'Diarization',
    definition: 'Speaker-attribution step produced by WhisperX; toggle in inference when multi-speaker audio is present.',
  },
  {
    term: 'Alignment',
    definition: 'The process of mapping edited transcript tokens onto ASR words to recover timestamps for training and inference.',
  },
];

type Props = {
  isOpen: boolean;
  onClose: () => void;
  onSelectTab: (tab: TabId) => void;
  quickstarts: QuickstartChecklist<TabId>[];
  tourSteps: HelpTourStep<TabId>[];
  activeTourIndex: number | null;
  onStartTour: () => void;
  onNextStep: () => void;
  onPreviousStep: () => void;
  onEndTour: () => void;
};

export function HelpCenter({
  isOpen,
  onClose,
  onSelectTab,
  quickstarts,
  tourSteps,
  activeTourIndex,
  onStartTour,
  onNextStep,
  onPreviousStep,
  onEndTour,
}: Props) {
  const activeTour = useMemo(() => {
    if (activeTourIndex === null) return null;
    return tourSteps[activeTourIndex] ?? null;
  }, [activeTourIndex, tourSteps]);

  if (!isOpen) return null;

  return (
    <div className="help-overlay" role="dialog" aria-modal="true">
      <div className="help-drawer">
        <header className="help-header">
          <div>
            <p className="eyebrow">Assisted onboarding</p>
            <h2>Help center</h2>
            <p className="muted">Glossary, quickstart checklists, and a guided tour for the ISCE web console.</p>
          </div>
          <button type="button" className="close" onClick={onClose} aria-label="Close help center">
            ✕
          </button>
        </header>

        <section className="help-section">
          <div className="section-header">
            <h3>Quickstart checklists</h3>
            <p className="muted">Jump directly to the relevant tab with curated steps.</p>
          </div>
          <div className="quickstart-grid">
            {quickstarts.map((item) => (
              <article key={item.id} className="quickstart-card">
                <header>
                  <div>
                    <p className="eyebrow">{item.title}</p>
                    <p className="muted">{item.description}</p>
                  </div>
                  <button type="button" className="link" onClick={() => onSelectTab(item.tab)}>
                    Open tab
                  </button>
                </header>
                <ol>
                  {item.steps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
                {item.docLink ? (
                  <a href={item.docLink} className="inline-link" target="_blank" rel="noreferrer">
                    Open documentation
                  </a>
                ) : null}
              </article>
            ))}
          </div>
        </section>

        <section className="help-section">
          <div className="section-header">
            <h3>Glossary</h3>
            <p className="muted">Short, front-of-mind definitions pulled from README.md and FRONTEND.md.</p>
          </div>
          <div className="glossary-grid">
            {GLOSSARY.map((entry) => (
              <article key={entry.term} className="glossary-card">
                <p className="eyebrow">{entry.term}</p>
                <p>{entry.definition}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="help-section">
          <div className="section-header">
            <h3>Guided tour</h3>
            <p className="muted">Follow the tour to understand each workflow surface.</p>
          </div>
          {activeTour ? (
            <div className="tour-card">
              <div>
                <p className="eyebrow">Step {activeTourIndex! + 1} of {tourSteps.length}</p>
                <h4>{activeTour.title}</h4>
                <p className="muted">{activeTour.description}</p>
              </div>
              <div className="tour-actions">
                <button type="button" className="ghost" onClick={onPreviousStep}>
                  Previous
                </button>
                <button type="button" className="ghost" onClick={onNextStep}>
                  Next
                </button>
                <button type="button" className="primary" onClick={onEndTour}>
                  End tour
                </button>
              </div>
            </div>
          ) : (
            <div className="tour-card">
              <p className="muted">Launch a lightweight, five-step walkthrough of the UI.</p>
              <button type="button" className="primary" onClick={onStartTour}>
                Start guided tour
              </button>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
