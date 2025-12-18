import { useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { InferenceForm } from './components/InferenceForm';
import { TrainingPairForm } from './components/TrainingPairForm';
import { ModelTrainingForm } from './components/ModelTrainingForm';
import { ConfigPanel } from './components/ConfigPanel';
import { JobBoard } from './components/JobBoard';
import { HelpCenter, type HelpTourStep, type QuickstartChecklist } from './components/HelpCenter';
import type { TabId } from './types';
import './styles/app.css';

const TABS: { id: TabId; label: string }[] = [
  { id: 'inference', label: 'Inference' },
  { id: 'trainingPair', label: 'Training pairs' },
  { id: 'modelTraining', label: 'Model training' },
  { id: 'config', label: 'Configuration' }
];

const QUICKSTARTS: QuickstartChecklist<TabId>[] = [
  {
    id: 'inference-quickstart',
    title: 'Inference quickstart',
    tab: 'inference',
    description: 'Generate captions from a media file plus optional transcript.',
    steps: [
      'Pick a media file path and optional transcript',
      'Choose output directory or accept the default job workspace',
      'Adjust overrides (beam width, diarization) only if needed',
      'Submit and monitor the job for an SRT download link',
    ],
    docLink: 'https://github.com/floke75/ISCE_Caption_Pipeline/blob/main/FRONTEND.md#inference',
  },
  {
    id: 'training-quickstart',
    title: 'Training pair quickstart',
    tab: 'trainingPair',
    description: 'Align a gold SRT with ASR output to enrich the training corpus.',
    steps: [
      'Select SRT and media paths from the allowlist',
      'Confirm transcript and media names match (case-sensitive)',
      'Add operator notes for lineage and curation context',
      'Submit and watch for alignment artifacts in the job workspace',
    ],
    docLink: 'https://github.com/floke75/ISCE_Caption_Pipeline/blob/main/FRONTEND.md#training-pairs',
  },
  {
    id: 'model-training-quickstart',
    title: 'Model training quickstart',
    tab: 'modelTraining',
    description: 'Train or refresh model weights from curated training pairs.',
    steps: [
      'Point to the curated training corpus folder',
      'Set iterations and error boost based on corpus size',
      'Add operator notes describing the experiment goal',
      'Submit and capture the resulting weights/constraints paths',
    ],
    docLink: 'https://github.com/floke75/ISCE_Caption_Pipeline/blob/main/FRONTEND.md#model-training',
  },
];

const HELP_TOUR_STEPS: HelpTourStep<TabId>[] = [
  {
    id: 'welcome',
    title: 'Welcome to ISCE',
    description: 'Use the tabs to switch between inference, training pair alignment, model training, and configuration.',
  },
  {
    id: 'inference',
    title: 'Inference setup',
    tab: 'inference',
    description: 'Provide media plus optional transcript. Use overrides to tweak beam search and diarization without editing YAML.',
  },
  {
    id: 'training',
    title: 'Training pair alignment',
    tab: 'trainingPair',
    description: 'Align edited SRTs to ASR JSON to generate enriched tokens for the corpus.',
  },
  {
    id: 'model-training',
    title: 'Model training',
    tab: 'modelTraining',
    description: 'Iterate on weights and constraints; note experiments in Operator notes for reproducibility.',
  },
  {
    id: 'config',
    title: 'Configuration metadata',
    tab: 'config',
    description: 'Review loaded configs, overrides, and allowlisted paths. Use this tab to confirm pipeline wiring.',
  },
  {
    id: 'monitoring',
    title: 'Monitor jobs',
    description:
      'The Job monitor sidebar streams logs and artifacts for every workflow. Copy workspace paths to inspect outputs.',
  },
];

/**
 * The main application component for the ISCE Pipeline UI.
 *
 * This component serves as the root of the application, managing the main layout
 * and the primary navigation between different functional tabs. It renders the
 * header, the tabbed workbench area, and the persistent `JobBoard` sidebar.
 *
 * @returns {JSX.Element} The rendered application shell.
 */
export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('inference');
  const [helpOpen, setHelpOpen] = useState(false);
  const [tourIndex, setTourIndex] = useState<number | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    if (tourIndex === null) return;
    const step = HELP_TOUR_STEPS[tourIndex];
    if (step?.tab) {
      setActiveTab(step.tab);
    }
  }, [tourIndex]);

  const renderTab = useMemo(() => {
    switch (activeTab) {
      case 'inference':
        return <InferenceForm onJobCreated={() => queryClient.invalidateQueries({ queryKey: ['jobs'] })} />;
      case 'trainingPair':
        return <TrainingPairForm onJobCreated={() => queryClient.invalidateQueries({ queryKey: ['jobs'] })} />;
      case 'modelTraining':
        return <ModelTrainingForm onJobCreated={() => queryClient.invalidateQueries({ queryKey: ['jobs'] })} />;
      case 'config':
        return <ConfigPanel />;
      default:
        return null;
    }
  }, [activeTab, queryClient]);

  const endTour = () => setTourIndex(null);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>ISCE Pipeline Control Center</h1>
          <p>Run inference, build training corpora, and tune models without touching YAML files.</p>
        </div>
        <div className="header-meta">
          <span className="badge">Beta</span>
          <a href="https://github.com/floke75/ISCE_Caption_Pipeline" target="_blank" rel="noreferrer" className="link">
            Repository
          </a>
          <button type="button" className="link ghost" onClick={() => setHelpOpen(true)}>
            Help center
          </button>
        </div>
      </header>
      <main className="app-main">
        <section className="workbench">
          <nav className="tab-strip">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={tab.id === activeTab ? 'tab active' : 'tab'}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          <div className="panel">{renderTab}</div>
        </section>
        <aside className="job-column">
          <JobBoard />
        </aside>
      </main>
      <HelpCenter
        isOpen={helpOpen}
        tourSteps={HELP_TOUR_STEPS}
        activeTourIndex={tourIndex}
        quickstarts={QUICKSTARTS}
        onClose={() => {
          setHelpOpen(false);
          endTour();
        }}
        onSelectTab={(tab) => setActiveTab(tab)}
        onStartTour={() => setTourIndex(0)}
        onEndTour={endTour}
        onNextStep={() => setTourIndex((value) => (value === null ? 0 : Math.min(value + 1, HELP_TOUR_STEPS.length - 1)))}
        onPreviousStep={() =>
          setTourIndex((value) => {
            if (value === null) return 0;
            return value > 0 ? value - 1 : 0;
          })
        }
      />
    </div>
  );
}
