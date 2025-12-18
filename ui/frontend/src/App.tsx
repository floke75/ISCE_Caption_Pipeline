import { useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Routes, Route } from 'react-router-dom';
import { InferenceForm } from './components/InferenceForm';
import { TrainingPairForm } from './components/TrainingPairForm';
import { ModelTrainingForm } from './components/ModelTrainingForm';
import { ConfigPanel } from './components/ConfigPanel';
import { JobBoard } from './components/JobBoard';
import { AlignmentViewer } from './components/AlignmentViewer';
import { ArtifactViewer } from './components/ArtifactViewer';
import { SystemStatus } from './components/SystemStatus';
import { HelpCenter } from './components/HelpCenter';
import './styles/app.css';

const TABS = [
  { id: 'inference', label: 'Inference' },
  { id: 'trainingPair', label: 'Training pairs' },
  { id: 'modelTraining', label: 'Model training' },
  { id: 'config', label: 'Configuration' }
] as const;

type TabId = (typeof TABS)[number]['id'];

function Dashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('inference');
  const queryClient = useQueryClient();

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

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>ISCE Pipeline Control Center</h1>
          <p>Run inference, build training corpora, and tune models without touching YAML files.</p>
        </div>
        <div className="header-meta">
          <SystemStatus />
          <HelpCenter />
          <span className="badge">Beta</span>
          <a href="https://github.com/floke75/ISCE_Caption_Pipeline" target="_blank" rel="noreferrer" className="link">
            Repository
          </a>
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
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/artifacts/view" element={<ArtifactViewer />} />
      <Route path="/jobs/alignment" element={<AlignmentViewer />} />
    </Routes>
  );
}
