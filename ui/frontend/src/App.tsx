import { useCallback, useEffect, useState } from "react";
import { fetchJobs as loadJobs } from "./api";
import { JobRecord } from "./types";
import InferenceForm from "./components/InferenceForm";
import TrainingPairsForm from "./components/TrainingPairsForm";
import ModelTrainingForm from "./components/ModelTrainingForm";
import ConfigEditor from "./components/ConfigEditor";
import JobMonitor from "./components/JobMonitor";

function useJobPolling() {
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await loadJobs();
      setJobs(response);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  return { jobs, loading, error, refresh };
}

type TabId = "inference" | "training" | "model" | "config";

const tabs: { id: TabId; label: string; description: string }[] = [
  { id: "inference", label: "Inference", description: "Process a media + transcript pair" },
  { id: "training", label: "Training pairs", description: "Generate enriched corpora" },
  { id: "model", label: "Model training", description: "Fit statistical weights" },
  { id: "config", label: "Configuration", description: "Edit pipeline settings" }
];

export default function App() {
  const { jobs, error, refresh } = useJobPolling();
  const [activeTab, setActiveTab] = useState<TabId>("inference");

  const handleJobCreated = () => {
    refresh();
  };

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <h1>ISCE Pipeline Control Center</h1>
          <p className="muted">Launch inference, produce training data, tune models, and supervise progress from one panel.</p>
        </div>
      </header>
      <div className="app__layout">
        <main className="app__main">
          <nav className="tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={tab.id === activeTab ? "tabs__item tabs__item--active" : "tabs__item"}
                onClick={() => setActiveTab(tab.id)}
              >
                <span>{tab.label}</span>
                <small>{tab.description}</small>
              </button>
            ))}
          </nav>
          <div className="tab-content">
            {activeTab === "inference" && <InferenceForm onJobCreated={handleJobCreated} />}
            {activeTab === "training" && <TrainingPairsForm onJobCreated={handleJobCreated} />}
            {activeTab === "model" && <ModelTrainingForm onJobCreated={handleJobCreated} />}
            {activeTab === "config" && (
              <div className="config-grid">
                <ConfigEditor kind="pipeline" />
                <ConfigEditor kind="model" />
              </div>
            )}
          </div>
        </main>
        <aside className="app__sidebar">
          {error && <div className="form__message form__message--error">{error}</div>}
          <JobMonitor jobs={jobs} onRefresh={refresh} />
        </aside>
      </div>
    </div>
  );
}
