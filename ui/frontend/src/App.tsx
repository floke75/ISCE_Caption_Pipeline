import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { useJobs } from "./hooks/useJobs";
import { useJobLog } from "./hooks/useJobLog";
import { usePipelineConfig } from "./hooks/usePipelineConfig";
import InferenceForm from "./components/InferenceForm";
import TrainingPairForm from "./components/TrainingPairForm";
import TrainingForm from "./components/TrainingForm";
import JobTable from "./components/JobTable";
import LogViewer from "./components/LogViewer";
import ConfigEditor from "./components/ConfigEditor";

const tabs = [
  { id: "operations", label: "Operations" },
  { id: "config", label: "Configuration" },
  { id: "jobs", label: "Activity" },
];

async function postJson(path: string, body: Record<string, unknown>) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Request failed");
  }
  return response.json();
}

export default function App() {
  const [tab, setTab] = useState<string>("operations");
  const queryClient = useQueryClient();
  const { data: jobs = [] } = useJobs();
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const { data: log = "" } = useJobLog(selectedJobId);
  const { data: config, isLoading: configLoading, updateConfig, refetch: refetchConfig, isUpdating } = usePipelineConfig();

  const runningJobs = useMemo(() => jobs.filter((job) => job.status === "running").length, [jobs]);

  useEffect(() => {
    if (!selectedJobId && jobs.length > 0) {
      setSelectedJobId(jobs[0].id);
    }
  }, [jobs, selectedJobId]);

  const handleInference = async (payload: Record<string, unknown>) => {
    await postJson("/api/jobs/inference", payload);
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    setTab("jobs");
  };

  const handleTrainingPair = async (payload: Record<string, unknown>) => {
    await postJson("/api/jobs/training-pair", payload);
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    setTab("jobs");
  };

  const handleTraining = async (payload: Record<string, unknown>) => {
    await postJson("/api/jobs/training", payload);
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    setTab("jobs");
  };

  const handleCopyLog = () => {
    if (!log) return;
    navigator.clipboard.writeText(log).catch(() => undefined);
  };

  return (
    <div className="main-container">
      <header style={{ marginBottom: "2rem" }}>
        <h1>ISCE Pipeline Control Center</h1>
        <p className="muted">Queue inference, build training corpora, and monitor jobs without touching YAML manually.</p>
        {runningJobs > 0 && <div className="alert success">{runningJobs} job(s) running</div>}
      </header>

      <nav className="tab-bar">
        {tabs.map((item) => (
          <button
            key={item.id}
            className={`tab-button ${tab === item.id ? "active" : ""}`}
            type="button"
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {tab === "operations" && (
        <div className="card-grid">
          <InferenceForm onSubmit={handleInference} />
          <TrainingPairForm onSubmit={handleTrainingPair} />
          <TrainingForm onSubmit={handleTraining} />
        </div>
      )}

      {tab === "config" && (
        <ConfigEditor
          config={config ?? {}}
          isLoading={configLoading}
          isUpdating={isUpdating}
          onSave={async (data) => {
            await updateConfig(data);
          }}
          onReload={refetchConfig}
        />
      )}

      {tab === "jobs" && (
        <div className="card-grid" style={{ gridTemplateColumns: "2fr 1fr" }}>
          <JobTable jobs={jobs} selectedId={selectedJobId} onSelect={(id) => setSelectedJobId(id)} />
          <LogViewer log={log} onCopy={handleCopyLog} />
        </div>
      )}
    </div>
  );
}
