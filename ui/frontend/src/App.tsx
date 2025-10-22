import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type JobStatus = "queued" | "running" | "succeeded" | "failed";

type JobSummary = {
  id: string;
  job_type: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
};

type JobDetail = JobSummary & {
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  log: string;
};

type ConfigValue = string | number | boolean | null | ConfigValue[] | { [key: string]: ConfigValue };
type ConfigData = { [key: string]: ConfigValue };

type TabKey = "inference" | "training" | "model" | "config";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });

  const contentType = response.headers.get("content-type") ?? "";

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    if (contentType.includes("application/json")) {
      const payload = await response.json();
      if (payload?.detail) {
        message = Array.isArray(payload.detail)
          ? payload.detail.map((item: unknown) => String(item)).join("; ")
          : String(payload.detail);
      }
    } else {
      const text = await response.text();
      if (text) {
        message = text;
      }
    }
    throw new Error(message);
  }

  if (contentType.includes("application/json")) {
    return (await response.json()) as T;
  }

  return (await response.text()) as T;
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function classForStatus(status: JobStatus): string {
  switch (status) {
    case "succeeded":
      return "status-pill status-success";
    case "failed":
      return "status-pill status-failed";
    case "running":
      return "status-pill status-running";
    default:
      return "status-pill";
  }
}

function updateAtPath(source: ConfigData, path: string[], value: ConfigValue): ConfigData {
  if (path.length === 0) {
    return source;
  }
  const [head, ...rest] = path;
  const clone: ConfigData = { ...source };
  if (rest.length === 0) {
    clone[head] = value;
    return clone;
  }
  const current = clone[head];
  if (current && typeof current === "object" && !Array.isArray(current)) {
    clone[head] = updateAtPath(current as ConfigData, rest, value) as ConfigValue;
  } else {
    clone[head] = updateAtPath({}, rest, value) as ConfigValue;
  }
  return clone;
}

function parseOverrideJSON(raw: string): Record<string, unknown> | undefined {
  if (!raw.trim()) {
    return undefined;
  }
  const parsed = JSON.parse(raw);
  if (parsed && typeof parsed === "object") {
    return parsed as Record<string, unknown>;
  }
  throw new Error("Overrides must evaluate to an object");
}

const TAB_LABELS: Record<TabKey, string> = {
  inference: "Manual Inference",
  training: "Training Pair Builder",
  model: "Model Training",
  config: "Configuration"
};

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  succeeded: "Completed",
  failed: "Failed"
};

function InferenceForm({ onJobCreated }: { onJobCreated: (jobId: string) => void }) {
  const [mediaFile, setMediaFile] = useState("");
  const [transcriptFile, setTranscriptFile] = useState("");
  const [overrides, setOverrides] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setMessage(null);

    if (!mediaFile.trim()) {
      setError("Please provide a media file path.");
      setSubmitting(false);
      return;
    }

    try {
      const payload = {
        media_file: mediaFile.trim(),
        transcript_file: transcriptFile.trim() ? transcriptFile.trim() : undefined,
        config_overrides: overrides.trim() ? parseOverrideJSON(overrides) : undefined
      };
      const response = await apiFetch<JobSummary>("/api/jobs/inference", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setMessage("Inference job submitted successfully.");
      onJobCreated(response.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="form" onSubmit={handleSubmit}>
      <h2>Manual Inference</h2>
      <p className="helper-text">
        Provide a media file to run the three-stage inference workflow. Optionally attach a
        transcript; if omitted the pipeline will fall back to ASR-only mode.
      </p>
      <label>
        Media file path
        <input
          type="text"
          value={mediaFile}
          onChange={(event) => setMediaFile(event.target.value)}
          placeholder="/path/to/video.mp4"
          required
        />
      </label>
      <label>
        Transcript (.txt)
        <input
          type="text"
          value={transcriptFile}
          onChange={(event) => setTranscriptFile(event.target.value)}
          placeholder="Optional /path/to/transcript.txt"
        />
      </label>
      <label>
        Advanced overrides (JSON)
        <textarea
          value={overrides}
          onChange={(event) => setOverrides(event.target.value)}
          placeholder='{"txt_placement_folder": "/tmp/custom"}'
          rows={3}
        />
      </label>
      {error && <p className="error-text">{error}</p>}
      {message && <p className="success-text">{message}</p>}
      <button className="primary" type="submit" disabled={submitting}>
        {submitting ? "Submitting..." : "Launch inference"}
      </button>
    </form>
  );
}

function TrainingPairForm({ onJobCreated }: { onJobCreated: (jobId: string) => void }) {
  const [mediaFile, setMediaFile] = useState("");
  const [srtFile, setSrtFile] = useState("");
  const [overrides, setOverrides] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setMessage(null);

    if (!mediaFile.trim() || !srtFile.trim()) {
      setError("Media and SRT file paths are required.");
      setSubmitting(false);
      return;
    }

    try {
      const payload = {
        media_file: mediaFile.trim(),
        srt_file: srtFile.trim(),
        config_overrides: overrides.trim() ? parseOverrideJSON(overrides) : undefined
      };
      const response = await apiFetch<JobSummary>("/api/jobs/training-pair", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setMessage("Training pair job submitted successfully.");
      onJobCreated(response.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="form" onSubmit={handleSubmit}>
      <h2>Training Pair Builder</h2>
      <p className="helper-text">
        Align a ground-truth SRT with its media file to create an enriched training JSON.
      </p>
      <label>
        Media file path
        <input
          type="text"
          value={mediaFile}
          onChange={(event) => setMediaFile(event.target.value)}
          placeholder="/path/to/video.mp4"
          required
        />
      </label>
      <label>
        SRT file path
        <input
          type="text"
          value={srtFile}
          onChange={(event) => setSrtFile(event.target.value)}
          placeholder="/path/to/ground_truth.srt"
          required
        />
      </label>
      <label>
        Advanced overrides (JSON)
        <textarea
          value={overrides}
          onChange={(event) => setOverrides(event.target.value)}
          placeholder='{"intermediate_dir": "/tmp/intermediate"}'
          rows={3}
        />
      </label>
      {error && <p className="error-text">{error}</p>}
      {message && <p className="success-text">{message}</p>}
      <button className="primary" type="submit" disabled={submitting}>
        {submitting ? "Submitting..." : "Launch training pair job"}
      </button>
    </form>
  );
}

function ModelTrainingForm({ onJobCreated }: { onJobCreated: (jobId: string) => void }) {
  const [corpusDir, setCorpusDir] = useState("");
  const [constraintsPath, setConstraintsPath] = useState("constraints.json");
  const [weightsPath, setWeightsPath] = useState("model_weights.json");
  const [configPath, setConfigPath] = useState("config.yaml");
  const [iterations, setIterations] = useState(3);
  const [errorBoost, setErrorBoost] = useState(1.0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setMessage(null);

    if (!corpusDir.trim()) {
      setError("Corpus directory is required.");
      setSubmitting(false);
      return;
    }

    try {
      const payload = {
        corpus_dir: corpusDir.trim(),
        constraints_path: constraintsPath.trim(),
        weights_path: weightsPath.trim(),
        config_path: configPath.trim(),
        iterations,
        error_boost_factor: errorBoost
      };
      const response = await apiFetch<JobSummary>("/api/jobs/model-training", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      setMessage("Model training job submitted successfully.");
      onJobCreated(response.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="form" onSubmit={handleSubmit}>
      <h2>Model Training</h2>
      <p className="helper-text">
        Run the full iterative weighting trainer using an engineered corpus directory.
      </p>
      <label>
        Corpus directory
        <input
          type="text"
          value={corpusDir}
          onChange={(event) => setCorpusDir(event.target.value)}
          placeholder="/path/to/_intermediate/_training"
          required
        />
      </label>
      <label>
        Constraints output path
        <input
          type="text"
          value={constraintsPath}
          onChange={(event) => setConstraintsPath(event.target.value)}
          placeholder="constraints.json"
        />
      </label>
      <label>
        Weights output path
        <input
          type="text"
          value={weightsPath}
          onChange={(event) => setWeightsPath(event.target.value)}
          placeholder="model_weights.json"
        />
      </label>
      <label>
        Config path
        <input
          type="text"
          value={configPath}
          onChange={(event) => setConfigPath(event.target.value)}
        />
      </label>
      <div className="field-grid">
        <label>
          Iterations
          <input
            type="number"
            min={1}
            value={iterations}
            onChange={(event) => setIterations(Number(event.target.value))}
          />
        </label>
        <label>
          Error boost factor
          <input
            type="number"
            min={0}
            step="0.1"
            value={errorBoost}
            onChange={(event) => setErrorBoost(Number(event.target.value))}
          />
        </label>
      </div>
      {error && <p className="error-text">{error}</p>}
      {message && <p className="success-text">{message}</p>}
      <button className="primary" type="submit" disabled={submitting}>
        {submitting ? "Submitting..." : "Launch model training"}
      </button>
    </form>
  );
}

function ConfigPanel() {
  return (
    <div className="config-panel">
      <ConfigEditor
        title="Pipeline configuration"
        endpoint="/config/pipeline"
        description="Manage directories, hot folders, and alignment parameters used across the orchestrator."
      />
      <ConfigEditor
        title="Segmentation configuration"
        endpoint="/config/model"
        description="Tune beam search, stylistic sliders, and model weight locations."
      />
    </div>
  );
}

function ConfigEditor({
  title,
  endpoint,
  description
}: {
  title: string;
  endpoint: "/config/pipeline" | "/config/model";
  description: string;
}) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ConfigData>({});
  const [working, setWorking] = useState<ConfigData>({});
  const [path, setPath] = useState<string>("");

  const loadConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await apiFetch<{ path: string; data: ConfigData }>(`/api${endpoint}`);
      const fresh = payload.data ?? {};
      setPath(payload.path);
      setData(fresh);
      setWorking(JSON.parse(JSON.stringify(fresh)) as ConfigData);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const handleChange = (changePath: string[], value: ConfigValue) => {
    setWorking((prev) => updateAtPath(prev, changePath, value));
  };

  const handleReset = () => {
    setWorking(JSON.parse(JSON.stringify(data)) as ConfigData);
    setError(null);
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const payload = await apiFetch<{ path: string; data: ConfigData }>(`/api${endpoint}`, {
        method: "PUT",
        body: JSON.stringify(working)
      });
      const fresh = payload.data ?? {};
      setPath(payload.path);
      setData(fresh);
      setWorking(JSON.parse(JSON.stringify(fresh)) as ConfigData);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const handleDownload = async () => {
    const response = await fetch(`${API_BASE}/api${endpoint}?format=yaml`);
    const text = await response.text();
    const blob = new Blob([text], { type: "text/yaml" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = endpoint === "/config/pipeline" ? "pipeline_config.yaml" : "config.yaml";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  };

  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>{title}</h2>
          <p className="helper-text">{description}</p>
          {path && <span className="config-path">Editing {path}</span>}
        </div>
        <div className="config-actions">
          <button type="button" onClick={handleDownload} className="ghost">
            Download YAML
          </button>
          <button type="button" onClick={handleReset} className="ghost">
            Reset
          </button>
          <button type="button" onClick={handleSave} className="primary" disabled={saving}>
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </header>
      {loading ? (
        <p>Loading configuration…</p>
      ) : error ? (
        <p className="error-text">{error}</p>
      ) : (
        <div className="config-fields">
          <ConfigFields data={working} path={[]} onChange={handleChange} />
        </div>
      )}
    </section>
  );
}

type ConfigFieldsProps = {
  data: ConfigData;
  path: string[];
  onChange: (path: string[], value: ConfigValue) => void;
};

function ConfigFields({ data, path, onChange }: ConfigFieldsProps) {
  return (
    <div className="config-fieldset">
      {Object.entries(data).map(([key, value]) => {
        const nextPath = [...path, key];
        if (value !== null && typeof value === "object" && !Array.isArray(value)) {
          return (
            <fieldset key={nextPath.join(".")}>
              <legend>{key}</legend>
              <ConfigFields data={value as ConfigData} path={nextPath} onChange={onChange} />
            </fieldset>
          );
        }
        if (Array.isArray(value)) {
          const display = value.map((item) => String(item)).join("\n");
          return (
            <label key={nextPath.join(".")}>
              {key}
              <textarea
                value={display}
                onChange={(event) => {
                  const lines = event.target.value
                    .split("\n")
                    .map((line) => line.trim())
                    .filter((line) => line.length > 0);
                  onChange(nextPath, lines as ConfigValue);
                }}
                rows={Math.max(2, value.length)}
              />
            </label>
          );
        }
        if (typeof value === "number") {
          const step = Number.isInteger(value) ? 1 : 0.1;
          return (
            <label key={nextPath.join(".")}>
              {key}
              <input
                type="number"
                value={value}
                step={step}
                onChange={(event) => onChange(nextPath, Number(event.target.value))}
              />
            </label>
          );
        }
        if (typeof value === "boolean") {
          return (
            <label key={nextPath.join(".")} className="checkbox">
              <input
                type="checkbox"
                checked={value}
                onChange={(event) => onChange(nextPath, event.target.checked)}
              />
              <span>{key}</span>
            </label>
          );
        }
        return (
          <label key={nextPath.join(".")}>
            {key}
            <input
              type="text"
              value={value === null ? "" : String(value)}
              onChange={(event) => {
                const raw = event.target.value;
                if (value === null && raw.trim() === "") {
                  onChange(nextPath, null);
                } else {
                  onChange(nextPath, raw);
                }
              }}
            />
          </label>
        );
      })}
    </div>
  );
}

function JobMonitor({
  jobs,
  selectedJobId,
  onSelect,
  detail,
  error
}: {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelect: (jobId: string) => void;
  detail: JobDetail | null;
  error: string | null;
}) {
  const sorted = useMemo(() => {
    return [...jobs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  }, [jobs]);

  const handleCopyLog = async () => {
    if (!detail?.log || !navigator.clipboard) {
      return;
    }
    try {
      await navigator.clipboard.writeText(detail.log);
    } catch (err) {
      console.warn("Unable to copy log", err);
    }
  };

  return (
    <section className="panel jobs-panel">
      <header className="panel-header">
        <div>
          <h2>Job monitor</h2>
          <p className="helper-text">Queued jobs refresh automatically every few seconds.</p>
        </div>
      </header>
      {error && <p className="error-text">{error}</p>}
      <div className="job-table-wrapper">
        <table className="job-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Type</th>
              <th>Status</th>
              <th>Submitted</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((job) => (
              <tr
                key={job.id}
                onClick={() => onSelect(job.id)}
                className={job.id === selectedJobId ? "selected" : ""}
              >
                <td title={job.id}>{job.id.slice(0, 8)}</td>
                <td>{job.job_type.replace(/_/g, " ")}</td>
                <td>
                  <span className={classForStatus(job.status)}>{STATUS_LABEL[job.status]}</span>
                </td>
                <td>{formatDateTime(job.created_at)}</td>
              </tr>
            ))}
            {sorted.length === 0 && (
              <tr>
                <td colSpan={4}>No jobs submitted yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="job-detail">
        {detail ? (
          <div>
            <header className="job-detail-header">
              <div>
                <h3>Job {detail.id.slice(0, 8)}</h3>
                <p className="helper-text">Last updated {formatDateTime(detail.updated_at)}</p>
              </div>
              <span className={classForStatus(detail.status)}>{STATUS_LABEL[detail.status]}</span>
            </header>
            <div className="job-detail-grid">
              <div>
                <h4>Parameters</h4>
                <pre>{JSON.stringify(detail.params, null, 2)}</pre>
              </div>
              <div>
                <h4>Result</h4>
                <pre>{detail.result ? JSON.stringify(detail.result, null, 2) : "Pending"}</pre>
              </div>
            </div>
            {detail.error && <p className="error-text">{detail.error}</p>}
            <div className="log-header">
              <h4>Live log</h4>
              <button type="button" className="ghost" onClick={handleCopyLog} disabled={!detail.log}>
                Copy log
              </button>
            </div>
            <pre className="log-viewer">{detail.log || "No output yet."}</pre>
          </div>
        ) : (
          <p className="helper-text">Select a job to see detailed output and results.</p>
        )}
      </div>
    </section>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("inference");
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      const data = await apiFetch<JobSummary[]>("/api/jobs");
      setJobs(data);
      setJobError(null);
    } catch (err) {
      setJobError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    fetchJobs();
    const timer = setInterval(fetchJobs, 3000);
    return () => clearInterval(timer);
  }, [fetchJobs]);

  const fetchDetail = useCallback(async () => {
    if (!selectedJobId) {
      setJobDetail(null);
      return;
    }
    try {
      const data = await apiFetch<JobDetail>(`/api/jobs/${selectedJobId}`);
      setJobDetail(data);
      setJobError(null);
    } catch (err) {
      setJobError(err instanceof Error ? err.message : String(err));
    }
  }, [selectedJobId]);

  useEffect(() => {
    fetchDetail();
    if (!selectedJobId) {
      return;
    }
    const timer = setInterval(fetchDetail, 2000);
    return () => clearInterval(timer);
  }, [fetchDetail, selectedJobId]);

  const handleJobCreated = (jobId: string) => {
    setSelectedJobId(jobId);
    fetchJobs();
    fetchDetail();
  };

  const renderActiveTab = () => {
    switch (activeTab) {
      case "inference":
        return <InferenceForm onJobCreated={handleJobCreated} />;
      case "training":
        return <TrainingPairForm onJobCreated={handleJobCreated} />;
      case "model":
        return <ModelTrainingForm onJobCreated={handleJobCreated} />;
      case "config":
        return <ConfigPanel />;
      default:
        return null;
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>ISCE Pipeline Control Panel</h1>
          <p className="helper-text">
            Launch inference and training runs, tweak configuration, and monitor progress without touching the command line.
          </p>
        </div>
      </header>
      <main className="app-main">
        <section className="left-pane">
          <nav className="tab-bar">
            {(Object.keys(TAB_LABELS) as TabKey[]).map((tab) => (
              <button
                key={tab}
                type="button"
                className={tab === activeTab ? "tab active" : "tab"}
                onClick={() => setActiveTab(tab)}
              >
                {TAB_LABELS[tab]}
              </button>
            ))}
          </nav>
          <div className="panel">{renderActiveTab()}</div>
        </section>
        <JobMonitor
          jobs={jobs}
          selectedJobId={selectedJobId}
          onSelect={setSelectedJobId}
          detail={jobDetail}
          error={jobError}
        />
      </main>
    </div>
  );
}

