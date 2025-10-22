import { FormEvent, useState } from "react";
import OverridesInput from "./OverridesInput";

interface Props {
  onSubmit: (payload: Record<string, unknown>) => Promise<void>;
}

export function TrainingForm({ onSubmit }: Props) {
  const [corpusDir, setCorpusDir] = useState("");
  const [iterations, setIterations] = useState(3);
  const [errorBoost, setErrorBoost] = useState(1.0);
  const [pythonExecutable, setPythonExecutable] = useState("");
  const [overrides, setOverrides] = useState<Record<string, unknown> | undefined>();
  const [status, setStatus] = useState<{ error?: string; success?: boolean }>({});
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setStatus({});
    try {
      await onSubmit({
        corpusDir,
        iterations,
        errorBoostFactor: errorBoost,
        pythonExecutable: pythonExecutable || undefined,
        configOverrides: overrides,
      });
      setStatus({ success: true });
    } catch (error: any) {
      setStatus({ error: error?.message ?? "Failed to queue training job" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card">
      <h2>Model training</h2>
      <p className="description">Kick off iterative reweighting to rebuild model weights and constraints.</p>
      <form className="form-grid" onSubmit={handleSubmit}>
        <label>
          <span>Corpus directory</span>
          <input value={corpusDir} onChange={(e) => setCorpusDir(e.target.value)} required placeholder="/path/to/corpus" />
        </label>
        <label>
          <span>Iterations</span>
          <input type="number" min={1} max={20} value={iterations} onChange={(e) => setIterations(Number(e.target.value))} />
        </label>
        <label>
          <span>Error boost factor</span>
          <input type="number" step="0.1" min={0} value={errorBoost} onChange={(e) => setErrorBoost(Number(e.target.value))} />
        </label>
        <label>
          <span>Python interpreter (optional)</span>
          <input value={pythonExecutable} onChange={(e) => setPythonExecutable(e.target.value)} placeholder="/usr/bin/python3" />
        </label>
        <OverridesInput onChange={setOverrides} />
        <button className="btn-primary" type="submit" disabled={submitting}>
          {submitting ? "Starting..." : "Start training"}
        </button>
        {status.error && <div className="alert error">{status.error}</div>}
        {status.success && <div className="alert success">Job queued</div>}
      </form>
    </div>
  );
}

export default TrainingForm;
