import { FormEvent, useState } from "react";
import OverridesInput from "./OverridesInput";

interface Props {
  onSubmit: (payload: Record<string, unknown>) => Promise<void>;
}

export function TrainingPairForm({ onSubmit }: Props) {
  const [transcriptPath, setTranscriptPath] = useState("");
  const [asrReference, setAsrReference] = useState("");
  const [mode, setMode] = useState("inference");
  const [pythonExecutable, setPythonExecutable] = useState("");
  const [outputBasename, setOutputBasename] = useState("");
  const [overrides, setOverrides] = useState<Record<string, unknown> | undefined>();
  const [status, setStatus] = useState<{ error?: string; success?: boolean }>({});
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setStatus({});
    try {
      await onSubmit({
        transcriptPath,
        asrReference,
        mode,
        pythonExecutable: pythonExecutable || undefined,
        outputBasename: outputBasename || undefined,
        configOverrides: overrides,
      });
      setStatus({ success: true });
    } catch (error: any) {
      setStatus({ error: error?.message ?? "Failed to queue training pair job" });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card">
      <h2>Build training/enrichment data</h2>
      <p className="description">Generate enriched tokens for inference or training corpora.</p>
      <form className="form-grid" onSubmit={handleSubmit}>
        <label>
          <span>Transcript or SRT</span>
          <input value={transcriptPath} onChange={(e) => setTranscriptPath(e.target.value)} required placeholder="/path/to/file.srt" />
        </label>
        <label>
          <span>ASR reference JSON</span>
          <input value={asrReference} onChange={(e) => setAsrReference(e.target.value)} required placeholder="/path/to/file.asr.json" />
        </label>
        <div>
          <span>Mode</span>
          <div className="radio-group">
            <label className="inline-controls">
              <input type="radio" checked={mode === "inference"} onChange={() => setMode("inference")} />
              <span>Inference</span>
            </label>
            <label className="inline-controls">
              <input type="radio" checked={mode === "training"} onChange={() => setMode("training")} />
              <span>Training</span>
            </label>
          </div>
        </div>
        <label>
          <span>Output basename (optional)</span>
          <input value={outputBasename} onChange={(e) => setOutputBasename(e.target.value)} placeholder="Custom file stem" />
        </label>
        <label>
          <span>Python interpreter (optional)</span>
          <input value={pythonExecutable} onChange={(e) => setPythonExecutable(e.target.value)} placeholder="/usr/bin/python3" />
        </label>
        <OverridesInput onChange={setOverrides} />
        <button className="btn-primary" type="submit" disabled={submitting}>
          {submitting ? "Starting..." : "Start job"}
        </button>
        {status.error && <div className="alert error">{status.error}</div>}
        {status.success && <div className="alert success">Job queued</div>}
      </form>
    </div>
  );
}

export default TrainingPairForm;
