import { FormEvent, useState } from "react";
import OverridesInput from "./OverridesInput";

interface Props {
  onSubmit: (payload: Record<string, unknown>) => Promise<void>;
}

export function InferenceForm({ onSubmit }: Props) {
  const [mediaPath, setMediaPath] = useState("");
  const [transcriptPath, setTranscriptPath] = useState("");
  const [pythonExecutable, setPythonExecutable] = useState("");
  const [overrides, setOverrides] = useState<Record<string, unknown> | undefined>();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setSuccess(false);
    try {
      await onSubmit({
        mediaPath,
        transcriptPath,
        pythonExecutable: pythonExecutable || undefined,
        configOverrides: overrides,
      });
      setSuccess(true);
    } catch (err: any) {
      setError(err?.message ?? "Failed to queue inference job");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card">
      <h2>Inference run</h2>
      <p className="description">Run the full pipeline for a media file and transcript pair.</p>
      <form className="form-grid" onSubmit={handleSubmit}>
        <label>
          <span>Media file</span>
          <input value={mediaPath} onChange={(e) => setMediaPath(e.target.value)} required placeholder="/path/to/video.mp4" />
        </label>
        <label>
          <span>Transcript (.txt)</span>
          <input value={transcriptPath} onChange={(e) => setTranscriptPath(e.target.value)} required placeholder="/path/to/transcript.txt" />
        </label>
        <label>
          <span>Python interpreter (optional)</span>
          <input value={pythonExecutable} onChange={(e) => setPythonExecutable(e.target.value)} placeholder="/usr/bin/python3" />
        </label>
        <OverridesInput onChange={setOverrides} />
        <button className="btn-primary" type="submit" disabled={submitting}>
          {submitting ? "Starting..." : "Start inference"}
        </button>
        {error && <div className="alert error">{error}</div>}
        {success && <div className="alert success">Job queued</div>}
      </form>
    </div>
  );
}

export default InferenceForm;
