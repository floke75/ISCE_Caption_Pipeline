import { FormEvent, useState } from "react";
import { createTrainingPairsJob, TrainingPairsPayload } from "../api";
import { ConfigMap, JobRecord } from "../types";
import Card from "./Card";
import JsonEditor from "./JsonEditor";

interface TrainingPairsFormProps {
  onJobCreated: (job: JobRecord) => void;
}

function parseOverrides(input: string): { value: ConfigMap | null; error: string | null } {
  const trimmed = input.trim();
  if (!trimmed) {
    return { value: null, error: null };
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { value: null, error: "Overrides must be a JSON object" };
    }
    return { value: parsed as ConfigMap, error: null };
  } catch (error) {
    return { value: null, error: (error as Error).message };
  }
}

export function TrainingPairsForm({ onJobCreated }: TrainingPairsFormProps) {
  const [transcriptPath, setTranscriptPath] = useState("");
  const [asrReferencePath, setAsrReferencePath] = useState("");
  const [outputBasename, setOutputBasename] = useState("");
  const [pipelineOverrides, setPipelineOverrides] = useState("");
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [asrOnlyMode, setAsrOnlyMode] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);

    const parsed = parseOverrides(pipelineOverrides);
    setPipelineError(parsed.error);
    if (parsed.error) {
      return;
    }

    const payload: TrainingPairsPayload = {
      transcriptPath: transcriptPath.trim(),
      asrReferencePath: asrReferencePath.trim(),
      outputBasename: outputBasename.trim() || undefined,
      asrOnlyMode,
      pipelineOverrides: parsed.value ?? undefined
    };

    if (!payload.transcriptPath || !payload.asrReferencePath) {
      setError("Transcript and ASR reference paths are required");
      return;
    }

    try {
      setIsSubmitting(true);
      const job = await createTrainingPairsJob(payload);
      setMessage(`Training pair job ${job.id} queued`);
      onJobCreated(job);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card
      title="Build training pairs"
      description="Convert edited subtitles and aligned ASR into enriched training corpora."
    >
      <form className="form" onSubmit={handleSubmit}>
        <label className="form-field">
          <span className="form-field__label">Transcript (.srt) path</span>
          <input
            className="form-field__input"
            type="text"
            value={transcriptPath}
            onChange={(event) => setTranscriptPath(event.target.value)}
            placeholder="/path/to/program.srt"
            required
          />
        </label>
        <label className="form-field">
          <span className="form-field__label">ASR reference (.json) path</span>
          <input
            className="form-field__input"
            type="text"
            value={asrReferencePath}
            onChange={(event) => setAsrReferencePath(event.target.value)}
            placeholder="/path/to/asr.visual.words.diar.json"
            required
          />
        </label>
        <div className="form__grid">
          <label className="form-field">
            <span className="form-field__label">Output base name override</span>
            <input
              className="form-field__input"
              type="text"
              value={outputBasename}
              onChange={(event) => setOutputBasename(event.target.value)}
              placeholder="Defaults to transcript name"
            />
          </label>
          <label className="form-field form-field--checkbox">
            <input
              type="checkbox"
              checked={asrOnlyMode}
              onChange={(event) => setAsrOnlyMode(event.target.checked)}
            />
            <span>ASR-only mode (skip transcript alignment)</span>
          </label>
        </div>
        <JsonEditor
          label="Pipeline overrides (JSON)"
          value={pipelineOverrides}
          onChange={setPipelineOverrides}
          error={pipelineError}
          placeholder='{"build_pair": {"emit_asr_style_training_copy": false}}'
        />
        {error && <div className="form__message form__message--error">{error}</div>}
        {message && <div className="form__message form__message--success">{message}</div>}
        <button type="submit" className="button" disabled={isSubmitting}>
          {isSubmitting ? "Launching…" : "Build training data"}
        </button>
      </form>
    </Card>
  );
}

export default TrainingPairsForm;
