import { FormEvent, useState } from "react";
import { createTrainingPairsJob, TrainingPairsPayload } from "../api";
import { ConfigMap, JobRecord } from "../types";
import Card from "./Card";
import OverridesEditor from "./OverridesEditor";
import PathPicker from "./PathPicker";

interface TrainingPairsFormProps {
  onJobCreated: (job: JobRecord) => void;
}

export function TrainingPairsForm({ onJobCreated }: TrainingPairsFormProps) {
  const [transcriptPath, setTranscriptPath] = useState("");
  const [asrReferencePath, setAsrReferencePath] = useState("");
  const [outputBasename, setOutputBasename] = useState("");
  const [pipelineOverrides, setPipelineOverrides] = useState<ConfigMap | null>(null);
  const [asrOnlyMode, setAsrOnlyMode] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);

    const payload: TrainingPairsPayload = {
      transcriptPath: transcriptPath.trim(),
      asrReferencePath: asrReferencePath.trim(),
      outputBasename: outputBasename.trim() || undefined,
      asrOnlyMode,
      pipelineOverrides: pipelineOverrides ?? undefined
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
        <PathPicker
          label="Transcript (.srt) path"
          value={transcriptPath}
          onChange={setTranscriptPath}
          placeholder="/path/to/program.srt"
          required
          expect="file"
        />
        <PathPicker
          label="ASR reference (.json) path"
          value={asrReferencePath}
          onChange={setAsrReferencePath}
          placeholder="/path/to/asr.visual.words.diar.json"
          required
          expect="file"
        />
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
        <OverridesEditor
          label="Pipeline overrides"
          kind="pipeline"
          value={pipelineOverrides}
          onChange={setPipelineOverrides}
          description="Adjust build_pair settings for this run without editing YAML."
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
