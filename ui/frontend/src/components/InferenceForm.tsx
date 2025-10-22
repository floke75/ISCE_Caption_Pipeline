import { FormEvent, useState } from "react";
import { createInferenceJob, InferencePayload } from "../api";
import { ConfigMap, JobRecord } from "../types";
import Card from "./Card";
import OverridesEditor from "./OverridesEditor";

interface InferenceFormProps {
  onJobCreated: (job: JobRecord) => void;
}

export function InferenceForm({ onJobCreated }: InferenceFormProps) {
  const [mediaPath, setMediaPath] = useState("");
  const [transcriptPath, setTranscriptPath] = useState("");
  const [asrOnlyMode, setAsrOnlyMode] = useState(false);
  const [outputPath, setOutputPath] = useState("");
  const [outputBasename, setOutputBasename] = useState("");
  const [pipelineOverrides, setPipelineOverrides] = useState<ConfigMap | null>(null);
  const [modelOverrides, setModelOverrides] = useState<ConfigMap | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);

    const payload: InferencePayload = {
      mediaPath: mediaPath.trim(),
      transcriptPath: asrOnlyMode ? undefined : transcriptPath.trim() || undefined,
      outputBasename: outputBasename.trim() || undefined,
      outputPath: outputPath.trim() || undefined,
      pipelineOverrides: pipelineOverrides ?? undefined,
      modelOverrides: modelOverrides ?? undefined
    };

    if (!payload.mediaPath) {
      setError("Media path is required");
      return;
    }

    if (!asrOnlyMode && !transcriptPath.trim()) {
      setError("Provide a transcript path or enable ASR-only mode");
      return;
    }

    try {
      setIsSubmitting(true);
      const job = await createInferenceJob(payload);
      setMessage(`Inference job ${job.id} queued`);
      onJobCreated(job);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card
      title="Run inference"
      description="Generate ASR, enrich features, and produce SRT captions for a single media file."
    >
      <form className="form" onSubmit={handleSubmit}>
        <label className="form-field">
          <span className="form-field__label">Media file path</span>
          <input
            className="form-field__input"
            type="text"
            value={mediaPath}
            onChange={(event) => setMediaPath(event.target.value)}
            placeholder="/path/to/video.mp4"
            required
          />
        </label>
        <label className="form-field">
          <span className="form-field__label">Transcript file path</span>
          <input
            className="form-field__input"
            type="text"
            value={transcriptPath}
            onChange={(event) => setTranscriptPath(event.target.value)}
            placeholder="/path/to/transcript.txt"
            required={!asrOnlyMode}
            disabled={asrOnlyMode}
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
        <div className="form__grid">
          <label className="form-field">
            <span className="form-field__label">Custom output SRT path (optional)</span>
            <input
              className="form-field__input"
              type="text"
              value={outputPath}
              onChange={(event) => setOutputPath(event.target.value)}
              placeholder="/desired/output/file.srt"
            />
          </label>
          <label className="form-field">
            <span className="form-field__label">Output base name override</span>
            <input
              className="form-field__input"
              type="text"
              value={outputBasename}
              onChange={(event) => setOutputBasename(event.target.value)}
              placeholder="Defaults to media file name"
            />
          </label>
        </div>
        <OverridesEditor
          label="Pipeline overrides"
          kind="pipeline"
          value={pipelineOverrides}
          onChange={setPipelineOverrides}
          description="Modify the runtime pipeline configuration without editing YAML manually."
        />
        <OverridesEditor
          label="Model overrides"
          kind="model"
          value={modelOverrides}
          onChange={setModelOverrides}
          description="Override segmentation parameters for this job."
        />
        {error && <div className="form__message form__message--error">{error}</div>}
        {message && <div className="form__message form__message--success">{message}</div>}
        <button type="submit" className="button" disabled={isSubmitting}>
          {isSubmitting ? "Launching…" : "Start inference"}
        </button>
      </form>
    </Card>
  );
}

export default InferenceForm;
