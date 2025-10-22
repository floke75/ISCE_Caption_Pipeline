import { FormEvent, useState } from "react";
import { createInferenceJob, InferencePayload } from "../api";
import { ConfigMap, JobRecord } from "../types";
import Card from "./Card";
import OverridesEditor from "./OverridesEditor";
import PathPicker from "./PathPicker";

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
        <PathPicker
          label="Media file path"
          value={mediaPath}
          onChange={setMediaPath}
          placeholder="/path/to/video.mp4"
          required
          expect="file"
        />
        <PathPicker
          label="Transcript file path"
          value={transcriptPath}
          onChange={setTranscriptPath}
          placeholder="/path/to/transcript.txt"
          required={!asrOnlyMode}
          disabled={asrOnlyMode}
          expect="file"
          description={asrOnlyMode ? "Disabled while ASR-only mode is enabled." : undefined}
        />
        <label className="form-field form-field--checkbox">
          <input
            type="checkbox"
            checked={asrOnlyMode}
            onChange={(event) => setAsrOnlyMode(event.target.checked)}
          />
          <span>ASR-only mode (skip transcript alignment)</span>
        </label>
        <div className="form__grid">
          <PathPicker
            label="Custom output SRT path (optional)"
            value={outputPath}
            onChange={setOutputPath}
            placeholder="/desired/output/file.srt"
            expect="any"
            description="Leave blank to store the result in the job artifacts folder."
          />
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
