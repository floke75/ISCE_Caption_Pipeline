import { FormEvent, useState } from "react";
import { createInferenceJob, InferencePayload } from "../api";
import { ConfigMap, JobRecord } from "../types";
import Card from "./Card";
import JsonEditor from "./JsonEditor";

interface InferenceFormProps {
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

export function InferenceForm({ onJobCreated }: InferenceFormProps) {
  const [mediaPath, setMediaPath] = useState("");
  const [transcriptPath, setTranscriptPath] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [outputBasename, setOutputBasename] = useState("");
  const [pipelineOverrides, setPipelineOverrides] = useState("");
  const [modelOverrides, setModelOverrides] = useState("");
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);

    const pipeline = parseOverrides(pipelineOverrides);
    const model = parseOverrides(modelOverrides);
    setPipelineError(pipeline.error);
    setModelError(model.error);
    if (pipeline.error || model.error) {
      return;
    }

    const payload: InferencePayload = {
      mediaPath: mediaPath.trim(),
      transcriptPath: transcriptPath.trim(),
      outputBasename: outputBasename.trim() || undefined,
      outputPath: outputPath.trim() || undefined,
      pipelineOverrides: pipeline.value ?? undefined,
      modelOverrides: model.value ?? undefined
    };

    if (!payload.mediaPath || !payload.transcriptPath) {
      setError("Media path and transcript path are required");
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
            required
          />
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
        <JsonEditor
          label="Pipeline overrides (JSON)"
          value={pipelineOverrides}
          onChange={setPipelineOverrides}
          error={pipelineError}
          placeholder='{"pipeline_root": "/tmp/pipeline"}'
        />
        <JsonEditor
          label="Model overrides (JSON)"
          value={modelOverrides}
          onChange={setModelOverrides}
          error={modelError}
          placeholder='{"sliders": {"flow": 1.1}}'
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
