import { FormEvent, useState } from "react";
import { createModelTrainingJob, ModelTrainingPayload } from "../api";
import { ConfigMap, JobRecord } from "../types";
import Card from "./Card";
import JsonEditor from "./JsonEditor";

interface ModelTrainingFormProps {
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

export function ModelTrainingForm({ onJobCreated }: ModelTrainingFormProps) {
  const [corpusDir, setCorpusDir] = useState("");
  const [constraintsOutput, setConstraintsOutput] = useState("");
  const [weightsOutput, setWeightsOutput] = useState("");
  const [iterations, setIterations] = useState(3);
  const [errorBoostFactor, setErrorBoostFactor] = useState(1.0);
  const [modelOverrides, setModelOverrides] = useState("");
  const [modelError, setModelError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setMessage(null);

    const overrides = parseOverrides(modelOverrides);
    setModelError(overrides.error);
    if (overrides.error) {
      return;
    }

    const payload: ModelTrainingPayload = {
      corpusDir: corpusDir.trim(),
      constraintsOutput: constraintsOutput.trim() || undefined,
      weightsOutput: weightsOutput.trim() || undefined,
      iterations,
      errorBoostFactor,
      modelOverrides: overrides.value ?? undefined
    };

    if (!payload.corpusDir) {
      setError("Training corpus directory is required");
      return;
    }

    try {
      setIsSubmitting(true);
      const job = await createModelTrainingJob(payload);
      setMessage(`Training job ${job.id} queued`);
      onJobCreated(job);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card
      title="Train segmentation model"
      description="Run the iterative weighting loop on a corpus of enriched training files."
    >
      <form className="form" onSubmit={handleSubmit}>
        <label className="form-field">
          <span className="form-field__label">Training corpus directory</span>
          <input
            className="form-field__input"
            type="text"
            value={corpusDir}
            onChange={(event) => setCorpusDir(event.target.value)}
            placeholder="/path/to/corpus"
            required
          />
        </label>
        <div className="form__grid">
          <label className="form-field">
            <span className="form-field__label">Constraints output path</span>
            <input
              className="form-field__input"
              type="text"
              value={constraintsOutput}
              onChange={(event) => setConstraintsOutput(event.target.value)}
              placeholder="Defaults to job workspace"
            />
          </label>
          <label className="form-field">
            <span className="form-field__label">Model weights output path</span>
            <input
              className="form-field__input"
              type="text"
              value={weightsOutput}
              onChange={(event) => setWeightsOutput(event.target.value)}
              placeholder="Defaults to job workspace"
            />
          </label>
        </div>
        <div className="form__grid">
          <label className="form-field">
            <span className="form-field__label">Iterations</span>
            <input
              className="form-field__input"
              type="number"
              min={1}
              value={iterations}
              onChange={(event) => setIterations(Number(event.target.value))}
            />
          </label>
          <label className="form-field">
            <span className="form-field__label">Error boost factor</span>
            <input
              className="form-field__input"
              type="number"
              step="0.1"
              value={errorBoostFactor}
              onChange={(event) => setErrorBoostFactor(Number(event.target.value))}
            />
          </label>
        </div>
        <JsonEditor
          label="Model overrides (JSON)"
          value={modelOverrides}
          onChange={setModelOverrides}
          error={modelError}
          placeholder='{"sliders": {"density": 2.5}}'
        />
        {error && <div className="form__message form__message--error">{error}</div>}
        {message && <div className="form__message form__message--success">{message}</div>}
        <button type="submit" className="button" disabled={isSubmitting}>
          {isSubmitting ? "Launching…" : "Start training"}
        </button>
      </form>
    </Card>
  );
}

export default ModelTrainingForm;
