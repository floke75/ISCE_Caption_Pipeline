import { FormEvent, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createModelTrainingJob, JobDetail } from '../../lib/api';
import OverrideEditor from '../OverrideEditor';
import {
  PathFieldValidation,
  ValidatedPathField,
} from '../inputs/ValidatedPathField';

interface ModelTrainingFormProps {
  onCreated(job: JobDetail): void;
}

export function ModelTrainingForm({ onCreated }: ModelTrainingFormProps) {
  const queryClient = useQueryClient();
  const [corpusDir, setCorpusDir] = useState('');
  const [constraintsPath, setConstraintsPath] = useState('');
  const [weightsPath, setWeightsPath] = useState('');
  const [iterations, setIterations] = useState(3);
  const [boost, setBoost] = useState(1.0);
  const [name, setName] = useState('');
  const [modelOverrides, setModelOverrides] = useState<Record<string, unknown>>({});
  const [modelOverridesValid, setModelOverridesValid] = useState(true);
  const [editorKey, setEditorKey] = useState(0);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [corpusValidation, setCorpusValidation] = useState<PathFieldValidation>({
    valid: false,
    resolvedPath: null,
    message: 'Corpus directory is required.',
    checking: false,
  });
  const [constraintsValidation, setConstraintsValidation] = useState<PathFieldValidation>({
    valid: true,
    resolvedPath: null,
    message: null,
    checking: false,
  });
  const [weightsValidation, setWeightsValidation] = useState<PathFieldValidation>({
    valid: true,
    resolvedPath: null,
    message: null,
    checking: false,
  });

  const mutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => createModelTrainingJob(payload),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      setMessage({ type: 'success', text: 'Model training queued successfully.' });
      onCreated(job);
    },
    onError: (error) => setMessage({ type: 'error', text: (error as Error).message }),
  });

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    setMessage(null);
    if (corpusValidation.checking || constraintsValidation.checking || weightsValidation.checking) {
      setMessage({ type: 'error', text: 'Please wait for path validation to finish.' });
      return;
    }
    if (!corpusValidation.valid) {
      setMessage({
        type: 'error',
        text: corpusValidation.message ?? 'Corpus directory is invalid.',
      });
      return;
    }
    if (!constraintsValidation.valid) {
      setMessage({
        type: 'error',
        text: constraintsValidation.message ?? 'Constraints output path is invalid.',
      });
      return;
    }
    if (!weightsValidation.valid) {
      setMessage({
        type: 'error',
        text: weightsValidation.message ?? 'Model weights output path is invalid.',
      });
      return;
    }
    if (!modelOverridesValid) {
      setMessage({ type: 'error', text: 'Resolve override errors before submitting.' });
      return;
    }

    const normaliseOptional = (resolved: string | null, raw: string) => {
      const trimmed = raw.trim();
      return resolved ?? (trimmed ? trimmed : undefined);
    };

    const payload: Record<string, unknown> = {
      corpus_dir: corpusValidation.resolvedPath ?? corpusDir.trim(),
      constraints_path: normaliseOptional(constraintsValidation.resolvedPath, constraintsPath),
      weights_path: normaliseOptional(weightsValidation.resolvedPath, weightsPath),
      iterations,
      error_boost_factor: boost,
      name: name.trim() || undefined,
      model_overrides: modelOverrides,
    };
    mutation.mutate(payload);
  };

  const submittingDisabled =
    mutation.isLoading ||
    corpusValidation.checking ||
    constraintsValidation.checking ||
    weightsValidation.checking ||
    !corpusValidation.valid ||
    !constraintsValidation.valid ||
    !weightsValidation.valid ||
    !modelOverridesValid;

  return (
    <form className="panel-body form-grid" onSubmit={handleSubmit}>
      <div className="field">
        <label>Friendly name (optional)</label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Quarterly retrain" />
      </div>
      <ValidatedPathField
        label="Training corpus directory"
        value={corpusDir}
        onChange={setCorpusDir}
        kind="directory"
        required
        placeholder="/data/corpus"
        description="Training corpus directory"
        onValidation={setCorpusValidation}
      />
      <ValidatedPathField
        label="Constraints output path (optional)"
        value={constraintsPath}
        onChange={setConstraintsPath}
        kind="file"
        required={false}
        mustExist={false}
        allowCreate
        placeholder="/models/constraints.json"
        description="Constraints output file"
        onValidation={setConstraintsValidation}
      />
      <ValidatedPathField
        label="Model weights output path (optional)"
        value={weightsPath}
        onChange={setWeightsPath}
        kind="file"
        required={false}
        mustExist={false}
        allowCreate
        placeholder="/models/model_weights.json"
        description="Model weights output file"
        onValidation={setWeightsValidation}
      />
      <div className="field">
        <label>Iterations</label>
        <input
          type="number"
          min={1}
          max={20}
          value={iterations}
          onChange={(e) => setIterations(Number(e.target.value) || 1)}
        />
      </div>
      <div className="field">
        <label>Error boost factor</label>
        <input
          type="number"
          step="0.1"
          min={0}
          value={boost}
          onChange={(e) => setBoost(Number(e.target.value) || 0)}
        />
      </div>
      <OverrideEditor
        key={editorKey}
        configType="model"
        label="Segmentation config overrides"
        onChange={setModelOverrides}
        onValidityChange={setModelOverridesValid}
      />
      {message && <div className={`message-banner ${message.type}`}>{message.text}</div>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={submittingDisabled}>
          {mutation.isLoading ? 'Submitting…' : 'Start model training'}
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            setCorpusDir('');
            setConstraintsPath('');
            setWeightsPath('');
            setIterations(3);
            setBoost(1.0);
            setModelOverrides({});
            setModelOverridesValid(true);
            setEditorKey((value) => value + 1);
            setCorpusValidation({ valid: false, resolvedPath: null, message: 'Corpus directory is required.', checking: false });
            setConstraintsValidation({ valid: true, resolvedPath: null, message: null, checking: false });
            setWeightsValidation({ valid: true, resolvedPath: null, message: null, checking: false });
            setName('');
            setMessage(null);
          }}
        >
          Reset
        </button>
      </div>
    </form>
  );
}

export default ModelTrainingForm;
