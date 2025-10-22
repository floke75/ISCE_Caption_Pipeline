import { FormEvent, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createModelTrainingJob, JobDetail } from '../../lib/api';

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
  const [configOverrides, setConfigOverrides] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

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
    if (!corpusDir.trim()) {
      setMessage({ type: 'error', text: 'Corpus directory is required.' });
      return;
    }

    try {
      const payload: Record<string, unknown> = {
        corpus_dir: corpusDir.trim(),
        constraints_path: constraintsPath.trim() || undefined,
        weights_path: weightsPath.trim() || undefined,
        iterations,
        error_boost_factor: boost,
        name: name.trim() || undefined,
        model_overrides: configOverrides ? JSON.parse(configOverrides) : {},
      };
      mutation.mutate(payload);
    } catch (err) {
      setMessage({ type: 'error', text: 'Overrides must be valid JSON.' });
    }
  };

  return (
    <form className="panel-body form-grid" onSubmit={handleSubmit}>
      <div className="field">
        <label>Friendly name (optional)</label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Quarterly retrain" />
      </div>
      <div className="field">
        <label>Training corpus directory</label>
        <input value={corpusDir} onChange={(e) => setCorpusDir(e.target.value)} placeholder="/data/corpus" required />
      </div>
      <div className="field">
        <label>Constraints output path (optional)</label>
        <input value={constraintsPath} onChange={(e) => setConstraintsPath(e.target.value)} placeholder="/models/constraints.json" />
      </div>
      <div className="field">
        <label>Model weights output path (optional)</label>
        <input value={weightsPath} onChange={(e) => setWeightsPath(e.target.value)} placeholder="/models/model_weights.json" />
      </div>
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
      <div className="field">
        <label>Config overrides (JSON)</label>
        <textarea
          value={configOverrides}
          onChange={(e) => setConfigOverrides(e.target.value)}
          placeholder={"{\n  \"sliders\": { \"balance\": 2.5 }\n}"}
        />
      </div>
      {message && <div className={`message-banner ${message.type}`}>{message.text}</div>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={mutation.isLoading}>
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
            setConfigOverrides('');
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
