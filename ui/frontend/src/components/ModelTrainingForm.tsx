import { FormEvent, useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import client from '../api/client';
import { OverrideEditor } from './OverrideEditor';
import '../styles/forms.css';

type Props = {
  onJobCreated: () => void;
};

export function ModelTrainingForm({ onJobCreated }: Props) {
  const [corpusDir, setCorpusDir] = useState('');
  const [iterations, setIterations] = useState<number | ''>('');
  const [errorBoost, setErrorBoost] = useState<number | ''>('');
  const [notes, setNotes] = useState('');
  const [overridePatch, setOverridePatch] = useState<Record<string, unknown>>({});
  const [overrideInvalid, setOverrideInvalid] = useState(false);

  const mutation = useMutation({
    mutationFn: async () => {
      const payload: Record<string, unknown> = {
        corpus_dir: corpusDir,
      };
      if (iterations !== '') payload.iterations = iterations;
      if (errorBoost !== '') payload.error_boost_factor = errorBoost;
      if (notes) payload.notes = notes;
      if (Object.keys(overridePatch).length) {
        payload.config_overrides = overridePatch;
      }
      const { data } = await client.post('/jobs/model-training', payload);
      return data;
    },
    onSuccess: () => {
      toast.success('Model training job queued');
      onJobCreated();
    },
    onError: (error: any) => {
      toast.error(error?.response?.data?.detail ?? 'Failed to queue model training job');
    },
  });

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!corpusDir.trim()) {
      toast.error('Corpus directory is required');
      return;
    }
    if (overrideInvalid) {
      toast.error('Resolve override validation errors before submitting');
      return;
    }
    mutation.mutate();
  };

  const handleOverrideChange = useCallback((patch: Record<string, unknown>, hasErrors: boolean) => {
    setOverridePatch(patch);
    setOverrideInvalid(hasErrors);
  }, []);

  return (
    <form onSubmit={handleSubmit} className="form-card">
      <div>
        <h2 className="section-title">Train statistical model</h2>
        <p className="section-subtitle">Launch the iterative weighting loop using an enriched training corpus.</p>
      </div>
      <div className="form-grid">
        <label className="field">
          <span>Training corpus directory *</span>
          <input type="text" value={corpusDir} onChange={(event) => setCorpusDir(event.target.value)} placeholder="/data/corpus" />
        </label>
        <label className="field">
          <span>Iterations</span>
          <input
            type="number"
            min={1}
            value={iterations}
            onChange={(event) => setIterations(event.target.value ? Number(event.target.value) : '')}
          />
        </label>
        <label className="field">
          <span>Error boost factor</span>
          <input
            type="number"
            step="0.1"
            value={errorBoost}
            onChange={(event) => setErrorBoost(event.target.value ? Number(event.target.value) : '')}
          />
        </label>
      </div>
      <label className="field">
        <span>Operator notes</span>
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Optional" />
      </label>
      <OverrideEditor onChange={handleOverrideChange} />
      <button type="submit" className="primary" disabled={mutation.isPending}>
        {mutation.isPending ? 'Submitting…' : 'Launch training run'}
      </button>
    </form>
  );
}
