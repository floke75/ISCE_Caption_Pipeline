import { FormEvent, useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import client from '../api/client';
import { OverrideEditor } from './OverrideEditor';
import '../styles/forms.css';

type Props = {
  onJobCreated: () => void;
};

export function TrainingPairForm({ onJobCreated }: Props) {
  const [mediaPath, setMediaPath] = useState('');
  const [srtPath, setSrtPath] = useState('');
  const [notes, setNotes] = useState('');
  const [overridePatch, setOverridePatch] = useState<Record<string, unknown>>({});
  const [overrideInvalid, setOverrideInvalid] = useState(false);

  const mutation = useMutation({
    mutationFn: async () => {
      const payload: Record<string, unknown> = {
        media_path: mediaPath,
        srt_path: srtPath,
      };
      if (notes) payload.notes = notes;
      if (Object.keys(overridePatch).length) {
        payload.config_overrides = overridePatch;
      }
      const { data } = await client.post('/jobs/training-pair', payload);
      return data;
    },
    onSuccess: () => {
      toast.success('Training-pair job queued');
      onJobCreated();
    },
    onError: (error: any) => {
      toast.error(error?.response?.data?.detail ?? 'Failed to queue training data job');
    },
  });

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!mediaPath.trim() || !srtPath.trim()) {
      toast.error('Media and SRT paths are required');
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
        <h2 className="section-title">Build training pair</h2>
        <p className="section-subtitle">Generate enriched training JSON from an SRT file and matching media.</p>
      </div>
      <div className="form-grid">
        <label className="field">
          <span>Media file path *</span>
          <input type="text" value={mediaPath} onChange={(event) => setMediaPath(event.target.value)} placeholder="/data/media.mp4" />
        </label>
        <label className="field">
          <span>SRT file path *</span>
          <input type="text" value={srtPath} onChange={(event) => setSrtPath(event.target.value)} placeholder="/data/captions.srt" />
        </label>
      </div>
      <label className="field">
        <span>Operator notes</span>
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Context for this corpus artifact" />
      </label>
      <OverrideEditor onChange={handleOverrideChange} />
      <button type="submit" className="primary" disabled={mutation.isPending}>
        {mutation.isPending ? 'Submitting…' : 'Launch training-pair job'}
      </button>
    </form>
  );
}
