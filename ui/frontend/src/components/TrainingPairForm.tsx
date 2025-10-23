import { FormEvent, useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import client from '../api/client';
import { OverrideEditor } from './OverrideEditor';
import { FilePathPicker } from './FilePathPicker';
import '../styles/forms.css';

type Props = {
  onJobCreated: () => void;
};

/**
 * A form for submitting new training pair generation jobs.
 *
 * This component provides input fields for the media file and ground-truth
 * SRT file required to generate a labeled and enriched training sample.
 *
 * @param {Props} props The props for the component.
 * @returns {JSX.Element} The rendered training pair form.
 */
export function TrainingPairForm({ onJobCreated }: Props) {
  const [mediaPath, setMediaPath] = useState('');
  const [srtPath, setSrtPath] = useState('');
  const [notes, setNotes] = useState('');
  const [overridePatch, setOverridePatch] = useState<Record<string, unknown>>({});
  const [overrideInvalid, setOverrideInvalid] = useState(false);
  const [mediaValid, setMediaValid] = useState(false);
  const [srtValid, setSrtValid] = useState(false);

  const mutation = useMutation({
    mutationFn: async () => {
      const payload: Record<string, unknown> = {
        media_path: mediaPath.trim(),
        srt_path: srtPath.trim(),
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
    if (!mediaValid || !srtValid) {
      toast.error('Provide valid media and SRT paths');
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
        <FilePathPicker
          label="Media file path"
          value={mediaPath}
          onChange={setMediaPath}
          required
          type="file"
          placeholder="/data/media.mp4"
          onValidityChange={setMediaValid}
        />
        <FilePathPicker
          label="SRT file path"
          value={srtPath}
          onChange={setSrtPath}
          required
          type="file"
          placeholder="/data/captions.srt"
          onValidityChange={setSrtValid}
        />
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
