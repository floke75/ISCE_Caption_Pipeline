import { FormEvent, useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import client from '../api/client';
import { OverrideEditor } from './OverrideEditor';
import '../styles/forms.css';

type Props = {
  onJobCreated: () => void;
};

export function InferenceForm({ onJobCreated }: Props) {
  const [mediaPath, setMediaPath] = useState('');
  const [transcriptPath, setTranscriptPath] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [modelConfigPath, setModelConfigPath] = useState('');
  const [notes, setNotes] = useState('');
  const [overridePatch, setOverridePatch] = useState<Record<string, unknown>>({});
  const [overrideInvalid, setOverrideInvalid] = useState(false);

  const mutation = useMutation({
    mutationFn: async () => {
      const payload: Record<string, unknown> = {
        media_path: mediaPath,
      };
      if (transcriptPath) payload.transcript_path = transcriptPath;
      if (outputDir) payload.output_dir = outputDir;
      if (modelConfigPath) payload.model_config_path = modelConfigPath;
      if (notes) payload.notes = notes;
      if (Object.keys(overridePatch).length) {
        payload.config_overrides = overridePatch;
      }
      const { data } = await client.post('/jobs/inference', payload);
      return data;
    },
    onSuccess: () => {
      toast.success('Inference job queued');
      onJobCreated();
    },
    onError: (error: any) => {
      toast.error(error?.response?.data?.detail ?? 'Failed to queue inference job');
    },
  });

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!mediaPath.trim()) {
      toast.error('Media file is required');
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
        <h2 className="section-title">Run inference</h2>
        <p className="section-subtitle">Provide a media file and optional transcript to generate an SRT subtitle file.</p>
      </div>
      <div className="form-grid">
        <label className="field">
          <span>Media file path *</span>
          <input type="text" value={mediaPath} onChange={(event) => setMediaPath(event.target.value)} placeholder="/data/media.mp4" />
          <span className="field-help">Absolute path on the host machine</span>
        </label>
        <label className="field">
          <span>Transcript (.txt)</span>
          <input type="text" value={transcriptPath} onChange={(event) => setTranscriptPath(event.target.value)} placeholder="Optional" />
        </label>
        <label className="field">
          <span>Output directory</span>
          <input type="text" value={outputDir} onChange={(event) => setOutputDir(event.target.value)} placeholder="Override output folder" />
        </label>
        <label className="field">
          <span>Model config</span>
          <input type="text" value={modelConfigPath} onChange={(event) => setModelConfigPath(event.target.value)} placeholder="config.yaml" />
        </label>
      </div>
      <label className="field">
        <span>Operator notes</span>
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Optional instructions or labels for this run" />
      </label>
      <OverrideEditor onChange={handleOverrideChange} />
      <button type="submit" className="primary" disabled={mutation.isPending}>
        {mutation.isPending ? 'Submitting…' : 'Launch inference run'}
      </button>
    </form>
  );
}
