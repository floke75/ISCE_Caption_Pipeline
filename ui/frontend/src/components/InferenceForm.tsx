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

export function InferenceForm({ onJobCreated }: Props) {
  const [mediaPath, setMediaPath] = useState('');
  const [transcriptPath, setTranscriptPath] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [modelConfigPath, setModelConfigPath] = useState('');
  const [notes, setNotes] = useState('');
  const [overridePatch, setOverridePatch] = useState<Record<string, unknown>>({});
  const [overrideInvalid, setOverrideInvalid] = useState(false);
  const [mediaValid, setMediaValid] = useState(false);
  const [transcriptValid, setTranscriptValid] = useState(true);
  const [outputDirValid, setOutputDirValid] = useState(true);
  const [configPathValid, setConfigPathValid] = useState(true);

  const mutation = useMutation({
    mutationFn: async () => {
      const payload: Record<string, unknown> = {
        media_path: mediaPath.trim(),
      };
      if (transcriptPath.trim()) payload.transcript_path = transcriptPath.trim();
      if (outputDir.trim()) payload.output_dir = outputDir.trim();
      if (modelConfigPath.trim()) payload.model_config_path = modelConfigPath.trim();
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
    if (!mediaValid) {
      toast.error('Select a valid media file path before submitting');
      return;
    }
    if (!transcriptValid) {
      toast.error('Transcript path must resolve to an allowed file');
      return;
    }
    if (!outputDirValid) {
      toast.error('Output directory must be a valid allowlisted path');
      return;
    }
    if (!configPathValid) {
      toast.error('Model config path must be a valid allowlisted file');
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
        <FilePathPicker
          label="Media file path"
          value={mediaPath}
          onChange={setMediaPath}
          required
          type="file"
          helperText="Absolute media file path on the host"
          placeholder="/data/media.mp4"
          onValidityChange={setMediaValid}
        />
        <FilePathPicker
          label="Transcript (.txt)"
          value={transcriptPath}
          onChange={setTranscriptPath}
          type="file"
          placeholder="Optional"
          onValidityChange={setTranscriptValid}
        />
        <FilePathPicker
          label="Output directory"
          value={outputDir}
          onChange={setOutputDir}
          type="directory"
          placeholder="Override output folder"
          onValidityChange={setOutputDirValid}
        />
        <FilePathPicker
          label="Model config"
          value={modelConfigPath}
          onChange={setModelConfigPath}
          type="file"
          placeholder="config.yaml"
          onValidityChange={setConfigPathValid}
        />
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
