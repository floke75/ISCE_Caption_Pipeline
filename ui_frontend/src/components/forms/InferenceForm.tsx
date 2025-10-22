import { FormEvent, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createInferenceJob, JobDetail } from '../../lib/api';

interface InferenceFormProps {
  onCreated(job: JobDetail): void;
}

export function InferenceForm({ onCreated }: InferenceFormProps) {
  const queryClient = useQueryClient();
  const [mediaPath, setMediaPath] = useState('');
  const [transcriptPath, setTranscriptPath] = useState('');
  const [outputDirectory, setOutputDirectory] = useState('');
  const [name, setName] = useState('');
  const [pipelineOverrides, setPipelineOverrides] = useState('');
  const [modelOverrides, setModelOverrides] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const mutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => createInferenceJob(payload),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      setMessage({ type: 'success', text: 'Inference job queued successfully.' });
      onCreated(job);
    },
    onError: (error) => {
      setMessage({ type: 'error', text: (error as Error).message });
    },
  });

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    setMessage(null);
    if (!mediaPath.trim()) {
      setMessage({ type: 'error', text: 'Media path is required.' });
      return;
    }

    try {
      const payload: Record<string, unknown> = {
        media_path: mediaPath.trim(),
        transcript_path: transcriptPath.trim() || undefined,
        output_directory: outputDirectory.trim() || undefined,
        name: name.trim() || undefined,
        pipeline_overrides: pipelineOverrides ? JSON.parse(pipelineOverrides) : {},
        model_overrides: modelOverrides ? JSON.parse(modelOverrides) : {},
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
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Client delivery 04" />
      </div>
      <div className="field">
        <label>Media file path</label>
        <input value={mediaPath} onChange={(e) => setMediaPath(e.target.value)} placeholder="/data/video.mp4" required />
      </div>
      <div className="field">
        <label>Transcript path (optional)</label>
        <input value={transcriptPath} onChange={(e) => setTranscriptPath(e.target.value)} placeholder="/data/transcript.txt" />
      </div>
      <div className="field">
        <label>Copy SRT to directory (optional)</label>
        <input value={outputDirectory} onChange={(e) => setOutputDirectory(e.target.value)} placeholder="/exports" />
      </div>
      <div className="field">
        <label>Pipeline overrides (JSON)</label>
        <textarea
          value={pipelineOverrides}
          onChange={(e) => setPipelineOverrides(e.target.value)}
          placeholder="{\n  \"pipeline_root\": \"/tmp/ui\"\n}"
        />
      </div>
      <div className="field">
        <label>Segmentation config overrides (JSON)</label>
        <textarea
          value={modelOverrides}
          onChange={(e) => setModelOverrides(e.target.value)}
          placeholder="{\n  \"sliders\": { \"flow\": 1.2 }\n}"
        />
      </div>
      {message && <div className={`message-banner ${message.type}`}>{message.text}</div>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={mutation.isLoading}>
          {mutation.isLoading ? 'Submitting…' : 'Launch inference'}
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            setMediaPath('');
            setTranscriptPath('');
            setOutputDirectory('');
            setName('');
            setPipelineOverrides('');
            setModelOverrides('');
            setMessage(null);
          }}
        >
          Reset
        </button>
      </div>
    </form>
  );
}

export default InferenceForm;
