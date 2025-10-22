import { FormEvent, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createTrainingPairJob, JobDetail } from '../../lib/api';

interface TrainingPairFormProps {
  onCreated(job: JobDetail): void;
}

export function TrainingPairForm({ onCreated }: TrainingPairFormProps) {
  const queryClient = useQueryClient();
  const [mediaPath, setMediaPath] = useState('');
  const [srtPath, setSrtPath] = useState('');
  const [outputDirectory, setOutputDirectory] = useState('');
  const [name, setName] = useState('');
  const [pipelineOverrides, setPipelineOverrides] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const mutation = useMutation({
    mutationFn: (payload: Record<string, unknown>) => createTrainingPairJob(payload),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
      setMessage({ type: 'success', text: 'Training data job queued successfully.' });
      onCreated(job);
    },
    onError: (error) => setMessage({ type: 'error', text: (error as Error).message }),
  });

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    setMessage(null);
    if (!mediaPath.trim() || !srtPath.trim()) {
      setMessage({ type: 'error', text: 'Media and SRT paths are required.' });
      return;
    }

    try {
      const payload: Record<string, unknown> = {
        media_path: mediaPath.trim(),
        srt_path: srtPath.trim(),
        output_directory: outputDirectory.trim() || undefined,
        name: name.trim() || undefined,
        pipeline_overrides: pipelineOverrides ? JSON.parse(pipelineOverrides) : {},
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
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Episode 4 training" />
      </div>
      <div className="field">
        <label>Media file path</label>
        <input value={mediaPath} onChange={(e) => setMediaPath(e.target.value)} placeholder="/data/video.mp4" required />
      </div>
      <div className="field">
        <label>Ground-truth SRT path</label>
        <input value={srtPath} onChange={(e) => setSrtPath(e.target.value)} placeholder="/data/video.srt" required />
      </div>
      <div className="field">
        <label>Copy training JSON to directory (optional)</label>
        <input value={outputDirectory} onChange={(e) => setOutputDirectory(e.target.value)} placeholder="/exports" />
      </div>
      <div className="field">
        <label>Pipeline overrides (JSON)</label>
        <textarea
          value={pipelineOverrides}
          onChange={(e) => setPipelineOverrides(e.target.value)}
          placeholder={"{\n  \"pipeline_root\": \"/tmp/ui-training\"\n}"}
        />
      </div>
      {message && <div className={`message-banner ${message.type}`}>{message.text}</div>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={mutation.isLoading}>
          {mutation.isLoading ? 'Submitting…' : 'Generate training pair'}
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            setMediaPath('');
            setSrtPath('');
            setOutputDirectory('');
            setPipelineOverrides('');
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

export default TrainingPairForm;
