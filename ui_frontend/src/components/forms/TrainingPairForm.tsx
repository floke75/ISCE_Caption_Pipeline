import { FormEvent, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createTrainingPairJob, JobDetail } from '../../lib/api';
import OverrideEditor from '../OverrideEditor';
import {
  PathFieldValidation,
  ValidatedPathField,
} from '../inputs/ValidatedPathField';

interface TrainingPairFormProps {
  onCreated(job: JobDetail): void;
}

export function TrainingPairForm({ onCreated }: TrainingPairFormProps) {
  const queryClient = useQueryClient();
  const [mediaPath, setMediaPath] = useState('');
  const [srtPath, setSrtPath] = useState('');
  const [outputDirectory, setOutputDirectory] = useState('');
  const [name, setName] = useState('');
  const [pipelineOverrides, setPipelineOverrides] = useState<Record<string, unknown>>({});
  const [pipelineOverridesValid, setPipelineOverridesValid] = useState(true);
  const [editorKey, setEditorKey] = useState(0);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [mediaValidation, setMediaValidation] = useState<PathFieldValidation>({
    valid: false,
    resolvedPath: null,
    message: 'Media path is required.',
    checking: false,
  });
  const [srtValidation, setSrtValidation] = useState<PathFieldValidation>({
    valid: false,
    resolvedPath: null,
    message: 'SRT path is required.',
    checking: false,
  });
  const [outputValidation, setOutputValidation] = useState<PathFieldValidation>({
    valid: true,
    resolvedPath: null,
    message: null,
    checking: false,
  });

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
    if (mediaValidation.checking || srtValidation.checking || outputValidation.checking) {
      setMessage({ type: 'error', text: 'Please wait for path validation to finish.' });
      return;
    }
    if (!mediaValidation.valid) {
      setMessage({
        type: 'error',
        text: mediaValidation.message ?? 'Media path is invalid.',
      });
      return;
    }
    if (!srtValidation.valid) {
      setMessage({
        type: 'error',
        text: srtValidation.message ?? 'SRT path is invalid.',
      });
      return;
    }
    if (!outputValidation.valid) {
      setMessage({
        type: 'error',
        text: outputValidation.message ?? 'Output directory is invalid.',
      });
      return;
    }
    if (!pipelineOverridesValid) {
      setMessage({ type: 'error', text: 'Resolve override errors before submitting.' });
      return;
    }

    const normaliseOptional = (resolved: string | null, raw: string) => {
      const trimmed = raw.trim();
      return resolved ?? (trimmed ? trimmed : undefined);
    };

    const payload: Record<string, unknown> = {
      media_path: mediaValidation.resolvedPath ?? mediaPath.trim(),
      srt_path: srtValidation.resolvedPath ?? srtPath.trim(),
      output_directory: normaliseOptional(outputValidation.resolvedPath, outputDirectory),
      name: name.trim() || undefined,
      pipeline_overrides: pipelineOverrides,
    };
    mutation.mutate(payload);
  };

  const submittingDisabled =
    mutation.isLoading ||
    mediaValidation.checking ||
    srtValidation.checking ||
    outputValidation.checking ||
    !mediaValidation.valid ||
    !srtValidation.valid ||
    !outputValidation.valid ||
    !pipelineOverridesValid;

  return (
    <form className="panel-body form-grid" onSubmit={handleSubmit}>
      <div className="field">
        <label>Friendly name (optional)</label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Episode 4 training" />
      </div>
      <ValidatedPathField
        label="Media file path"
        value={mediaPath}
        onChange={setMediaPath}
        kind="file"
        required
        placeholder="/data/video.mp4"
        description="Media file"
        onValidation={setMediaValidation}
      />
      <ValidatedPathField
        label="Ground-truth SRT path"
        value={srtPath}
        onChange={setSrtPath}
        kind="file"
        required
        placeholder="/data/video.srt"
        description="Ground-truth SRT file"
        onValidation={setSrtValidation}
      />
      <ValidatedPathField
        label="Copy training JSON to directory (optional)"
        value={outputDirectory}
        onChange={setOutputDirectory}
        kind="directory"
        required={false}
        mustExist={false}
        allowCreate
        placeholder="/exports"
        description="Output directory"
        onValidation={setOutputValidation}
      />
      <OverrideEditor
        key={editorKey}
        configType="pipeline"
        label="Pipeline overrides"
        onChange={setPipelineOverrides}
        onValidityChange={setPipelineOverridesValid}
      />
      {message && <div className={`message-banner ${message.type}`}>{message.text}</div>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={submittingDisabled}>
          {mutation.isLoading ? 'Submitting…' : 'Generate training pair'}
        </button>
        <button
          type="button"
          className="ghost-button"
          onClick={() => {
            setMediaPath('');
            setSrtPath('');
            setOutputDirectory('');
            setPipelineOverrides({});
            setPipelineOverridesValid(true);
            setEditorKey((value) => value + 1);
            setMediaValidation({ valid: false, resolvedPath: null, message: 'Media path is required.', checking: false });
            setSrtValidation({ valid: false, resolvedPath: null, message: 'SRT path is required.', checking: false });
            setOutputValidation({ valid: true, resolvedPath: null, message: null, checking: false });
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
