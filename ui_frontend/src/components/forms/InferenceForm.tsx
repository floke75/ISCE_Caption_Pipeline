import { FormEvent, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createInferenceJob, JobDetail } from '../../lib/api';
import OverrideEditor from '../OverrideEditor';
import {
  PathFieldValidation,
  ValidatedPathField,
} from '../inputs/ValidatedPathField';

interface InferenceFormProps {
  onCreated(job: JobDetail): void;
}

export function InferenceForm({ onCreated }: InferenceFormProps) {
  const queryClient = useQueryClient();
  const [mediaPath, setMediaPath] = useState('');
  const [transcriptPath, setTranscriptPath] = useState('');
  const [outputDirectory, setOutputDirectory] = useState('');
  const [name, setName] = useState('');
  const [pipelineOverrides, setPipelineOverrides] = useState<Record<string, unknown>>({});
  const [pipelineOverridesValid, setPipelineOverridesValid] = useState(true);
  const [pipelineEditorKey, setPipelineEditorKey] = useState(0);
  const [modelOverrides, setModelOverrides] = useState<Record<string, unknown>>({});
  const [modelOverridesValid, setModelOverridesValid] = useState(true);
  const [modelEditorKey, setModelEditorKey] = useState(0);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [mediaValidation, setMediaValidation] = useState<PathFieldValidation>({
    valid: false,
    resolvedPath: null,
    message: 'Media path is required.',
    checking: false,
  });
  const [transcriptValidation, setTranscriptValidation] = useState<PathFieldValidation>({
    valid: true,
    resolvedPath: null,
    message: null,
    checking: false,
  });
  const [outputValidation, setOutputValidation] = useState<PathFieldValidation>({
    valid: true,
    resolvedPath: null,
    message: null,
    checking: false,
  });

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
    if (mediaValidation.checking || transcriptValidation.checking || outputValidation.checking) {
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
    if (!transcriptValidation.valid) {
      setMessage({
        type: 'error',
        text: transcriptValidation.message ?? 'Transcript path is invalid.',
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
    if (!pipelineOverridesValid || !modelOverridesValid) {
      setMessage({ type: 'error', text: 'Resolve override errors before submitting.' });
      return;
    }

    const normaliseOptional = (resolved: string | null, raw: string) => {
      const trimmed = raw.trim();
      return resolved ?? (trimmed ? trimmed : undefined);
    };

    const payload: Record<string, unknown> = {
      media_path: mediaValidation.resolvedPath ?? mediaPath.trim(),
      transcript_path: normaliseOptional(transcriptValidation.resolvedPath, transcriptPath),
      output_directory: normaliseOptional(outputValidation.resolvedPath, outputDirectory),
      name: name.trim() || undefined,
      pipeline_overrides: pipelineOverrides,
      model_overrides: modelOverrides,
    };
    mutation.mutate(payload);
  };

  const submittingDisabled =
    mutation.isLoading ||
    mediaValidation.checking ||
    transcriptValidation.checking ||
    outputValidation.checking ||
    !mediaValidation.valid ||
    !transcriptValidation.valid ||
    !outputValidation.valid ||
    !pipelineOverridesValid ||
    !modelOverridesValid;

  return (
    <form className="panel-body form-grid" onSubmit={handleSubmit}>
      <div className="field">
        <label>Friendly name (optional)</label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Client delivery 04" />
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
        label="Transcript path (optional)"
        value={transcriptPath}
        onChange={setTranscriptPath}
        kind="file"
        required={false}
        placeholder="/data/transcript.txt"
        description="Transcript file"
        onValidation={setTranscriptValidation}
      />
      <ValidatedPathField
        label="Copy SRT to directory (optional)"
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
        key={pipelineEditorKey}
        configType="pipeline"
        label="Pipeline overrides"
        onChange={setPipelineOverrides}
        onValidityChange={setPipelineOverridesValid}
      />
      <OverrideEditor
        key={modelEditorKey}
        configType="model"
        label="Segmentation config overrides"
        onChange={setModelOverrides}
        onValidityChange={setModelOverridesValid}
      />
      {message && <div className={`message-banner ${message.type}`}>{message.text}</div>}
      <div className="form-actions">
        <button className="primary-button" type="submit" disabled={submittingDisabled}>
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
            setPipelineOverrides({});
            setModelOverrides({});
            setPipelineOverridesValid(true);
            setModelOverridesValid(true);
            setPipelineEditorKey((value) => value + 1);
            setModelEditorKey((value) => value + 1);
            setMediaValidation({ valid: false, resolvedPath: null, message: 'Media path is required.', checking: false });
            setTranscriptValidation({ valid: true, resolvedPath: null, message: null, checking: false });
            setOutputValidation({ valid: true, resolvedPath: null, message: null, checking: false });
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
