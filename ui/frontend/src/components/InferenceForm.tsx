import { FormEvent, useCallback, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import client from '../api/client';
import { OverrideEditor, type OverridePatches, type OverrideEdits } from './OverrideEditor';
import { FilePathPicker } from './FilePathPicker';
import '../styles/forms.css';

type Props = {
  onJobCreated: () => void;
};

type Preset = 'standard' | 'high_precision' | 'fast_draft';

const PRESETS: Record<Preset, string> = {
  standard: 'Standard (Balanced)',
  high_precision: 'High Precision (Refinement enabled)',
  fast_draft: 'Fast Draft (Speed optimized)',
};

export function InferenceForm({ onJobCreated }: Props) {
  const [mediaPath, setMediaPath] = useState('');
  const [transcriptPath, setTranscriptPath] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [modelConfigPath, setModelConfigPath] = useState('');
  const [notes, setNotes] = useState('');

  // Controlled Overrides State
  const [edits, setEdits] = useState<OverrideEdits>(
    () => ({ pipeline: {}, segmentation: {} })
  );

  // Helper state for submission payload (computed by OverrideEditor)
  const [overridePatch, setOverridePatch] = useState<OverridePatches>({
    pipeline: {},
    segmentation: {},
  });

  const [overrideInvalid, setOverrideInvalid] = useState(false);
  const [mediaValid, setMediaValid] = useState(false);
  const [transcriptValid, setTranscriptValid] = useState(true);
  const [outputDirValid, setOutputDirValid] = useState(true);
  const [configPathValid, setConfigPathValid] = useState(true);

  // New UI Controls State
  const [preset, setPreset] = useState<Preset>('standard');
  const [diarization, setDiarization] = useState(true);
  const [beamWidth, setBeamWidth] = useState(5);

  const mutation = useMutation({
    mutationFn: async () => {
      const payload: Record<string, unknown> = {
        media_path: mediaPath.trim(),
      };
      if (transcriptPath.trim()) payload.transcript_path = transcriptPath.trim();
      if (outputDir.trim()) payload.output_dir = outputDir.trim();
      if (modelConfigPath.trim()) payload.model_config_path = modelConfigPath.trim();
      if (notes) payload.notes = notes;

      // Use the computed patches from OverrideEditor
      if (Object.keys(overridePatch.pipeline).length) {
        payload.config_overrides = overridePatch.pipeline;
      }
      if (Object.keys(overridePatch.segmentation).length) {
        payload.segmentation_overrides = overridePatch.segmentation;
      }
      const { data } = await client.post('/jobs/inference', payload);
      return data;
    },
    onSuccess: () => {
      toast.success('Inference job queued');
      onJobCreated();
    },
    onError: (error: any) => { // eslint-disable-line @typescript-eslint/no-explicit-any
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

  const formInvalid = !mediaValid || !transcriptValid || !outputDirValid || !configPathValid || overrideInvalid;

  const handlePresetChange = (newPreset: Preset) => {
    setPreset(newPreset);
    setEdits(prev => {
        const next = { ...prev, pipeline: { ...prev.pipeline }, segmentation: { ...prev.segmentation } };

        if (newPreset === 'standard') {
             delete next.segmentation['beam_search.width'];
             setBeamWidth(5);
        } else if (newPreset === 'high_precision') {
             next.segmentation['beam_search.width'] = 10;
             setBeamWidth(10);
        } else if (newPreset === 'fast_draft') {
             next.segmentation['beam_search.width'] = 2;
             setBeamWidth(2);
        }
        return next;
    });
  };

  const handleDiarizationChange = (checked: boolean) => {
      setDiarization(checked);
      setEdits(prev => ({
          ...prev,
          pipeline: { ...prev.pipeline, 'align_make.do_diarization': checked }
      }));
  };

  const handleBeamWidthChange = (value: number) => {
      setBeamWidth(value);
      setEdits(prev => ({
          ...prev,
          segmentation: { ...prev.segmentation, 'beam_search.width': value }
      }));
  };

  const handleEditsChange = useCallback((newEdits: OverrideEdits) => {
      setEdits(newEdits);

      if ('align_make.do_diarization' in newEdits.pipeline) {
          setDiarization(Boolean(newEdits.pipeline['align_make.do_diarization']));
      } else {
          setDiarization(true);
      }

      if ('beam_search.width' in newEdits.segmentation) {
          const w = Number(newEdits.segmentation['beam_search.width']);
          if (!isNaN(w)) setBeamWidth(w);
      } else {
          setBeamWidth(5);
      }
  }, []);

  const handleOverrideChange = useCallback((patches: OverridePatches, hasErrors: boolean) => {
    setOverridePatch(patches);
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
          helperText="Upload a corrected script to align heavily edited text"
          onValidityChange={setTranscriptValid}
        />
        <FilePathPicker
          label="Output directory"
          value={outputDir}
          onChange={setOutputDir}
          type="directory"
          placeholder="Override output folder"
          helperText="Directory will be created if it does not exist"
          onValidityChange={setOutputDirValid}
        />
        <FilePathPicker
          label="Model config"
          value={modelConfigPath}
          onChange={setModelConfigPath}
          type="file"
          placeholder="config.yaml"
          helperText="Advanced: Load a full alternative configuration file"
          onValidityChange={setConfigPathValid}
        />
      </div>

      <div className="form-card" style={{marginTop: '1rem', background: 'rgba(15, 23, 42, 0.3)'}}>
         <h3 className="section-title" style={{fontSize: '1rem'}}>Quality Settings</h3>
         <div className="form-grid">
            <label className="field">
               <span>Preset</span>
               <select value={preset} onChange={(e) => handlePresetChange(e.target.value as Preset)}>
                  {Object.entries(PRESETS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
               </select>
            </label>

            <label className="field">
               <span>Beam Width (Search Depth): {beamWidth}</span>
               <input
                   type="range" min="1" max="20" step="1"
                   value={beamWidth}
                   onChange={(e) => handleBeamWidthChange(Number(e.target.value))}
               />
            </label>

            <div style={{display: 'flex', alignItems: 'flex-end', paddingBottom: '0.5rem'}}>
              <label className="toggle">
                <input
                    type="checkbox"
                    checked={diarization}
                    onChange={(e) => handleDiarizationChange(e.target.checked)}
                />
                <span>Enable Speaker Diarization</span>
              </label>
            </div>
         </div>
      </div>

      <label className="field">
        <span>Operator notes</span>
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Optional instructions or labels for this run" />
      </label>

      <OverrideEditor
         edits={edits}
         onEditsChange={handleEditsChange}
         onChange={handleOverrideChange}
      />

      <button
        type="submit"
        className="primary"
        disabled={mutation.isPending || formInvalid}
        title={formInvalid ? 'Please resolve validation errors' : 'Launch inference run'}
      >
        {mutation.isPending ? 'Submitting…' : 'Launch inference run'}
      </button>
    </form>
  );
}
