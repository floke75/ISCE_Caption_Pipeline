import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  fetchModelConfig,
  fetchModelYaml,
  fetchPipelineConfig,
  fetchPipelineYaml,
  saveModelConfig,
  savePipelineConfig,
} from '../lib/api';
import { ConfigMap, ConfigValue, isObject, updateAtPath } from '../lib/configTree';

interface ConfigEditorProps {
  onSaved?: () => void;
}

const clone = (value: Record<string, unknown>): ConfigMap => JSON.parse(JSON.stringify(value));

export function ConfigEditor({ onSaved }: ConfigEditorProps) {
  const queryClient = useQueryClient();
  const pipelineQuery = useQuery({ queryKey: ['config', 'pipeline'], queryFn: fetchPipelineConfig });
  const modelQuery = useQuery({ queryKey: ['config', 'model'], queryFn: fetchModelConfig });

  const [pipelineDraft, setPipelineDraft] = useState<ConfigMap | null>(null);
  const [modelDraft, setModelDraft] = useState<ConfigMap | null>(null);
  const [pipelineMessage, setPipelineMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [modelMessage, setModelMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    if (pipelineQuery.data) {
      setPipelineDraft(clone(pipelineQuery.data));
    }
  }, [pipelineQuery.data]);

  useEffect(() => {
    if (modelQuery.data) {
      setModelDraft(clone(modelQuery.data));
    }
  }, [modelQuery.data]);

  const savePipelineMutation = useMutation({
    mutationFn: (payload: ConfigMap) => savePipelineConfig(payload),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['config', 'pipeline'] });
      setPipelineDraft(clone(data));
      setPipelineMessage({ type: 'success', text: 'Pipeline configuration saved.' });
      onSaved?.();
    },
    onError: (err) => setPipelineMessage({ type: 'error', text: (err as Error).message }),
  });

  const saveModelMutation = useMutation({
    mutationFn: (payload: ConfigMap) => saveModelConfig(payload),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['config', 'model'] });
      setModelDraft(clone(data));
      setModelMessage({ type: 'success', text: 'Model configuration saved.' });
      onSaved?.();
    },
    onError: (err) => setModelMessage({ type: 'error', text: (err as Error).message }),
  });

  const handlePipelineChange = (path: string[], value: ConfigValue) => {
    setPipelineDraft((prev) => (prev ? (updateAtPath(prev, path, value) as ConfigMap) : prev));
  };

  const handleModelChange = (path: string[], value: ConfigValue) => {
    setModelDraft((prev) => (prev ? (updateAtPath(prev, path, value) as ConfigMap) : prev));
  };

  const handleDownload = async (type: 'pipeline' | 'model') => {
    try {
      const yaml = type === 'pipeline' ? await fetchPipelineYaml() : await fetchModelYaml();
      const filename = type === 'pipeline' ? 'pipeline_config.yaml' : 'config.yaml';
      const blob = new Blob([yaml], { type: 'text/yaml' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const msg = { type: 'error', text: 'Download failed: ' + (err as Error).message };
      if (type === 'pipeline') {
        setPipelineMessage(msg);
      } else {
        setModelMessage(msg);
      }
    }
  };

  return (
    <div className="panel-body">
      <div className="config-editor">
        <div className="config-section">
          <h3>Pipeline configuration</h3>
          {pipelineDraft ? (
            <ConfigTree value={pipelineDraft} onChange={handlePipelineChange} />
          ) : (
            <div style={{ color: '#64748b' }}>Loading pipeline config…</div>
          )}
          {pipelineMessage && (
            <div className={`message-banner ${pipelineMessage.type}`}>{pipelineMessage.text}</div>
          )}
          <div className="config-actions">
            <button
              className="primary-button"
              type="button"
              onClick={() => pipelineDraft && savePipelineMutation.mutate(pipelineDraft)}
              disabled={savePipelineMutation.isLoading || !pipelineDraft}
            >
              {savePipelineMutation.isLoading ? 'Saving…' : 'Save pipeline config'}
            </button>
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                setPipelineDraft(pipelineQuery.data ? clone(pipelineQuery.data) : pipelineDraft);
                setPipelineMessage(null);
              }}
            >
              Reset
            </button>
            <button className="ghost-button" type="button" onClick={() => handleDownload('pipeline')}>
              Download YAML
            </button>
          </div>
        </div>

        <div className="config-section">
          <h3>Segmentation config</h3>
          {modelDraft ? (
            <ConfigTree value={modelDraft} onChange={handleModelChange} />
          ) : (
            <div style={{ color: '#64748b' }}>Loading segmentation config…</div>
          )}
          {modelMessage && (
            <div className={`message-banner ${modelMessage.type}`}>{modelMessage.text}</div>
          )}
          <div className="config-actions">
            <button
              className="primary-button"
              type="button"
              onClick={() => modelDraft && saveModelMutation.mutate(modelDraft)}
              disabled={saveModelMutation.isLoading || !modelDraft}
            >
              {saveModelMutation.isLoading ? 'Saving…' : 'Save segmentation config'}
            </button>
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                setModelDraft(modelQuery.data ? clone(modelQuery.data) : modelDraft);
                setModelMessage(null);
              }}
            >
              Reset
            </button>
            <button className="ghost-button" type="button" onClick={() => handleDownload('model')}>
              Download YAML
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ConfigTreeProps {
  value: ConfigMap;
  onChange(path: string[], value: ConfigValue): void;
}

const ConfigTree = ({ value, onChange }: ConfigTreeProps) => {
  const entries = useMemo(() => Object.entries(value), [value]);
  return (
    <div className="config-tree">
      {entries.map(([key, child]) => (
        <ConfigNode key={key} path={[key]} label={key} value={child} onChange={onChange} />
      ))}
    </div>
  );
};

interface ConfigNodeProps {
  path: string[];
  label: string;
  value: ConfigValue;
  onChange(path: string[], value: ConfigValue): void;
}

const ConfigNode = ({ path, label, value, onChange }: ConfigNodeProps) => {
  if (isObject(value)) {
    return (
      <details className="config-node" open>
        <summary>{label}</summary>
        <div className="config-children">
          {Object.entries(value).map(([key, child]) => (
            <ConfigNode key={key} path={[...path, key]} label={key} value={child} onChange={onChange} />
          ))}
        </div>
      </details>
    );
  }

  if (Array.isArray(value)) {
    return <ArrayField label={label} value={value} onChange={(next) => onChange(path, next)} />;
  }

  if (typeof value === 'boolean') {
    return (
      <div className="field">
        <label>{label}</label>
        <div>
          <input
            type="checkbox"
            checked={value}
            onChange={(e) => onChange(path, e.target.checked)}
          />
        </div>
      </div>
    );
  }

  if (typeof value === 'number') {
    return (
      <div className="field">
        <label>{label}</label>
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(path, Number(e.target.value))}
        />
      </div>
    );
  }

  return (
    <div className="field">
      <label>{label}</label>
      <input
        value={value === null ? '' : String(value)}
        onChange={(e) => onChange(path, e.target.value)}
        placeholder={value === null ? 'null' : undefined}
      />
    </div>
  );
};

interface ArrayFieldProps {
  label: string;
  value: ConfigValue[];
  onChange(value: ConfigValue[]): void;
}

const ArrayField = ({ label, value, onChange }: ArrayFieldProps) => {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setText(JSON.stringify(value, null, 2));
  }, [value]);

  const handleBlur = () => {
    try {
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) {
        setError('Value must be a JSON array.');
        return;
      }
      onChange(parsed as ConfigValue[]);
      setError(null);
    } catch (err) {
      setError('Invalid JSON.');
    }
  };

  return (
    <div className="field">
      <label>{label}</label>
      <textarea
        className="array-editor"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={handleBlur}
      />
      {error && <small style={{ color: '#b91c1c' }}>{error}</small>}
    </div>
  );
};

export default ConfigEditor;
