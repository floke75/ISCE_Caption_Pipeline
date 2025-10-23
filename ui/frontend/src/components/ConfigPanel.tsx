import { useEffect, useMemo, useState } from 'react';
import YAML from 'yaml';
import {
  usePipelineConfig,
  useReplaceConfig,
  useSegmentationConfig,
  useUpdateConfig,
  useUpdateConfigYaml,
} from '../hooks/useConfig';
import { ConfigField, ConfigSnapshot } from '../types';
import '../styles/forms.css';

function getValueFromPath(source: Record<string, unknown>, path: string[]): unknown {
  return path.reduce<unknown>((acc, key) => {
    if (acc && typeof acc === 'object' && key in (acc as Record<string, unknown>)) {
      return (acc as Record<string, unknown>)[key];
    }
    return undefined;
  }, source);
}

function areEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

interface ConfigEditorProps {
  label: string;
  summary: string;
  snapshot?: ConfigSnapshot;
  isLoading: boolean;
  updateMutation: ReturnType<typeof useUpdateConfig>;
  replaceMutation: ReturnType<typeof useReplaceConfig>;
  yamlMutation: ReturnType<typeof useUpdateConfigYaml>;
}

function ConfigEditor({
  label,
  summary,
  snapshot,
  isLoading,
  updateMutation,
  replaceMutation,
  yamlMutation,
}: ConfigEditorProps) {
  const [formValues, setFormValues] = useState<Record<string, unknown>>({});
  const [initialValues, setInitialValues] = useState<Record<string, unknown>>({});
  const [yamlOverrides, setYamlOverrides] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    if (!snapshot) return;
    const values: Record<string, unknown> = {};
    snapshot.fields.forEach((field) => {
      const dotted = field.path.join('.');
      values[dotted] = getValueFromPath(snapshot.effective as Record<string, unknown>, field.path);
    });
    setFormValues(values);
    setInitialValues(values);
    setYamlOverrides(
      snapshot.overrides && Object.keys(snapshot.overrides).length
        ? YAML.stringify(snapshot.overrides)
        : '# no overrides'
    );
  }, [snapshot]);

  const groupedFields = useMemo(() => {
    if (!snapshot) return new Map<string, ConfigField[]>();
    const map = new Map<string, ConfigField[]>();
    snapshot.fields.forEach((field) => {
      const arr = map.get(field.section) ?? [];
      arr.push(field);
      map.set(field.section, arr);
    });
    return map;
  }, [snapshot]);

  const handleFieldChange = (field: ConfigField, raw: unknown) => {
    const dotted = field.path.join('.');
    let value: unknown = raw;
    if (field.fieldType === 'number' && typeof raw === 'string') {
      value = raw === '' ? '' : Number(raw);
    }
    if (field.fieldType === 'list' && typeof raw === 'string') {
      value = raw
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean);
    }
    if (field.fieldType === 'boolean' && typeof raw === 'boolean') {
      value = raw;
    }
    setFormValues((prev) => ({ ...prev, [dotted]: value }));
  };

  const handleStructuredSave = () => {
    const updates: Record<string, unknown> = {};
    Object.entries(formValues).forEach(([key, value]) => {
      const initial = initialValues[key];
      if (!areEqual(value, initial)) {
        updates[key] = value;
      }
    });
    if (Object.keys(updates).length === 0) {
      return;
    }
    updateMutation.mutate(updates, {
      onSuccess: () => {
        setInitialValues(formValues);
      },
    });
  };

  const handleResetOverrides = () => {
    replaceMutation.mutate({}, {
      onSuccess: () => {
        setYamlOverrides('# no overrides');
      },
    });
  };

  const handleYamlSave = () => {
    yamlMutation.mutate(yamlOverrides);
  };

  if (isLoading || !snapshot) {
    return <div className="form-card">Loading configuration…</div>;
  }

  return (
    <div className="config-editor" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 className="section-title">{label}</h2>
        <p className="section-subtitle">{summary}</p>
      </div>
      <div className="form-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 className="section-title">Structured editor</h2>
            <p className="section-subtitle">Edit the most common knobs with validation and type hints.</p>
          </div>
          <button type="button" className="ghost" onClick={() => setShowAdvanced((prev) => !prev)}>
            {showAdvanced ? 'Hide advanced' : 'Show advanced'}
          </button>
        </div>
        {[...groupedFields.entries()].map(([section, fields]) => {
          const visible = fields.filter((field) => showAdvanced || !field.advanced);
          if (!visible.length) return null;
          return (
            <div key={section} style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <h3 className="section-title">{section}</h3>
              <div className="form-grid">
                {visible.map((field) => {
                  const dotted = field.path.join('.');
                  const value = formValues[dotted];
                  return (
                    <label key={dotted} className="field">
                      <span>{field.label}</span>
                      {field.fieldType === 'boolean' ? (
                        <input
                          type="checkbox"
                          checked={Boolean(value)}
                          onChange={(event) => handleFieldChange(field, event.target.checked)}
                        />
                      ) : field.fieldType === 'number' ? (
                        <input
                          type="number"
                          value={
                            typeof value === 'number'
                              ? value
                              : typeof value === 'string'
                              ? value
                              : ''
                          }
                          onChange={(event) => handleFieldChange(field, event.target.value)}
                        />
                      ) : field.fieldType === 'list' ? (
                        <input
                          type="text"
                          value={Array.isArray(value) ? value.join(', ') : ''}
                          onChange={(event) => handleFieldChange(field, event.target.value)}
                        />
                      ) : field.fieldType === 'select' && field.options ? (
                        <select value={(value as string) ?? ''} onChange={(event) => handleFieldChange(field, event.target.value)}>
                          {field.options.map((option) => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="text"
                          value={(value as string) ?? ''}
                          onChange={(event) => handleFieldChange(field, event.target.value)}
                        />
                      )}
                      {field.description ? <span className="field-help">{field.description}</span> : null}
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}
        <button type="button" className="primary" onClick={handleStructuredSave} disabled={updateMutation.isPending}>
          {updateMutation.isPending ? 'Saving…' : 'Save changes'}
        </button>
      </div>

      <div className="form-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h2 className="section-title">Raw overrides</h2>
            <p className="section-subtitle">Edit or paste YAML overrides for full control.</p>
          </div>
          <button type="button" className="ghost" onClick={handleResetOverrides} disabled={replaceMutation.isPending}>
            Reset overrides
          </button>
        </div>
        <textarea value={yamlOverrides} onChange={(event) => setYamlOverrides(event.target.value)} />
        <button type="button" className="primary" onClick={handleYamlSave} disabled={yamlMutation.isPending}>
          {yamlMutation.isPending ? 'Saving…' : 'Save YAML'}
        </button>
      </div>
    </div>
  );
}

type ConfigTab = 'pipeline' | 'segmentation';

type ConfigTabState = {
  label: string;
  summary: string;
  query: ReturnType<typeof usePipelineConfig>;
  update: ReturnType<typeof useUpdateConfig>;
  replace: ReturnType<typeof useReplaceConfig>;
  yaml: ReturnType<typeof useUpdateConfigYaml>;
};

export function ConfigPanel() {
  const [activeTab, setActiveTab] = useState<ConfigTab>('pipeline');

  const pipelineQuery = usePipelineConfig();
  const segmentationQuery = useSegmentationConfig();

  const pipelineUpdate = useUpdateConfig('pipeline');
  const segmentationUpdate = useUpdateConfig('segmentation');

  const pipelineReplace = useReplaceConfig('pipeline');
  const segmentationReplace = useReplaceConfig('segmentation');

  const pipelineYaml = useUpdateConfigYaml('pipeline');
  const segmentationYaml = useUpdateConfigYaml('segmentation');

  const tabs: Record<ConfigTab, ConfigTabState> = {
    pipeline: {
      label: 'Pipeline configuration',
      summary: 'Paths, stage toggles, and orchestration defaults for the ingestion pipeline.',
      query: pipelineQuery,
      update: pipelineUpdate,
      replace: pipelineReplace,
      yaml: pipelineYaml,
    },
    segmentation: {
      label: 'Segmentation model configuration',
      summary: 'Beam search knobs, stylistic sliders, and model asset paths for segmentation.',
      query: segmentationQuery,
      update: segmentationUpdate,
      replace: segmentationReplace,
      yaml: segmentationYaml,
    },
  };

  const active = tabs[activeTab];

  return (
    <div className="config-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="config-tabs">
        {(Object.keys(tabs) as ConfigTab[]).map((tabKey) => (
          <button
            key={tabKey}
            type="button"
            className={tabKey === activeTab ? 'config-tab active' : 'config-tab'}
            onClick={() => setActiveTab(tabKey)}
          >
            {tabs[tabKey].label}
          </button>
        ))}
      </div>
      <ConfigEditor
        key={activeTab}
        label={active.label}
        summary={active.summary}
        snapshot={active.query.data}
        isLoading={active.query.isLoading}
        updateMutation={active.update}
        replaceMutation={active.replace}
        yamlMutation={active.yaml}
      />
    </div>
  );
}
