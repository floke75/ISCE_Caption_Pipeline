import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { usePipelineConfig, useSegmentationConfig } from '../hooks/useConfig';
import { ConfigNode } from '../types';
import { buildNested } from '../utils/overrides';
import '../styles/forms.css';

export type OverridePatches = {
  pipeline: Record<string, unknown>;
  segmentation: Record<string, unknown>;
};

export type OverrideResource = 'pipeline' | 'segmentation';

export type OverrideEdits = Record<OverrideResource, Record<string, unknown>>;

export interface OverrideEditorProps {
  onChange: (patches: OverridePatches, hasErrors: boolean) => void;
  edits?: OverrideEdits;
  onEditsChange?: (edits: OverrideEdits) => void;
}

type OverrideErrors = Record<string, string>;

type CoerceResult = {
  value?: unknown;
  error?: string;
  unset?: boolean;
};

function valuesEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function formatValue(value: unknown): string {
  if (value === null) return 'null';
  if (value === undefined) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === 'object') return JSON.stringify(value, null, 0);
  return String(value ?? '—');
}

function parseScalar(token: string): unknown {
  const trimmed = token.trim();
  if (!trimmed.length) return undefined;
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (trimmed === 'null') return null;
  const maybeNumber = Number(trimmed);
  if (!Number.isNaN(maybeNumber) && trimmed === maybeNumber.toString()) {
    return maybeNumber;
  }
  return trimmed;
}

function coerceValue(node: ConfigNode, raw: unknown): CoerceResult {
  switch (node.valueType) {
    case 'boolean':
      return { value: Boolean(raw) };
    case 'number': {
      if (typeof raw !== 'string') {
        return { error: 'Expected numeric input' };
      }
      const trimmed = raw.trim();
      if (!trimmed.length) {
        return { unset: true };
      }
      const asNumber = Number(trimmed);
      if (Number.isNaN(asNumber)) {
        return { error: 'Enter a valid number' };
      }
      return { value: asNumber };
    }
    case 'list': {
      if (typeof raw !== 'string') {
        return { error: 'Enter a JSON array or comma-separated values' };
      }
      const trimmed = raw.trim();
      if (!trimmed.length) {
        return { unset: true };
      }
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) {
          return { value: parsed };
        }
      } catch {
        // Fall through to custom parsing
      }
      const tokens = trimmed
        .split(/[,\n]/)
        .map((segment) => parseScalar(segment))
        .filter((value) => value !== undefined);
      return { value: tokens };
    }
    case 'select':
    case 'path':
    case 'string':
    default:
      if (typeof raw !== 'string') {
        return { error: 'Expected text input' };
      }
      return { value: raw };
  }
}

interface TreeItemProps {
  node: ConfigNode;
  depth: number;
  showAdvanced: boolean;
  edits: Record<string, unknown>;
  errors: OverrideErrors;
  onValueChange: (node: ConfigNode, raw: unknown) => void;
  onClear: (node: ConfigNode) => void;
}

function OverrideTreeItem({
  node,
  depth,
  showAdvanced,
  edits,
  errors,
  onValueChange,
  onClear,
}: TreeItemProps): JSX.Element | null {
  const hasChildren = Boolean(node.children && node.children.length);
  if (hasChildren) {
    const childElements = (node.children ?? [])
      .map((child) => (
        <OverrideTreeItem
          key={child.path.join('.')}
          node={child}
          depth={depth + 1}
          showAdvanced={showAdvanced}
          edits={edits}
          errors={errors}
          onValueChange={onValueChange}
          onClear={onClear}
        />
      ))
      .filter((child): child is JSX.Element => Boolean(child));

    if (!childElements.length && node.advanced && !showAdvanced) {
      return null;
    }

    return (
      <details className="override-branch" open style={{ marginLeft: depth ? depth * 16 : 0 }}>
        <summary>
          <span className="override-branch-label">{node.label}</span>
        </summary>
        <div className="override-children">{childElements}</div>
      </details>
    );
  }

  if (node.advanced && !showAdvanced) {
    return null;
  }

  const dotted = node.path.join('.');
  const error = errors[dotted];
  const overrideValue = edits[dotted];
  const currentDisplay = formatValue(node.current ?? node.default);
  const defaultDisplay = formatValue(node.default);

  const handleInputChange = (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    onValueChange(node, event.target.value);
  };

  const renderInput = () => {
    switch (node.valueType) {
      case 'boolean': {
        const checkedValue =
          typeof overrideValue === 'boolean'
            ? overrideValue
            : Boolean(overrideValue ?? node.current ?? node.default);
        return (
          <div className="override-control">
            <label className="toggle">
              <input
                type="checkbox"
                checked={checkedValue}
                onChange={(event) => onValueChange(node, event.target.checked)}
              />
              <span>Override value</span>
            </label>
          </div>
        );
      }
      case 'number':
        return (
          <input
            type="number"
            value={
              typeof overrideValue === 'number'
                ? overrideValue
                : typeof node.current === 'number'
                ? node.current
                : typeof node.default === 'number'
                ? node.default
                : ''
            }
            onChange={handleInputChange}
          />
        );
      case 'list': {
        const listValue = Array.isArray(overrideValue)
          ? overrideValue
          : Array.isArray(node.current)
          ? node.current
          : Array.isArray(node.default)
          ? node.default
          : undefined;
        const textValue =
          listValue !== undefined
            ? JSON.stringify(listValue, null, 2)
            : typeof overrideValue === 'string'
            ? overrideValue
            : typeof node.current === 'string'
            ? node.current
            : typeof node.default === 'string'
            ? node.default
            : '';
        return <textarea value={textValue} onChange={handleInputChange} placeholder='["value1", "value2"]' />;
      }
      case 'select': {
        return (
          <select
            value={typeof overrideValue === 'string' ? overrideValue : '__CONFIG__'}
            onChange={(event) => {
              if (event.target.value === '__CONFIG__') {
                onClear(node);
              } else {
                onValueChange(node, event.target.value);
              }
            }}
          >
            <option value="__CONFIG__">Use configured value ({currentDisplay})</option>
            {(node.options ?? []).map((option) => (
              <option key={String(option)} value={String(option)}>
                {String(option)}
              </option>
            ))}
          </select>
        );
      }
      default:
        return (
          <input
            type="text"
            value={
              typeof overrideValue === 'string'
                ? overrideValue
                : typeof node.current === 'string'
                ? node.current
                : typeof node.default === 'string'
                ? node.default
                : ''
            }
            onChange={handleInputChange}
          />
        );
    }
  };

  const showClear = dotted in edits;

  return (
    <div className="override-leaf" style={{ marginLeft: depth ? depth * 16 : 0 }}>
      <div className="override-leaf-header">
        <span className="override-leaf-label">{node.label}</span>
        {showClear ? (
          <button type="button" className="ghost" onClick={() => onClear(node)}>
            Clear override
          </button>
        ) : null}
      </div>
      {node.description ? <span className="field-help">{node.description}</span> : null}
      <div className="override-meta">
        <span>Current: {currentDisplay}</span>
        <span>Default: {defaultDisplay}</span>
      </div>
      {renderInput()}
      {error ? <span className="override-error">{error}</span> : null}
    </div>
  );
}

const RESOURCE_LABEL: Record<OverrideResource, string> = {
  pipeline: 'Pipeline configuration',
  segmentation: 'Segmentation configuration',
};

const RESOURCE_SUMMARY: Record<OverrideResource, string> = {
  pipeline: 'Paths, orchestration toggles, and pipeline defaults applied to every run.',
  segmentation: 'Segmentation model sliders and guardrails used when producing captions.',
};

type ConfigQueryResult = ReturnType<typeof usePipelineConfig>;

export function OverrideEditor({ onChange, edits: controlledEdits, onEditsChange }: OverrideEditorProps) {
  const pipelineQuery = usePipelineConfig();
  const segmentationQuery = useSegmentationConfig();

  const isControlled = controlledEdits !== undefined;

  const [internalEdits, setInternalEdits] = useState<OverrideEdits>(
    () => ({ pipeline: {}, segmentation: {} })
  );

  const edits = isControlled ? controlledEdits! : internalEdits;

  const updateEdits = useCallback((update: (prev: OverrideEdits) => OverrideEdits) => {
    if (isControlled) {
      if (onEditsChange) {
         onEditsChange(update(edits));
      }
    } else {
      setInternalEdits(update);
    }
  }, [isControlled, onEditsChange, edits]);

  const [activeTab, setActiveTab] = useState<OverrideResource>('pipeline');

  const [errors, setErrors] = useState<Record<OverrideResource, OverrideErrors>>(
    () => ({ pipeline: {}, segmentation: {} })
  );
  const [showAdvanced, setShowAdvanced] = useState<Record<OverrideResource, boolean>>(
    () => ({ pipeline: false, segmentation: false })
  );

  useEffect(() => {
    if (!isControlled && pipelineQuery.data?.effective) {
      updateEdits((prev) => ({ ...prev, pipeline: {} }));
      setErrors((prev) => ({ ...prev, pipeline: {} }));
    }
  }, [pipelineQuery.data?.effective, updateEdits, isControlled]);

  useEffect(() => {
    if (!isControlled && segmentationQuery.data?.effective) {
      updateEdits((prev) => ({ ...prev, segmentation: {} }));
      setErrors((prev) => ({ ...prev, segmentation: {} }));
    }
  }, [segmentationQuery.data?.effective, updateEdits, isControlled]);

  const pipelinePatch = useMemo(() => buildNested(edits.pipeline), [edits.pipeline]);
  const segmentationPatch = useMemo(() => buildNested(edits.segmentation), [edits.segmentation]);
  const hasErrors = useMemo(
    () => Object.values(errors).some((group) => Object.keys(group).length > 0),
    [errors]
  );

  useEffect(() => {
    onChange({ pipeline: pipelinePatch, segmentation: segmentationPatch }, hasErrors);
  }, [onChange, pipelinePatch, segmentationPatch, hasErrors]);

  const handleValueChange = useCallback((resource: OverrideResource, node: ConfigNode, raw: unknown) => {
    const dotted = node.path.join('.');
    const result = coerceValue(node, raw);

    setErrors((prev) => {
      const group = prev[resource];
      if (!(dotted in group)) {
        return prev;
      }
      const nextGroup = { ...group };
      delete nextGroup[dotted];
      return { ...prev, [resource]: nextGroup };
    });

    if (result.unset) {
      updateEdits((prev) => {
        const group = prev[resource];
        if (!(dotted in group)) {
          return prev;
        }
        const nextGroup = { ...group };
        delete nextGroup[dotted];
        return { ...prev, [resource]: nextGroup };
      });
      return;
    }

    if (result.error) {
      setErrors((prev) => ({
        ...prev,
        [resource]: { ...prev[resource], [dotted]: result.error! },
      }));
      updateEdits((prev) => {
        const group = prev[resource];
        if (!(dotted in group)) {
          return prev;
        }
        const nextGroup = { ...group };
        delete nextGroup[dotted];
        return { ...prev, [resource]: nextGroup };
      });
      return;
    }

    const value = result.value;
    if (valuesEqual(value, node.current)) {
      updateEdits((prev) => {
        const group = prev[resource];
        if (!(dotted in group)) {
          return prev;
        }
        const nextGroup = { ...group };
        delete nextGroup[dotted];
        return { ...prev, [resource]: nextGroup };
      });
      return;
    }

    updateEdits((prev) => ({
      ...prev,
      [resource]: { ...prev[resource], [dotted]: value },
    }));
  }, [updateEdits]);

  const handleClear = useCallback((resource: OverrideResource, node: ConfigNode) => {
    const dotted = node.path.join('.');
    updateEdits((prev) => {
      const group = prev[resource];
      if (!(dotted in group)) {
        return prev;
      }
      const nextGroup = { ...group };
      delete nextGroup[dotted];
      return { ...prev, [resource]: nextGroup };
    });
    setErrors((prev) => {
      const group = prev[resource];
      if (!(dotted in group)) {
        return prev;
      }
      const nextGroup = { ...group };
      delete nextGroup[dotted];
      return { ...prev, [resource]: nextGroup };
    });
  }, [updateEdits]);

  const handleClearAll = useCallback((resource: OverrideResource) => {
    updateEdits((prev) => ({ ...prev, [resource]: {} }));
    setErrors((prev) => ({ ...prev, [resource]: {} }));
  }, [updateEdits]);

  const queries: Record<OverrideResource, ConfigQueryResult> = {
    pipeline: pipelineQuery,
    segmentation: segmentationQuery,
  };

  const activeQuery = queries[activeTab];
  const activeData = activeQuery.data;
  const activeEdits = edits[activeTab];
  const activeErrors = errors[activeTab];
  const activeShowAdvanced = showAdvanced[activeTab];
  const overrideCount = Object.keys(activeEdits).length;
  const activePatch = useMemo(() => buildNested(activeEdits), [activeEdits]);
  const diffPreview = useMemo(() => JSON.stringify(activePatch, null, 2), [activePatch]);
  const activeHasErrors = Object.keys(activeErrors).length > 0;
  const diffWarning = activeHasErrors
    ? ' • resolve errors in this tab'
    : hasErrors
    ? ' • resolve errors in other tabs'
    : '';

  return (
    <div className="form-card" style={{ gap: '0.75rem' }}>
      <div className="override-header">
        <div>
          <p className="section-title">Per-run overrides</p>
          <p className="section-subtitle">
            Browse the pipeline and segmentation configuration trees, set typed overrides, and preview the merged diff for this
            job.
          </p>
        </div>
        <div className="override-actions">
          <button
            type="button"
            className="ghost"
            onClick={() =>
              setShowAdvanced((prev) => ({ ...prev, [activeTab]: !prev[activeTab] }))
            }
          >
            {activeShowAdvanced ? 'Hide advanced' : 'Show advanced'}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => handleClearAll(activeTab)}
            disabled={!overrideCount}
          >
            Clear {activeTab === 'pipeline' ? 'pipeline' : 'segmentation'} overrides
          </button>
        </div>
      </div>
      <div className="config-tabs">
        {(Object.keys(queries) as OverrideResource[]).map((resource) => (
          <button
            key={resource}
            type="button"
            className={resource === activeTab ? 'config-tab active' : 'config-tab'}
            onClick={() => setActiveTab(resource)}
          >
            {RESOURCE_LABEL[resource]}
          </button>
        ))}
      </div>
      <p className="section-subtitle">{RESOURCE_SUMMARY[activeTab]}</p>
      {activeQuery.isLoading ? (
        <div>Loading configuration…</div>
      ) : activeQuery.isError || !activeData ? (
        <div className="override-error">Failed to load configuration metadata.</div>
      ) : (
        <div className="override-tree">
          {activeData.schema
            .map((node) => (
              <OverrideTreeItem
                key={node.path.join('.')}
                node={node}
                depth={0}
                showAdvanced={activeShowAdvanced}
                edits={activeEdits}
                errors={activeErrors}
                onValueChange={(item, raw) => handleValueChange(activeTab, item, raw)}
                onClear={(item) => handleClear(activeTab, item)}
              />
            ))
            .filter((child): child is JSX.Element => Boolean(child))}
          {!activeData.schema.length ? <div>No configurable values found.</div> : null}
        </div>
      )}
      <div>
        <div className="override-summary">
          <p className="section-subtitle">
            Diff preview for {RESOURCE_LABEL[activeTab].toLowerCase()} ({overrideCount} override
            {overrideCount === 1 ? '' : 's'}){diffWarning}
          </p>
        </div>
        <pre className="override-preview">{overrideCount ? diffPreview : '{}'}</pre>
      </div>
    </div>
  );
}
