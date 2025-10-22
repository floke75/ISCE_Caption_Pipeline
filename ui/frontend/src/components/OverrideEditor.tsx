import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import { usePipelineConfig } from '../hooks/useConfig';
import { ConfigNode } from '../types';
import '../styles/forms.css';

interface OverrideEditorProps {
  onChange: (patch: Record<string, unknown>, hasErrors: boolean) => void;
}

type OverrideErrors = Record<string, string>;

type CoerceResult = {
  value?: unknown;
  error?: string;
  unset?: boolean;
};

function buildNested(edits: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [dotted, value] of Object.entries(edits)) {
    const parts = dotted.split('.');
    let cursor: Record<string, unknown> = result;
    parts.slice(0, -1).forEach((part) => {
      if (!(part in cursor) || typeof cursor[part] !== 'object' || cursor[part] === null) {
        cursor[part] = {};
      }
      cursor = cursor[part] as Record<string, unknown>;
    });
    cursor[parts[parts.length - 1]] = value;
  }
  return result;
}

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
      } catch (error) {
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

export function OverrideEditor({ onChange }: OverrideEditorProps) {
  const { data, isLoading, isError } = usePipelineConfig();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [edits, setEdits] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<OverrideErrors>({});

  useEffect(() => {
    setEdits({});
    setErrors({});
  }, [data?.effective]);

  const patch = useMemo(() => buildNested(edits), [edits]);
  const hasErrors = useMemo(() => Object.keys(errors).length > 0, [errors]);

  useEffect(() => {
    onChange(patch, hasErrors);
  }, [patch, hasErrors, onChange]);

  const handleValueChange = useCallback(
    (node: ConfigNode, raw: unknown) => {
      const dotted = node.path.join('.');
      const result = coerceValue(node, raw);
      setErrors((prev) => {
        if (!(dotted in prev)) {
          return prev;
        }
        const next = { ...prev };
        delete next[dotted];
        return next;
      });
      if (result.unset) {
        setEdits((prev) => {
          if (!(dotted in prev)) {
            return prev;
          }
          const next = { ...prev };
          delete next[dotted];
          return next;
        });
        return;
      }
      if (result.error) {
        setErrors((prev) => ({ ...prev, [dotted]: result.error! }));
        setEdits((prev) => {
          if (!(dotted in prev)) {
            return prev;
          }
          const next = { ...prev };
          delete next[dotted];
          return next;
        });
        return;
      }
      const value = result.value;
      if (valuesEqual(value, node.current)) {
        setEdits((prev) => {
          if (!(dotted in prev)) {
            return prev;
          }
          const next = { ...prev };
          delete next[dotted];
          return next;
        });
        return;
      }
      setEdits((prev) => ({ ...prev, [dotted]: value }));
    },
    [setEdits, setErrors]
  );

  const handleClear = useCallback((node: ConfigNode) => {
    const dotted = node.path.join('.');
    setEdits((prev) => {
      if (!(dotted in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[dotted];
      return next;
    });
    setErrors((prev) => {
      if (!(dotted in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[dotted];
      return next;
    });
  }, []);

  const handleClearAll = () => {
    setEdits({});
    setErrors({});
  };

  const overrideCount = Object.keys(edits).length;
  const diffPreview = useMemo(() => JSON.stringify(patch, null, 2), [patch]);

  return (
    <div className="form-card" style={{ gap: '0.75rem' }}>
      <div className="override-header">
        <div>
          <p className="section-title">Per-run overrides</p>
          <p className="section-subtitle">
            Browse the configuration tree, set typed overrides, and preview the diff that will be merged for this job.
          </p>
        </div>
        <div className="override-actions">
          <button type="button" className="ghost" onClick={() => setShowAdvanced((prev) => !prev)}>
            {showAdvanced ? 'Hide advanced' : 'Show advanced'}
          </button>
          <button type="button" className="ghost" onClick={handleClearAll} disabled={!overrideCount}>
            Clear all
          </button>
        </div>
      </div>
      {isLoading ? (
        <div>Loading configuration…</div>
      ) : isError || !data ? (
        <div className="override-error">Failed to load configuration metadata.</div>
      ) : (
        <div className="override-tree">
          {data.schema
            .map((node) => (
              <OverrideTreeItem
                key={node.path.join('.')}
                node={node}
                depth={0}
                showAdvanced={showAdvanced}
                edits={edits}
                errors={errors}
                onValueChange={handleValueChange}
                onClear={handleClear}
              />
            ))
            .filter((child): child is JSX.Element => Boolean(child))}
          {!data.schema.length ? <div>No configurable values found.</div> : null}
        </div>
      )}
      <div>
        <div className="override-summary">
          <p className="section-subtitle">
            Diff preview ({overrideCount} override{overrideCount === 1 ? '' : 's'}){hasErrors ? ' • resolve errors to continue' : ''}
          </p>
        </div>
        <pre className="override-preview">{overrideCount ? diffPreview : '{}'}</pre>
      </div>
    </div>
  );
}
