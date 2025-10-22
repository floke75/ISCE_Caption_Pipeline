import { useEffect, useMemo, useState } from 'react';
import { OverrideEntry } from '../types';
import '../styles/forms.css';

interface OverrideEditorProps {
  value?: OverrideEntry[];
  onChange: (entries: OverrideEntry[], objectValue: Record<string, unknown>) => void;
}

const DEFAULT_ROW: OverrideEntry = { path: '', value: '' };

function parseValue(raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed) {
    return '';
  }
  if (trimmed === 'true') return true;
  if (trimmed === 'false') return false;
  if (trimmed === 'null') return null;
  const maybeNumber = Number(trimmed);
  if (!Number.isNaN(maybeNumber) && trimmed === maybeNumber.toString()) {
    return maybeNumber;
  }
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      return JSON.parse(trimmed);
    } catch (error) {
      return trimmed;
    }
  }
  return trimmed;
}

function toNested(entries: OverrideEntry[]): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const entry of entries) {
    if (!entry.path.trim()) continue;
    const parts = entry.path.split('.').map((segment) => segment.trim()).filter(Boolean);
    if (!parts.length) continue;
    let cursor: Record<string, unknown> = result;
    parts.slice(0, -1).forEach((part) => {
      if (!(part in cursor) || typeof cursor[part] !== 'object' || cursor[part] === null) {
        cursor[part] = {};
      }
      cursor = cursor[part] as Record<string, unknown>;
    });
    cursor[parts[parts.length - 1]] = parseValue(entry.value);
  }
  return result;
}

export function OverrideEditor({ value, onChange }: OverrideEditorProps) {
  const [rows, setRows] = useState<OverrideEntry[]>(value?.length ? value : [DEFAULT_ROW]);

  useEffect(() => {
    setRows(value?.length ? value : [DEFAULT_ROW]);
  }, [value]);

  const objectValue = useMemo(() => toNested(rows), [rows]);

  useEffect(() => {
    onChange(rows, objectValue);
  }, [rows, objectValue, onChange]);

  const updateRow = (index: number, patch: Partial<OverrideEntry>) => {
    setRows((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], ...patch };
      return next;
    });
  };

  const addRow = () => {
    setRows((prev) => [...prev, { ...DEFAULT_ROW }]);
  };

  const removeRow = (index: number) => {
    setRows((prev) => {
      if (prev.length === 1) {
        return [{ ...DEFAULT_ROW }];
      }
      return prev.filter((_, idx) => idx !== index);
    });
  };

  return (
    <div className="form-card" style={{ gap: '0.75rem' }}>
      <div>
        <p className="section-title">Per-run overrides</p>
        <p className="section-subtitle">Specify dotted paths to override individual config values for this job only.</p>
      </div>
      {rows.map((row, index) => (
        <div key={index} className="form-grid" style={{ alignItems: 'flex-end' }}>
          <label className="field">
            <span>Config key path</span>
            <input
              type="text"
              placeholder="e.g. build_pair.spacy_enable"
              value={row.path}
              onChange={(event) => updateRow(index, { path: event.target.value })}
            />
          </label>
          <label className="field">
            <span>Value</span>
            <input
              type="text"
              placeholder="true | 0.5 | ['mp4']"
              value={row.value}
              onChange={(event) => updateRow(index, { value: event.target.value })}
            />
          </label>
          <button type="button" className="ghost" onClick={() => removeRow(index)}>
            Remove
          </button>
        </div>
      ))}
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
        <button type="button" className="ghost" onClick={addRow}>
          Add override
        </button>
      </div>
      <div>
        <p className="section-subtitle">Preview</p>
        <pre style={{ margin: 0, background: 'rgba(15,23,42,0.8)', padding: '0.75rem', borderRadius: '12px', maxHeight: '200px', overflow: 'auto' }}>
{JSON.stringify(objectValue, null, 2)}
        </pre>
      </div>
    </div>
  );
}
