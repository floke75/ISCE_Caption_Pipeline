import { Fragment, useMemo, useState } from "react";
import { PipelineConfig } from "../hooks/usePipelineConfig";

interface Props {
  config: PipelineConfig;
  isLoading: boolean;
  isUpdating: boolean;
  onSave: (data: PipelineConfig) => Promise<void>;
  onReload: () => void;
}

interface NodeProps {
  path: (string | number)[];
  value: unknown;
  onChange: (path: (string | number)[], value: unknown) => void;
}

function updateNested(config: PipelineConfig, path: (string | number)[], value: unknown): PipelineConfig {
  if (path.length === 0) return config;
  const [head, ...rest] = path;
  const clone: any = Array.isArray(config) ? [...config] : { ...config };
  if (rest.length === 0) {
    clone[head as any] = value;
    return clone;
  }
  const current = clone[head as any];
  clone[head as any] = updateNested(current ?? {}, rest, value);
  return clone;
}

function ConfigNode({ path, value, onChange }: NodeProps) {
  if (Array.isArray(value)) {
    return (
      <label>
        <span>{path[path.length - 1]}</span>
        <textarea
          value={JSON.stringify(value, null, 2)}
          rows={4}
          onChange={(event) => {
            try {
              const parsed = JSON.parse(event.target.value || "[]");
              onChange(path, parsed);
            } catch (error) {
              // ignore parsing errors while the user types
            }
          }}
        />
        <span className="muted">Array value encoded as JSON</span>
      </label>
    );
  }

  if (value !== null && typeof value === "object") {
    return (
      <fieldset className="fieldset">
        <legend>{String(path[path.length - 1] ?? "root")}</legend>
        {Object.entries(value as Record<string, unknown>).map(([key, child]) => (
          <ConfigNode key={key} path={[...path, key]} value={child} onChange={onChange} />
        ))}
      </fieldset>
    );
  }

  if (typeof value === "boolean") {
    return (
      <label>
        <span>{path[path.length - 1]}</span>
        <div className="inline-controls">
          <input
            type="checkbox"
            checked={value}
            onChange={(event) => onChange(path, event.target.checked)}
          />
          <span>{value ? "Enabled" : "Disabled"}</span>
        </div>
      </label>
    );
  }

  if (typeof value === "number") {
    return (
      <label>
        <span>{path[path.length - 1]}</span>
        <input
          type="number"
          value={value}
          onChange={(event) => onChange(path, Number(event.target.value))}
        />
      </label>
    );
  }

  return (
    <label>
      <span>{path[path.length - 1]}</span>
      <input
        type="text"
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(event) => onChange(path, event.target.value)}
      />
    </label>
  );
}

export function ConfigEditor({ config, isLoading, isUpdating, onSave, onReload }: Props) {
  const [draft, setDraft] = useState(config);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const entries = useMemo(() => Object.entries(draft ?? {}), [draft]);

  if (isLoading) {
    return <div className="card">Loading configuration...</div>;
  }

  const handleChange = (path: (string | number)[], value: unknown) => {
    setDraft((prev) => updateNested(prev ?? {}, path, value));
  };

  const handleSave = async () => {
    try {
      await onSave(draft ?? {});
      setMessage({ type: "success", text: "Configuration updated" });
    } catch (error: any) {
      setMessage({ type: "error", text: error?.message ?? "Failed to update configuration" });
    }
  };

  const handleReload = () => {
    setDraft(config);
    setMessage(null);
    onReload();
  };

  return (
    <div className="card">
      <div className="inline-controls" style={{ justifyContent: "space-between" }}>
        <div>
          <h2>Pipeline configuration</h2>
          <p className="description">Edit stored defaults safely with structured inputs.</p>
        </div>
        <div className="config-actions">
          <button className="btn-secondary" type="button" onClick={handleReload}>
            Reset
          </button>
          <button className="btn-primary" type="button" onClick={handleSave} disabled={isUpdating}>
            {isUpdating ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
      {message && <div className={`alert ${message.type}`}>{message.text}</div>}
      <div className="form-grid">
        {entries.length === 0 && <span className="muted">Configuration file is empty.</span>}
        {entries.map(([key, value]) => (
          <Fragment key={key}>
            <ConfigNode path={[key]} value={value} onChange={handleChange} />
          </Fragment>
        ))}
      </div>
    </div>
  );
}

export default ConfigEditor;
