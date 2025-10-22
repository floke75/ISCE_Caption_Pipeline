import { useCallback, useEffect, useMemo, useState } from "react";
import { dump } from "js-yaml";
import { ConfigMap, ConfigValue } from "../types";
import { fetchConfig, updateConfig } from "../api";
import Card from "./Card";

type ConfigKind = "pipeline" | "model";

interface ConfigEditorProps {
  kind: ConfigKind;
}

function isObject(value: ConfigValue): value is ConfigMap {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function cloneConfig<T extends ConfigValue>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function getChild(original: ConfigValue, key: string): ConfigValue {
  if (isObject(original) && key in original) {
    return original[key];
  }
  return undefined;
}

function setNestedValue(source: ConfigMap, path: string[], newValue: ConfigValue): ConfigMap {
  const clone = { ...source };
  let cursor: ConfigMap = clone;
  for (let i = 0; i < path.length - 1; i += 1) {
    const key = path[i];
    const next = cursor[key];
    if (isObject(next)) {
      cursor[key] = { ...next };
    } else {
      cursor[key] = {};
    }
    cursor = cursor[key] as ConfigMap;
  }
  cursor[path[path.length - 1]] = newValue;
  return clone;
}

function diffConfigs(original: ConfigValue, current: ConfigValue): ConfigValue | undefined {
  if (isObject(current)) {
    if (!isObject(original)) {
      return cloneConfig(current);
    }
    const keys = new Set([...Object.keys(original), ...Object.keys(current)]);
    const diff: ConfigMap = {};
    keys.forEach((key) => {
      const childDiff = diffConfigs(original[key], current[key]);
      if (childDiff !== undefined) {
        diff[key] = childDiff;
      }
    });
    return Object.keys(diff).length > 0 ? diff : undefined;
  }
  if (isObject(original)) {
    return current;
  }
  if (
    typeof original === "number" &&
    typeof current === "number" &&
    Number.isNaN(original) &&
    Number.isNaN(current)
  ) {
    return undefined;
  }
  return original === current ? undefined : current;
}

function inferFieldType(original: ConfigValue, current: ConfigValue): "boolean" | "number" | "string" {
  if (typeof current === "boolean" || typeof original === "boolean") {
    return "boolean";
  }
  if (typeof current === "number" || typeof original === "number") {
    return "number";
  }
  return "string";
}

interface ConfigNodeProps {
  label: string;
  path: string[];
  value: ConfigValue;
  originalValue: ConfigValue;
  onChange: (path: string[], value: ConfigValue) => void;
}

function ConfigNode({ label, path, value, originalValue, onChange }: ConfigNodeProps) {
  if (isObject(value)) {
    return (
      <ConfigObjectEditor
        label={label}
        path={path}
        value={value}
        originalValue={originalValue}
        onChange={onChange}
      />
    );
  }
  return (
    <ConfigPrimitiveEditor
      label={label}
      path={path}
      value={value}
      originalValue={originalValue}
      onChange={onChange}
    />
  );
}

function ConfigObjectEditor({ label, path, value, originalValue, onChange }: ConfigNodeProps) {
  const [newKey, setNewKey] = useState("");
  const entries = useMemo(() => Object.entries(value).sort(([a], [b]) => a.localeCompare(b)), [value]);

  const handleAdd = () => {
    const trimmed = newKey.trim();
    if (!trimmed || Object.prototype.hasOwnProperty.call(value, trimmed)) {
      return;
    }
    onChange([...path, trimmed], "");
    setNewKey("");
  };

  return (
    <details className="config-node" open>
      <summary className="config-node__summary">{label}</summary>
      <div className="config-node__content">
        {entries.map(([key, child]) => (
          <ConfigNode
            key={key}
            label={key}
            path={[...path, key]}
            value={child}
            originalValue={getChild(originalValue, key)}
            onChange={onChange}
          />
        ))}
        <div className="config-node__add">
          <input
            type="text"
            placeholder="Add key"
            value={newKey}
            onChange={(event) => setNewKey(event.target.value)}
          />
          <button type="button" className="button button--secondary" onClick={handleAdd}>
            Add
          </button>
        </div>
      </div>
    </details>
  );
}

function ConfigPrimitiveEditor({ label, path, value, originalValue, onChange }: ConfigNodeProps) {
  const type = inferFieldType(originalValue, value);

  if (type === "boolean") {
    return (
      <label className="config-field config-field--boolean">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(path, event.target.checked)}
        />
        <span>{label}</span>
      </label>
    );
  }

  const displayValue = value === null || value === undefined ? "" : String(value);
  return (
    <label className="config-field">
      <span className="config-field__label">{label}</span>
      <input
        className="config-field__input"
        type={type === "number" ? "number" : "text"}
        value={displayValue}
        onChange={(event) => {
          if (type === "number") {
            const raw = event.target.value;
            onChange(path, raw === "" ? null : Number(raw));
          } else {
            onChange(path, event.target.value);
          }
        }}
      />
    </label>
  );
}

export function ConfigEditor({ kind }: ConfigEditorProps) {
  const [config, setConfig] = useState<ConfigMap | null>(null);
  const [original, setOriginal] = useState<ConfigMap | null>(null);
  const [yaml, setYaml] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const response = await fetchConfig(kind);
      setConfig(cloneConfig(response.config));
      setOriginal(cloneConfig(response.config));
      setYaml(response.yaml);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [kind]);

  useEffect(() => {
    load();
  }, [load]);

  const patch = useMemo(() => {
    if (!config || !original) {
      return undefined;
    }
    return diffConfigs(original, config);
  }, [config, original]);

  const hasChanges = patch !== undefined;

  const handleChange = (path: string[], value: ConfigValue) => {
    setConfig((current) => {
      if (!current) {
        return current;
      }
      return setNestedValue(current, path, value);
    });
  };

  const handleSave = async () => {
    if (!config || !patch || typeof patch !== "object") {
      setMessage("No changes to save");
      return;
    }
    try {
      setSaving(true);
      const updated = await updateConfig(kind, patch as ConfigMap);
      setConfig(cloneConfig(updated.config));
      setOriginal(cloneConfig(updated.config));
      setYaml(updated.yaml);
      setMessage("Configuration saved");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleDownload = () => {
    if (!config) {
      return;
    }
    const content = hasChanges ? dump(config) : yaml;
    const blob = new Blob([content], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = kind === "pipeline" ? "pipeline_config.yaml" : "config.yaml";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (loading || !config || !original) {
    return (
      <Card
        title={kind === "pipeline" ? "Pipeline configuration" : "Model configuration"}
        description="Loading configuration…"
      >
        <p className="muted">Fetching configuration from server…</p>
      </Card>
    );
  }

  return (
    <Card
      title={kind === "pipeline" ? "Pipeline configuration" : "Model configuration"}
      description="Review and tweak configuration values. Only modified keys are persisted."
      actions={
        <div className="config-actions">
          <button type="button" className="button button--secondary" onClick={handleDownload}>
            Download YAML
          </button>
          <button type="button" className="button button--secondary" onClick={load}>
            Reset
          </button>
          <button type="button" className="button" onClick={handleSave} disabled={!hasChanges || saving}>
            {saving ? "Saving…" : "Save changes"}
          </button>
        </div>
      }
    >
      {error && <div className="form__message form__message--error">{error}</div>}
      {message && <div className="form__message form__message--success">{message}</div>}
      <div className="config-tree">
        {Object.entries(config).map(([key, value]) => (
          <ConfigNode
            key={key}
            label={key}
            path={[key]}
            value={value}
            originalValue={getChild(original, key)}
            onChange={handleChange}
          />
        ))}
      </div>
    </Card>
  );
}

export default ConfigEditor;
