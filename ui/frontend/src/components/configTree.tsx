import { useMemo, useState } from "react";
import { ConfigMap, ConfigValue } from "../types";

export function isObject(value: ConfigValue | undefined): value is ConfigMap {
  return typeof value === "object" && value !== null;
}

export function cloneConfig<T extends ConfigValue>(value: T): T {
  if (isObject(value)) {
    return JSON.parse(JSON.stringify(value)) as T;
  }
  return value;
}

export function setNestedValue(source: ConfigMap, path: string[], newValue: ConfigValue): ConfigMap {
  const clone: ConfigMap = { ...source };
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

export function configsEqual(a: ConfigValue | undefined, b: ConfigValue | undefined): boolean {
  if (a === b) {
    return true;
  }
  if (typeof a === "number" && typeof b === "number") {
    return Number.isNaN(a) && Number.isNaN(b);
  }
  if (isObject(a) && isObject(b)) {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const key of keys) {
      if (!configsEqual(a[key], b[key])) {
        return false;
      }
    }
    return true;
  }
  return false;
}

export function diffConfigs(original: ConfigValue, current: ConfigValue): ConfigValue | undefined {
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

export function mergeWithOverrides(base: ConfigMap, overrides: ConfigMap | null | undefined): ConfigMap {
  if (!overrides) {
    return cloneConfig(base);
  }

  const merge = (baseline: ConfigValue | undefined, patch: ConfigValue): ConfigValue => {
    if (isObject(patch)) {
      const baselineObject = isObject(baseline) ? baseline : {};
      const result: ConfigMap = { ...baselineObject };
      Object.entries(patch).forEach(([key, child]) => {
        result[key] = merge(baselineObject[key], child);
      });
      return result;
    }
    return cloneConfig(patch);
  };

  return merge(base, overrides) as ConfigMap;
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

interface ConfigObjectEditorProps extends ConfigNodeProps {
  collapsible?: boolean;
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

function ConfigObjectEditor({ label, path, value, originalValue, onChange, collapsible = true }: ConfigObjectEditorProps) {
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

  const content = (
    <div className="config-node__content">
      {entries.map(([key, child]) => (
        <ConfigNode
          key={key}
          label={key}
          path={[...path, key]}
          value={child}
          originalValue={isObject(originalValue) ? originalValue[key] : undefined}
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
  );

  if (!collapsible) {
    return (
      <div className="config-node config-node--root">
        <div className="config-node__summary">{label}</div>
        {content}
      </div>
    );
  }

  return (
    <details className="config-node" open>
      <summary className="config-node__summary">{label}</summary>
      {content}
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

export interface ConfigTreeProps {
  value: ConfigMap;
  originalValue: ConfigMap;
  onChange: (path: string[], value: ConfigValue) => void;
  rootLabel?: string;
}

export function ConfigTree({ value, originalValue, onChange, rootLabel = "Configuration" }: ConfigTreeProps) {
  return (
    <div className="config-tree">
      <ConfigObjectEditor
        label={rootLabel}
        path={[]}
        value={value}
        originalValue={originalValue}
        onChange={onChange}
        collapsible={false}
      />
    </div>
  );
}

export function countOverrides(value: ConfigValue | undefined): number {
  if (value === undefined || value === null) {
    return 0;
  }
  if (isObject(value)) {
    return Object.values(value).reduce((total, child) => total + countOverrides(child), 0);
  }
  return 1;
}
