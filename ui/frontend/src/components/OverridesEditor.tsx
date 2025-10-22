import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchConfig } from "../api";
import { ConfigMap, ConfigValue } from "../types";
import {
  ConfigTree,
  cloneConfig,
  countOverrides,
  diffConfigs,
  mergeWithOverrides,
  setNestedValue,
} from "./configTree";

interface OverridesEditorProps {
  label: string;
  kind: "pipeline" | "model";
  value: ConfigMap | null;
  onChange: (value: ConfigMap | null) => void;
  description?: string;
}

export function OverridesEditor({ label, kind, value, onChange, description }: OverridesEditorProps) {
  const [baseConfig, setBaseConfig] = useState<ConfigMap | null>(null);
  const [config, setConfig] = useState<ConfigMap | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(Boolean(value));

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchConfig(kind);
      const base = cloneConfig(response.config);
      setBaseConfig(base);
      setConfig(base);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [kind]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!baseConfig) {
      return;
    }
    setConfig(mergeWithOverrides(baseConfig, value));
  }, [baseConfig, value]);

  const overridesPatch = useMemo(() => {
    if (!baseConfig || !config) {
      return undefined;
    }
    return diffConfigs(baseConfig, config);
  }, [baseConfig, config]);

  const overridesValue = useMemo(() => {
    if (!overridesPatch || typeof overridesPatch !== "object") {
      return null;
    }
    return overridesPatch as ConfigMap;
  }, [overridesPatch]);

  const overridesCount = useMemo(() => countOverrides(overridesValue ?? undefined), [overridesValue]);

  useEffect(() => {
    if (overridesCount > 0) {
      setExpanded(true);
    }
  }, [overridesCount]);

  const handleChange = (path: string[], newValue: ConfigValue) => {
    if (!baseConfig) {
      return;
    }
    setConfig((current) => {
      if (!current) {
        return current;
      }
      const updated = setNestedValue(current, path, newValue);
      const diff = diffConfigs(baseConfig, updated);
      const patch = diff && typeof diff === "object" ? (diff as ConfigMap) : null;
      onChange(patch);
      return updated;
    });
  };

  const handleReset = () => {
    if (!baseConfig) {
      return;
    }
    setConfig(cloneConfig(baseConfig));
    onChange(null);
  };

  return (
    <details
      className="overrides-editor"
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary className="overrides-editor__summary">
        <span>{label}</span>
        <span className="overrides-editor__count">
          {overridesCount > 0 ? `${overridesCount} override${overridesCount === 1 ? "" : "s"}` : "None"}
        </span>
      </summary>
      <div className="overrides-editor__body">
        {description && <p className="overrides-editor__description muted">{description}</p>}
        {error && <div className="form__message form__message--error">{error}</div>}
        {loading && <p className="muted">Loading configuration…</p>}
        {!loading && !error && config && baseConfig && (
          <>
            <ConfigTree
              value={config}
              originalValue={baseConfig}
              onChange={handleChange}
              rootLabel="Configuration overrides"
            />
            <div className="overrides-editor__actions">
              <button
                type="button"
                className="button button--secondary button--tiny"
                onClick={handleReset}
                disabled={overridesCount === 0}
              >
                Clear overrides
              </button>
              <span className="muted">
                {overridesCount > 0 ? `${overridesCount} override${overridesCount === 1 ? "" : "s"} selected` : "No overrides selected"}
              </span>
            </div>
          </>
        )}
      </div>
    </details>
  );
}

export default OverridesEditor;
