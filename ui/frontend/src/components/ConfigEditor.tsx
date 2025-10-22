import { useCallback, useEffect, useMemo, useState } from "react";
import { dump } from "js-yaml";
import { fetchConfig, updateConfig } from "../api";
import { ConfigMap, ConfigValue } from "../types";
import Card from "./Card";
import { ConfigTree, cloneConfig, diffConfigs, setNestedValue } from "./configTree";

type ConfigKind = "pipeline" | "model";

interface ConfigEditorProps {
  kind: ConfigKind;
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

  const patch = useMemo<ConfigMap | undefined>(() => {
    if (!config || !original) {
      return undefined;
    }
    const diff = diffConfigs(original, config);
    return diff && typeof diff === "object" ? (diff as ConfigMap) : undefined;
  }, [config, original]);

  const hasChanges = Boolean(patch);

  const handleChange = (path: string[], value: ConfigValue) => {
    setConfig((current) => {
      if (!current) {
        return current;
      }
      return setNestedValue(current, path, value);
    });
  };

  const handleSave = async () => {
    if (!patch) {
      setMessage("No changes to save");
      return;
    }
    try {
      setSaving(true);
      const updated = await updateConfig(kind, patch);
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
      <ConfigTree value={config} originalValue={original} onChange={handleChange} rootLabel="Configuration" />
    </Card>
  );
}

export default ConfigEditor;
