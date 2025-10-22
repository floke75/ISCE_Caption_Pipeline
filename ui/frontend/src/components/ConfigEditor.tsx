import { useEffect, useMemo, useState } from "react";
import { ConfigEnvelope, useConfig, useUpdateConfig } from "../api";
import { FormField } from "./FormField";

interface Props {
  configKey: "pipeline" | "core";
}

function flattenConfig(content: Record<string, unknown>, prefix = ""): Record<string, string> {
  return Object.entries(content).reduce<Record<string, string>>((acc, [key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    if (Array.isArray(value)) {
      acc[path] = JSON.stringify(value);
    } else if (value && typeof value === "object") {
      Object.assign(acc, flattenConfig(value as Record<string, unknown>, path));
    } else if (value !== undefined && value !== null) {
      acc[path] = String(value);
    }
    return acc;
  }, {});
}

function parseValue(value: string): unknown {
  const trimmed = value.trim();

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    try {
      return JSON.parse(trimmed);
    } catch {
      // fall through to other parsers if JSON parsing fails
    }
  }

  if (trimmed === "true" || trimmed === "false") {
    return trimmed === "true";
  }
  if (/[/\\]/.test(value) || value.includes(":")) {
    return value;
  }
  const num = Number(value);
  if (!Number.isNaN(num) && trimmed !== "") {
    return num;
  }
  return value;
}

function buildNestedConfig(flat: Record<string, string>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  Object.entries(flat).forEach(([key, value]) => {
    const parts = key.split(".");
    let cursor: Record<string, unknown> = result;
    parts.forEach((part, index) => {
      if (index === parts.length - 1) {
        cursor[part] = parseValue(value);
      } else {
        if (!cursor[part] || typeof cursor[part] !== "object") {
          cursor[part] = {};
        }
        cursor = cursor[part] as Record<string, unknown>;
      }
    });
  });
  return result;
}

export function ConfigEditor({ configKey }: Props) {
  const { data, isLoading, error } = useConfig(configKey);
  const updateConfig = useUpdateConfig(configKey);
  const [localConfig, setLocalConfig] = useState<Record<string, string>>({});
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  const flattened = useMemo(() => {
    if (!data) return {};
    return flattenConfig(data.overrides);
  }, [data]);

  useEffect(() => {
    setLocalConfig(flattened);
  }, [flattened]);

  if (isLoading) {
    return <p className="text-sm text-slate-400">Loading configuration…</p>;
  }

  if (error) {
    return <p className="text-sm text-rose-300">Failed to load config: {(error as Error).message}</p>;
  }

  const handleChange = (key: string, value: string) => {
    setLocalConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    updateConfig.mutate(buildNestedConfig(localConfig));
  };

  const handleAdd = () => {
    if (!newKey) return;
    setLocalConfig((prev) => ({ ...prev, [newKey]: newValue }));
    setNewKey("");
    setNewValue("");
  };

  const config: ConfigEnvelope | undefined = data;

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      {config && (
        <div className="rounded-lg bg-slate-900/40 p-4 text-xs text-slate-400">
          <div className="mb-2 font-semibold text-slate-200">Effective values</div>
          <pre className="whitespace-pre-wrap break-all">{JSON.stringify(config.resolved, null, 2)}</pre>
        </div>
      )}
      <div className="space-y-3">
        {Object.entries(localConfig).map(([key, value]) => (
          <div key={key} className="flex items-center gap-3">
            <div className="flex-1">
              <FormField label={key}>
                <input
                  className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
                  value={value}
                  onChange={(event) => handleChange(key, event.target.value)}
                />
              </FormField>
            </div>
            <button
              type="button"
              onClick={() => {
                setLocalConfig((prev) => {
                  const copy = { ...prev };
                  delete copy[key];
                  return copy;
                });
              }}
              className="rounded-lg border border-slate-700 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-300 transition hover:bg-rose-500/20 hover:text-rose-200"
            >
              Remove
            </button>
          </div>
        ))}
        {Object.keys(localConfig).length === 0 && (
          <p className="text-sm text-slate-500">
            No overrides saved yet. Add keys to <code>pipeline_config.yaml</code> or <code>config.yaml</code> to customize the
            pipeline.
          </p>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-dashed border-slate-700 bg-slate-900/40 p-4 text-sm text-slate-300">
        <div className="flex flex-1 flex-col gap-2 md:flex-row">
          <input
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            placeholder="new.key.path"
            value={newKey}
            onChange={(event) => setNewKey(event.target.value)}
          />
          <input
            className="w-full rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 focus:border-brand-400 focus:outline-none"
            placeholder="value"
            value={newValue}
            onChange={(event) => setNewValue(event.target.value)}
          />
        </div>
        <button
          type="button"
          onClick={handleAdd}
          className="rounded-lg bg-slate-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-100 transition hover:bg-brand-500"
        >
          Add override
        </button>
      </div>
      <button
        type="submit"
        className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-brand-500/30 transition hover:bg-brand-400"
        disabled={updateConfig.isLoading}
      >
        {updateConfig.isLoading ? "Saving…" : "Save configuration"}
      </button>
      {updateConfig.isSuccess && <div className="text-sm text-emerald-300">Configuration updated.</div>}
      {updateConfig.isError && <div className="text-sm text-rose-300">Save failed: {(updateConfig.error as Error).message}</div>}
    </form>
  );
}
