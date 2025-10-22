import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';

import { fetchModelConfig, fetchPipelineConfig } from '../lib/api';

type ConfigType = 'pipeline' | 'model';

interface OverrideEditorProps {
  configType: ConfigType;
  label: string;
  onChange(overrides: Record<string, unknown>): void;
  onValidityChange?(valid: boolean): void;
}

interface ConfigNodeProps {
  name: string;
  value: unknown;
  path: string[];
  overrides: Record<string, unknown>;
  errors: Record<string, string>;
  onUpdate(path: string[], value: unknown): void;
  onClear(path: string[]): void;
  onError(path: string[], message?: string): void;
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const formatValue = (value: unknown): string => {
  if (typeof value === 'string') {
    return value || '""';
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  if (value === null || typeof value === 'undefined') {
    return 'null';
  }
  return JSON.stringify(value, null, 2);
};

const deepEqual = (a: unknown, b: unknown): boolean => {
  if (a === b) {
    return true;
  }
  if (typeof a !== typeof b) {
    return false;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      return false;
    }
    return a.every((item, index) => deepEqual(item, b[index]));
  }
  if (isPlainObject(a) && isPlainObject(b)) {
    const aKeys = Object.keys(a);
    const bKeys = Object.keys(b);
    if (aKeys.length !== bKeys.length) {
      return false;
    }
    return aKeys.every((key) => deepEqual(a[key], b[key]));
  }
  return false;
};

const getNested = (tree: Record<string, unknown>, path: string[]): unknown => {
  return path.reduce<unknown>((cursor, key) => {
    if (!cursor || typeof cursor !== 'object') {
      return undefined;
    }
    return (cursor as Record<string, unknown>)[key];
  }, tree);
};

const setNested = (
  tree: Record<string, unknown>,
  path: string[],
  value: unknown
): Record<string, unknown> => {
  if (path.length === 0) {
    return tree;
  }
  const [head, ...rest] = path;
  const next = { ...tree };
  if (rest.length === 0) {
    if (typeof value === 'undefined') {
      delete next[head];
    } else {
      next[head] = value;
    }
    return next;
  }

  const currentChild = isPlainObject(next[head]) ? (next[head] as Record<string, unknown>) : {};
  const updatedChild = setNested(currentChild, rest, value);
  if (Object.keys(updatedChild).length === 0) {
    delete next[head];
  } else {
    next[head] = updatedChild;
  }
  return next;
};

const sortedEntries = (value: Record<string, unknown>) =>
  Object.entries(value).sort(([a], [b]) => a.localeCompare(b));

function ConfigLeaf({
  name,
  value,
  path,
  overrides,
  errors,
  onUpdate,
  onClear,
  onError,
}: ConfigNodeProps) {
  const overrideValue = getNested(overrides, path);
  const pathKey = path.join('.');
  const [text, setText] = useState<string>('');

  useEffect(() => {
    if (overrideValue !== undefined) {
      setText(formatValue(overrideValue));
    } else {
      setText(formatValue(value));
    }
  }, [overrideValue, value]);

  const reset = () => {
    onError(path, undefined);
    onClear(path);
  };

  const handleStringChange = (next: string) => {
    setText(next);
    if (typeof value === 'string') {
      if (next === value) {
        reset();
      } else {
        onUpdate(path, next);
      }
    } else {
      onUpdate(path, next);
    }
  };

  const handleNumberChange = (next: string) => {
    setText(next);
    if (!next.trim()) {
      onError(path, 'Enter a numeric value.');
      return;
    }
    const parsed = Number(next);
    if (Number.isNaN(parsed)) {
      onError(path, 'Enter a numeric value.');
      return;
    }
    onError(path, undefined);
    if (typeof value === 'number' && parsed === value) {
      reset();
    } else {
      onUpdate(path, parsed);
    }
  };

  const handleBooleanChange = (checked: boolean) => {
    if (typeof value === 'boolean' && checked === value) {
      reset();
    } else {
      onUpdate(path, checked);
    }
  };

  const handleJsonChange = (next: string) => {
    setText(next);
    if (!next.trim()) {
      onError(path, 'Provide a valid JSON value.');
      return;
    }
    try {
      const parsed = JSON.parse(next);
      onError(path, undefined);
      if (deepEqual(parsed, value)) {
        reset();
      } else {
        onUpdate(path, parsed);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid JSON value.';
      onError(path, message);
    }
  };

  const valueType = useMemo(() => {
    if (Array.isArray(value)) {
      return 'array';
    }
    if (value === null || typeof value === 'undefined') {
      return 'null';
    }
    return typeof value;
  }, [value]);

  return (
    <div className="override-leaf">
      <div className="override-leaf-header">
        <span className="leaf-name">{name}</span>
        {overrideValue !== undefined && (
          <button type="button" className="ghost-button" onClick={reset}>
            Use default
          </button>
        )}
      </div>
      <div className="override-leaf-control">
        {valueType === 'boolean' ? (
          <label className="toggle">
            <input
              type="checkbox"
              checked={overrideValue !== undefined ? Boolean(overrideValue) : Boolean(value)}
              onChange={(event) => handleBooleanChange(event.target.checked)}
            />
            <span>{overrideValue !== undefined ? String(overrideValue) : String(value)}</span>
          </label>
        ) : valueType === 'number' ? (
          <input
            type="number"
            value={text}
            onChange={(event) => handleNumberChange(event.target.value)}
          />
        ) : valueType === 'array' || valueType === 'object' ? (
          <textarea value={text} onChange={(event) => handleJsonChange(event.target.value)} rows={4} />
        ) : (
          <input value={text} onChange={(event) => handleStringChange(event.target.value)} />
        )}
      </div>
      <div className="override-leaf-meta">
        <span className="default-value">Default: {formatValue(value)}</span>
        {overrideValue !== undefined && (
          <span className="override-value">Override: {formatValue(overrideValue)}</span>
        )}
      </div>
      {errors[pathKey] && <p className="field-hint error">{errors[pathKey]}</p>}
    </div>
  );
}

function ConfigNode(props: ConfigNodeProps) {
  const { name, value, path } = props;
  if (isPlainObject(value)) {
    return (
      <details className="override-branch" open>
        <summary>{name}</summary>
        <div className="override-branch-children">
          {sortedEntries(value).map(([childKey, childValue]) => (
            <ConfigNode
              key={[...path, childKey].join('.')}
              name={childKey}
              value={childValue}
              path={[...path, childKey]}
              {...props}
            />
          ))}
        </div>
      </details>
    );
  }

  return <ConfigLeaf {...props} />;
}

export function OverrideEditor({
  configType,
  label,
  onChange,
  onValidityChange,
}: OverrideEditorProps) {
  const query = useQuery({
    queryKey: ['config-schema', configType],
    queryFn: () => (configType === 'pipeline' ? fetchPipelineConfig() : fetchModelConfig()),
  });

  const [overrides, setOverrides] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    if (query.isError) {
      onValidityChange?.(false);
    } else if (!query.isLoading && Object.keys(errors).length === 0) {
      onValidityChange?.(true);
    }
  }, [errors, onValidityChange, query.isError, query.isLoading]);

  useEffect(() => {
    onChange(overrides);
  }, [onChange, overrides]);

  useEffect(() => {
    setOverrides({});
    setErrors({});
  }, [query.data]);

  const updateOverride = (path: string[], value: unknown) => {
    setOverrides((current) => setNested(current, path, value));
  };

  const clearOverride = (path: string[]) => {
    setOverrides((current) => setNested(current, path, undefined));
  };

  const updateError = (path: string[], message?: string) => {
    const key = path.join('.');
    setErrors((current) => {
      const next = { ...current };
      if (!message) {
        delete next[key];
      } else {
        next[key] = message;
      }
      onValidityChange?.(Object.keys(next).length === 0 && !query.isError);
      return next;
    });
  };

  return (
    <div className="field override-editor">
      <label>{label}</label>
      {query.isLoading && <p className="field-hint">Loading configuration…</p>}
      {query.isError && (
        <p className="field-hint error">Unable to load configuration schema.</p>
      )}
      {query.data && (
        <div className="override-tree">
          {sortedEntries(query.data as Record<string, unknown>).map(([key, value]) => (
            <ConfigNode
              key={key}
              name={key}
              value={value}
              path={[key]}
              overrides={overrides}
              errors={errors}
              onUpdate={updateOverride}
              onClear={clearOverride}
              onError={updateError}
            />
          ))}
        </div>
      )}
      <p className="field-hint">
        {Object.keys(overrides).length === 0
          ? 'No overrides selected; defaults will be used.'
          : 'Overrides ready to apply.'}
      </p>
    </div>
  );
}

export default OverrideEditor;
