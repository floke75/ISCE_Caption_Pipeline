import { useEffect, useMemo, useState } from "react";
import { browseEntries, validatePath } from "../api";
import { FileEntry } from "../types";

interface PathPickerProps {
  label: string;
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  required?: boolean;
  disabled?: boolean;
  expect?: "file" | "directory" | "any";
  description?: string;
}

export function PathPicker({
  label,
  value,
  onChange,
  placeholder,
  required = false,
  disabled = false,
  expect = "any",
  description
}: PathPickerProps) {
  const [browserOpen, setBrowserOpen] = useState(false);
  const [currentDir, setCurrentDir] = useState<string | undefined>(undefined);
  const [parentDir, setParentDir] = useState<string | null>(null);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [pendingDir, setPendingDir] = useState<string | undefined>(undefined);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [validationState, setValidationState] = useState<"idle" | "valid" | "invalid">("idle");

  const directories = useMemo(() => entries.filter((entry) => entry.type === "directory"), [entries]);
  const files = useMemo(() => entries.filter((entry) => entry.type === "file"), [entries]);

  useEffect(() => {
    if (!browserOpen) {
      return;
    }
    const target = pendingDir ?? currentDir ?? (value ? value.trim() || undefined : undefined);
    setLoading(true);
    browseEntries(target)
      .then((res) => {
        setCurrentDir(res.path);
        setParentDir(res.parent);
        setEntries(res.entries);
        setBrowseError(null);
      })
      .catch((err) => {
        setBrowseError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        setLoading(false);
      });
    setPendingDir(undefined);
  }, [browserOpen, pendingDir, value]);

  const handleToggleBrowser = () => {
    if (disabled) {
      return;
    }
    setBrowserOpen((open) => {
      if (!open) {
        setPendingDir(value ? value.trim() || undefined : undefined);
      }
      return !open;
    });
  };

  const handleSelect = (entry: FileEntry) => {
    if (entry.type === "directory") {
      setPendingDir(entry.path);
      return;
    }
    onChange(entry.path);
    setBrowserOpen(false);
    setValidationState("idle");
    setValidationMessage(null);
  };

  const handleParent = () => {
    if (parentDir) {
      setPendingDir(parentDir);
    }
  };

  const handleValidate = async () => {
    if (!value.trim()) {
      setValidationState("invalid");
      setValidationMessage("Enter a path to validate");
      return;
    }
    try {
      const result = await validatePath(value.trim(), expect);
      onChange(result.path);
      setValidationState("valid");
      setValidationMessage(`Resolved ${result.type === "directory" ? "directory" : "file"}`);
    } catch (err) {
      setValidationState("invalid");
      setValidationMessage((err as Error).message);
    }
  };

  const validationClass =
    validationState === "valid"
      ? "form__message form__message--success"
      : validationState === "invalid"
      ? "form__message form__message--error"
      : "form__message";

  return (
    <div className="path-picker">
      <label className="form-field">
        <span className="form-field__label">{label}</span>
        <div className="path-picker__input">
          <input
            className="form-field__input"
            type="text"
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={placeholder}
            required={required}
            disabled={disabled}
          />
          <div className="path-picker__actions">
            <button
              type="button"
              className="button button--secondary"
              onClick={handleValidate}
              disabled={disabled}
            >
              Validate
            </button>
            <button
              type="button"
              className="button button--secondary"
              onClick={handleToggleBrowser}
              disabled={disabled}
            >
              {browserOpen ? "Close" : "Browse"}
            </button>
          </div>
        </div>
        {description && <span className="form-field__description">{description}</span>}
      </label>
      {validationMessage && <div className={validationClass}>{validationMessage}</div>}
      {browserOpen && (
        <div className="path-browser">
          <div className="path-browser__header">
            <strong>{currentDir ?? "Loading…"}</strong>
            <div className="path-browser__controls">
              <button type="button" className="button button--secondary" onClick={handleParent} disabled={!parentDir}>
                Up one level
              </button>
              <button
                type="button"
                className="button button--secondary"
                onClick={() => setPendingDir(currentDir)}
                disabled={loading}
              >
                Refresh
              </button>
            </div>
          </div>
          {browseError && <div className="form__message form__message--error">{browseError}</div>}
          <div className="path-browser__body">
            {loading ? (
              <p className="muted">Loading directory…</p>
            ) : (
              <ul className="path-browser__list">
                {directories.map((entry) => (
                  <li key={entry.path}>
                    <button type="button" className="link" onClick={() => handleSelect(entry)}>
                      📁 {entry.name}
                    </button>
                  </li>
                ))}
                {files.length > 0 && (
                  <li className="path-browser__divider" aria-hidden="true" />
                )}
                {files.map((entry) => (
                  <li key={entry.path}>
                    <button type="button" className="link" onClick={() => handleSelect(entry)}>
                      📄 {entry.name}
                    </button>
                  </li>
                ))}
                {directories.length === 0 && files.length === 0 && <li className="muted">No entries</li>}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default PathPicker;
