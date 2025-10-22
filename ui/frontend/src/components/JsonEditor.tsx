interface JsonEditorProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  error?: string | null;
  placeholder?: string;
}

export function JsonEditor({ label, value, onChange, error, placeholder }: JsonEditorProps) {
  return (
    <label className="form-field">
      <span className="form-field__label">{label}</span>
      <textarea
        className={`form-field__input form-field__input--monospace ${error ? "form-field__input--error" : ""}`}
        value={value}
        placeholder={placeholder ?? "Paste JSON overrides"}
        rows={6}
        onChange={(event) => onChange(event.target.value)}
      />
      {error && <span className="form-field__error">{error}</span>}
    </label>
  );
}

export default JsonEditor;
