import { useState } from "react";

interface Props {
  onChange: (value: Record<string, unknown> | undefined) => void;
}

export function OverridesInput({ onChange }: Props) {
  const [raw, setRaw] = useState("{}");
  const [error, setError] = useState<string | null>(null);

  const handleBlur = () => {
    if (!raw.trim()) {
      onChange(undefined);
      setError(null);
      return;
    }
    try {
      const parsed = JSON.parse(raw);
      setError(null);
      onChange(parsed);
    } catch (err) {
      setError("Overrides must be valid JSON");
    }
  };

  return (
    <label>
      <span>Config overrides (JSON)</span>
      <textarea
        rows={5}
        value={raw}
        onChange={(event) => {
          setRaw(event.target.value);
        }}
        onBlur={handleBlur}
        placeholder="{""build_pair"": {""time_tolerance_s"": 0.2}}"
      />
      <span className="muted">Only provided keys will override the stored defaults.</span>
      {error && <span className="alert error" style={{ marginTop: "0.5rem" }}>{error}</span>}
    </label>
  );
}

export default OverridesInput;
