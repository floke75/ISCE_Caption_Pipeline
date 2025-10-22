interface Props {
  log: string;
  onCopy?: () => void;
}

export function LogViewer({ log, onCopy }: Props) {
  return (
    <div className="card">
      <div className="inline-controls" style={{ justifyContent: "space-between" }}>
        <h2>Logs</h2>
        <button className="btn-secondary" onClick={onCopy} type="button">
          Copy log
        </button>
      </div>
      <div className="log-viewer">{log || "No log output yet."}</div>
    </div>
  );
}

export default LogViewer;
