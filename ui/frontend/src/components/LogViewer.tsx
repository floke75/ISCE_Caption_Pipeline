import { useEffect, useRef, useState } from "react";
import { API_BASE } from "../api";
import { JobStatus } from "../types";

interface LogViewerProps {
  jobId: string;
  status: JobStatus;
}

export function LogViewer({ jobId, status }: LogViewerProps) {
  const [content, setContent] = useState("");
  const [complete, setComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const offsetRef = useRef(0);
  const sourceRef = useRef<EventSource | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setContent("");
    setComplete(false);
    setError(null);
    offsetRef.current = 0;
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, [jobId]);

  useEffect(() => {
    const url = `${API_BASE}/jobs/${jobId}/log/stream`;
    const source = new EventSource(url);
    sourceRef.current = source;

    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data ?? "{}");
        if (typeof payload.offset === "number") {
          offsetRef.current = payload.offset;
        }
        if (typeof payload.complete === "boolean") {
          setComplete(payload.complete);
        }
        if (typeof payload.content === "string" && payload.content.length > 0) {
          setContent((current) => current + payload.content);
        }
        setError(null);
      } catch (err) {
        setError((err as Error).message);
      }
    };

    const handleComplete = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data ?? "{}");
        if (typeof payload.offset === "number") {
          offsetRef.current = payload.offset;
        }
      } catch (err) {
        setError((err as Error).message);
      }
      setComplete(true);
      source.close();
      sourceRef.current = null;
    };

    source.addEventListener("complete", handleComplete);

    source.onerror = () => {
      setError("Disconnected from log stream; attempting to reconnect…");
    };

    return () => {
      source.removeEventListener("complete", handleComplete);
      source.close();
      if (sourceRef.current === source) {
        sourceRef.current = null;
      }
    };
  }, [jobId]);

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const element = containerRef.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [content, autoScroll]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="log-viewer">
      <header className="log-viewer__header">
        <strong>Job log</strong>
        <div className="log-viewer__controls">
          <label className="toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(event) => setAutoScroll(event.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
          <button type="button" className="button button--secondary" onClick={handleCopy}>
            Copy log
          </button>
        </div>
      </header>
      {error && <div className="form__message form__message--error">{error}</div>}
      <div className="log-viewer__content" ref={containerRef}>
        <pre>{content || "No log output yet."}</pre>
      </div>
      <footer className="log-viewer__footer">
        {complete ? (
          <span className="muted">Log capture complete.</span>
        ) : (
          <span className="muted">
            {status === "running" ? "Streaming logs…" : "Waiting for log output…"}
          </span>
        )}
      </footer>
    </div>
  );
}

export default LogViewer;
