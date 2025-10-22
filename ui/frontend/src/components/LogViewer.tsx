import { useEffect, useRef, useState } from "react";
import { fetchJobLog } from "../api";
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
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setContent("");
    setComplete(false);
    setError(null);
    offsetRef.current = 0;
  }, [jobId]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function loadChunk() {
      try {
        const chunk = await fetchJobLog(jobId, offsetRef.current);
        if (cancelled) {
          return;
        }
        if (chunk.content) {
          setContent((current) => current + chunk.content);
          offsetRef.current = chunk.offset;
        }
        setComplete(chunk.complete);
      } catch (err) {
        setError((err as Error).message);
      }
    }

    loadChunk();
    const interval = status === "running" && !complete ? 2000 : 4000;
    timer = window.setInterval(loadChunk, interval);

    return () => {
      cancelled = true;
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, [jobId, status, complete]);

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
        {complete ? <span className="muted">Log capture complete.</span> : <span className="muted">Streaming logs…</span>}
      </footer>
    </div>
  );
}

export default LogViewer;
