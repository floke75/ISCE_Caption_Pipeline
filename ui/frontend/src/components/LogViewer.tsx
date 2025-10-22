import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE } from "../api";
import { JobStatus } from "../types";

type ConnectionState = "connecting" | "open" | "closed" | "error";

interface LogViewerProps {
  jobId: string;
  status: JobStatus;
}

export function LogViewer({ jobId, status }: LogViewerProps) {
  const [content, setContent] = useState("");
  const [complete, setComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [autoScrollPaused, setAutoScrollPaused] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>("connecting");
  const [streamToken, setStreamToken] = useState(0);
  const offsetRef = useRef(0);
  const sourceRef = useRef<EventSource | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setContent("");
    setComplete(false);
    setError(null);
    setAutoScroll(true);
    setAutoScrollPaused(false);
    offsetRef.current = 0;
    setConnectionState("connecting");
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, [jobId]);

  useEffect(() => {
    const url = new URL(`${API_BASE}/jobs/${jobId}/log/stream`, window.location.origin);
    if (offsetRef.current > 0) {
      url.searchParams.set("offset", String(offsetRef.current));
    }

    const source = new EventSource(url.toString());
    sourceRef.current = source;
    setConnectionState("connecting");

    source.onopen = () => {
      setConnectionState("open");
      setError(null);
    };

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
      setConnectionState("closed");
      source.close();
      if (sourceRef.current === source) {
        sourceRef.current = null;
      }
    };

    source.addEventListener("complete", handleComplete);

    source.onerror = () => {
      setConnectionState("error");
      setError("Disconnected from log stream. Use Reconnect to resume streaming.");
      source.close();
      if (sourceRef.current === source) {
        sourceRef.current = null;
      }
    };

    return () => {
      source.removeEventListener("complete", handleComplete);
      source.close();
      if (sourceRef.current === source) {
        sourceRef.current = null;
      }
    };
  }, [jobId, streamToken]);

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const element = containerRef.current;
    if (element) {
      element.scrollTop = element.scrollHeight;
    }
  }, [content, autoScroll]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) {
      return;
    }
    const handleScroll = () => {
      const atBottom =
        element.scrollHeight - element.scrollTop - element.clientHeight < 16;
      if (!atBottom && autoScroll) {
        setAutoScroll(false);
        setAutoScrollPaused(true);
      } else if (atBottom && autoScrollPaused && !autoScroll) {
        setAutoScrollPaused(false);
      }
    };
    element.addEventListener("scroll", handleScroll);
    return () => {
      element.removeEventListener("scroll", handleScroll);
    };
  }, [autoScroll, autoScrollPaused]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleAutoScrollChange = (checked: boolean) => {
    setAutoScroll(checked);
    setAutoScrollPaused(false);
    if (checked) {
      requestAnimationFrame(() => {
        const element = containerRef.current;
        if (element) {
          element.scrollTop = element.scrollHeight;
        }
      });
    }
  };

  const handleReconnect = () => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setConnectionState("connecting");
    setError(null);
    setComplete(false);
    setStreamToken((current) => current + 1);
  };

  const connectionLabel = useMemo(() => {
    switch (connectionState) {
      case "open":
        return "Live";
      case "closed":
        return "Complete";
      case "error":
        return "Disconnected";
      default:
        return "Connecting…";
    }
  }, [connectionState]);

  const reconnectDisabled = connectionState === "connecting" || connectionState === "open";

  return (
    <div className="log-viewer">
      <header className="log-viewer__header">
        <div className="log-viewer__heading">
          <strong>Job log</strong>
          <span
            className={`log-viewer__status log-viewer__status--${connectionState}`}
            aria-live="polite"
          >
            <span className="log-viewer__status-indicator" aria-hidden />
            {connectionLabel}
          </span>
        </div>
        <div className="log-viewer__controls">
          <label className="toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(event) => handleAutoScrollChange(event.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
          <button
            type="button"
            className="button button--secondary"
            onClick={handleReconnect}
            disabled={reconnectDisabled}
          >
            Reconnect
          </button>
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
        <span className="muted">
          {complete
            ? "Log capture complete."
            : status === "running"
              ? "Streaming logs…"
              : "Waiting for log output…"}
        </span>
        {autoScrollPaused && (
          <span className="muted log-viewer__hint">Auto-scroll paused — toggle to resume.</span>
        )}
      </footer>
    </div>
  );
}

export default LogViewer;
