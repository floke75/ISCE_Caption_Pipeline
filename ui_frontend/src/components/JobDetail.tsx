import { useEffect, useMemo, useRef, useState } from 'react';
import { JobDetail as JobDetailType } from '../lib/api';
import { LogStreamStatus, useJobLogs } from '../hooks/useJobLogs';
import { formatDistanceToNow } from '../lib/time';
import { copyToClipboard } from '../lib/clipboard';

interface JobDetailProps {
  job?: JobDetailType;
}

export function JobDetail({ job }: JobDetailProps) {
  const { log, status: logStatus, error: logError, reconnect, refresh, retryDelay } = useJobLogs(job?.id);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const logContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!autoScroll || !logContainerRef.current) {
      return;
    }
    logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
  }, [log, autoScroll]);

  useEffect(() => {
    setCopiedKey(null);
  }, [job?.id]);

  useEffect(() => {
    if (!copiedKey) {
      return undefined;
    }
    const timeout = window.setTimeout(() => setCopiedKey(null), 2000);
    return () => window.clearTimeout(timeout);
  }, [copiedKey]);

  const statusText: Record<LogStreamStatus, string> = {
    idle: 'Idle',
    streaming: 'Streaming…',
    reconnecting: 'Reconnecting…',
    complete: 'Stream closed',
    error: 'Disconnected',
  };

  const parameterEntries = useMemo(() => {
    if (!job) return [];
    return Object.entries(job.parameters ?? {});
  }, [job]);

  const resultEntries = useMemo(() => {
    if (!job) return [];
    return Object.entries(job.result ?? {});
  }, [job]);

  const metricEntries = useMemo(() => {
    if (!job) return [];
    return Object.entries(job.metrics ?? {});
  }, [job]);

  const handleCopy = async (value: string, key: string) => {
    if (!value) return;
    const success = await copyToClipboard(value);
    if (success) {
      setCopiedKey(key);
    }
  };

  if (!job) {
    return <div style={{ color: '#64748b' }}>Select a job to inspect its progress and logs.</div>;
  }

  const runtimeEntries = [
    { key: 'status', label: 'Status', value: job.status },
    { key: 'stage', label: 'Stage', value: job.stage ?? '—' },
    { key: 'created', label: 'Created', value: formatDistanceToNow(job.created_at) },
    { key: 'started', label: 'Started', value: formatDistanceToNow(job.started_at ?? undefined) },
    { key: 'finished', label: 'Finished', value: formatDistanceToNow(job.finished_at ?? undefined) },
    { key: 'workspace', label: 'Workspace', value: job.workspace, copy: true },
  ];

  const logFooterMessage = log ? undefined : 'Log output will appear here once the job starts.';
  const reconnectingCountdown = retryDelay ? `Retrying in ${Math.ceil(retryDelay / 1000)}s` : null;

  return (
    <div className="job-detail">
      <div className="detail-card">
        <h3>Runtime</h3>
        <div className="kv-grid">
          {runtimeEntries.map((entry) => (
            <div key={entry.key} className="kv-entry">
              <div className="kv-label">{entry.label}</div>
              <div className="kv-value">
                <span className="kv-text">{entry.value}</span>
                {entry.copy && typeof entry.value === 'string' && entry.value && (
                  <button
                    type="button"
                    className="copy-button"
                    onClick={() => handleCopy(entry.value as string, entry.key)}
                  >
                    {copiedKey === entry.key ? 'Copied' : 'Copy'}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="detail-card">
        <h3>Results</h3>
        {resultEntries.length > 0 ? (
          <div className="kv-stack">
            {resultEntries.map(([key, value]) => (
              <div key={key} className="kv-row">
                <div className="kv-label">{key}</div>
                <div className="kv-value">
                  {typeof value === 'string' ? (
                    <span className="kv-text">{value}</span>
                  ) : (
                    <pre className="kv-code">{JSON.stringify(value, null, 2)}</pre>
                  )}
                  {typeof value === 'string' && value && (
                    <button
                      type="button"
                      className="copy-button"
                      onClick={() => handleCopy(value, `result:${key}`)}
                    >
                      {copiedKey === `result:${key}` ? 'Copied' : 'Copy'}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <span className="kv-empty">No outputs recorded yet.</span>
        )}
      </div>

      <div className="detail-card">
        <h3>Metrics</h3>
        {metricEntries.length > 0 ? (
          <div className="kv-stack">
            {metricEntries.map(([key, value]) => (
              <div key={key} className="kv-row">
                <div className="kv-label">{key}</div>
                <div className="kv-value">
                  {typeof value === 'object' ? (
                    <pre className="kv-code">{JSON.stringify(value, null, 2)}</pre>
                  ) : (
                    <span className="kv-text">{String(value)}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <span className="kv-empty">No metrics reported yet.</span>
        )}
      </div>

      <div className="detail-card">
        <h3>Parameters</h3>
        {parameterEntries.length > 0 ? (
          <div className="kv-stack">
            {parameterEntries.map(([key, value]) => (
              <div key={key} className="kv-row">
                <div className="kv-label">{key}</div>
                <div className="kv-value">
                  {typeof value === 'string' ? (
                    <span className="kv-text">{value}</span>
                  ) : (
                    <pre className="kv-code">{JSON.stringify(value, null, 2)}</pre>
                  )}
                  {typeof value === 'string' && value && (
                    <button
                      type="button"
                      className="copy-button"
                      onClick={() => handleCopy(value, `param:${key}`)}
                    >
                      {copiedKey === `param:${key}` ? 'Copied' : 'Copy'}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <span className="kv-empty">No parameters recorded.</span>
        )}
      </div>

      <div className="detail-card">
        <div className="log-toolbar">
          <h3 style={{ margin: 0 }}>Live Log</h3>
          <div className="log-status">
            <span className={`log-status-label ${logStatus}`}>{statusText[logStatus]}</span>
            {reconnectingCountdown && <span className="log-status-hint">{reconnectingCountdown}</span>}
            {logError && <span className="log-status-error">{logError}</span>}
            <button type="button" className={`row-action ${autoScroll ? 'active' : ''}`} onClick={() => setAutoScroll((prev) => !prev)}>
              {autoScroll ? 'Auto-scroll on' : 'Auto-scroll off'}
            </button>
            <button type="button" className="row-action" onClick={refresh}>
              Refresh
            </button>
            {(logStatus === 'error' || logStatus === 'reconnecting') && (
              <button type="button" className="row-action" onClick={reconnect}>
                Reconnect
              </button>
            )}
          </div>
        </div>
        <div className="log-viewer" ref={logContainerRef}>
          {log || logFooterMessage}
        </div>
      </div>
    </div>
  );
}

export default JobDetail;
