import { useMemo } from 'react';
import { JobDetail as JobDetailType } from '../lib/api';
import { LogStreamStatus, useJobLogs } from '../hooks/useJobLogs';
import { formatDistanceToNow } from '../lib/time';

interface JobDetailProps {
  job?: JobDetailType;
}

export function JobDetail({ job }: JobDetailProps) {
  const { log, status: logStatus, error: logError, reconnect } = useJobLogs(job?.id);

  const statusText: Record<LogStreamStatus, string> = {
    idle: 'Idle',
    streaming: 'Streaming…',
    complete: 'Stream closed',
    error: 'Disconnected',
  };

  const parameterText = useMemo(() => {
    if (!job) return '';
    return JSON.stringify(job.parameters, null, 2);
  }, [job]);

  if (!job) {
    return <div style={{ color: '#64748b' }}>Select a job to inspect its progress and logs.</div>;
  }

  return (
    <div className="job-detail">
      <div className="detail-card">
        <h3>Runtime</h3>
        <div className="kv-grid">
          <div>
            <strong>Status</strong>
            <span>{job.status}</span>
          </div>
          <div>
            <strong>Stage</strong>
            <span>{job.stage ?? '—'}</span>
          </div>
          <div>
            <strong>Created</strong>
            <span>{formatDistanceToNow(job.created_at)}</span>
          </div>
          <div>
            <strong>Started</strong>
            <span>{formatDistanceToNow(job.started_at ?? undefined)}</span>
          </div>
          <div>
            <strong>Finished</strong>
            <span>{formatDistanceToNow(job.finished_at ?? undefined)}</span>
          </div>
          <div>
            <strong>Workspace</strong>
            <span>{job.workspace}</span>
          </div>
        </div>
      </div>

      <div className="detail-card">
        <h3>Results</h3>
        <div className="kv-grid">
          {job.result &&
            Object.entries(job.result).map(([key, value]) => (
              <div key={key}>
                <strong>{key}</strong>
                <span>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
              </div>
            ))}
          {(!job.result || Object.keys(job.result).length === 0) && <span>No outputs recorded yet.</span>}
        </div>
      </div>

      <div className="detail-card">
        <h3>Metrics</h3>
        <div className="kv-grid">
          {Object.entries(job.metrics || {}).map(([key, value]) => (
            <div key={key}>
              <strong>{key}</strong>
              <span>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
            </div>
          ))}
          {(!job.metrics || Object.keys(job.metrics).length === 0) && <span>No metrics reported yet.</span>}
        </div>
      </div>

      <div className="detail-card">
        <h3>Parameters</h3>
        <pre className="parameters-json">{parameterText}</pre>
      </div>

      <div className="detail-card">
        <div className="log-toolbar">
          <h3 style={{ margin: 0 }}>Live Log</h3>
          <div className="log-status">
            <span className={`log-status-label ${logStatus}`}>{statusText[logStatus]}</span>
            {logError && <span className="log-status-error">{logError}</span>}
            {logStatus === 'error' && (
              <button type="button" onClick={reconnect}>
                Reconnect
              </button>
            )}
          </div>
        </div>
        <div className="log-viewer">{log || 'Log output will appear here once the job starts.'}</div>
      </div>
    </div>
  );
}

export default JobDetail;
