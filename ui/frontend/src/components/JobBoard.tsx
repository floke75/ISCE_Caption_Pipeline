import { ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import client from '../api/client';
import { useEventStream } from '../hooks/useEventStream';
import { useJobLog, useJobs } from '../hooks/useJobs';
import { JobRecord, JobStatus } from '../types';
import '../styles/jobs.css';

const STATUS_COLORS: Record<string, string> = {
  pending: '#fbbf24',
  running: '#38bdf8',
  succeeded: '#34d399',
  failed: '#f87171',
  cancelled: '#cbd5f5',
};

const PATH_KEY_HINTS = new Set([
  'workspace',
  'workspace_path',
  'workspace_artifact',
  'output_srt',
  'enriched_tokens',
  'asr_reference',
  'media_path',
  'transcript_path',
  'output_dir',
  'model_config_path',
]);

const VIEWABLE_EXTENSIONS = new Set(['.json', '.srt', '.txt', '.log', '.yaml', '.yml', '.csv', '.md']);

type DetailRow = {
  key: string;
  label: string;
  value: ReactNode;
  copyValue?: string;
};

function formatTimestamp(value: string) {
  const date = new Date(value);
  return date.toLocaleString();
}

function relativeTime(value: string) {
  const delta = Date.now() - new Date(value).getTime();
  if (Number.isNaN(delta)) return '';
  const minutes = Math.floor(delta / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function jobTitle(job: JobRecord) {
  switch (job.jobType) {
    case 'inference':
      return 'Inference run';
    case 'training_pair':
      return 'Training pair';
    case 'model_training':
      return 'Model training';
    default:
      return job.jobType;
  }
}

function formatLabel(key: string) {
  return key
    .split('_')
    .filter((part) => part.length)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function isPathField(key: string, value: unknown): value is string {
  if (typeof value !== 'string') {
    return false;
  }
  const lower = key.toLowerCase();
  if (lower.endsWith('_path') || lower.endsWith('_dir') || PATH_KEY_HINTS.has(lower)) {
    return true;
  }
  return value.includes('/') || value.includes('\\');
}

function isViewable(path: string): boolean {
  const lower = path.toLowerCase();
  return Array.from(VIEWABLE_EXTENSIONS).some(ext => lower.endsWith(ext));
}

function buildDetailRows(data?: Record<string, unknown> | null, jobId?: string): DetailRow[] {
  if (!data) {
    return [];
  }

  return Object.entries(data)
    .filter(([, value]) => value !== undefined)
    .map(([key, value]) => {
      const label = formatLabel(key);

      if (value === null) {
        return {
          key,
          label,
          value: <span className="detail-muted">—</span>,
        };
      }

      if (isPathField(key, value)) {
        const canView = isViewable(value);
        return {
          key,
          label,
          value: (
            <div className="flex items-center gap-2">
               <code className="path-value">{value}</code>
               {canView && jobId && (
                 <Link
                   to={`/artifacts/view?path=${encodeURIComponent(value)}`}
                   className="text-xs text-blue-400 hover:text-blue-300 underline ml-2"
                 >
                   View
                 </Link>
               )}
            </div>
          ),
          copyValue: value,
        };
      }

      if (typeof value === 'string') {
        return {
          key,
          label,
          value: <span>{value}</span>,
        };
      }

      if (typeof value === 'number' || typeof value === 'boolean') {
        return {
          key,
          label,
          value: <span>{String(value)}</span>,
        };
      }

      try {
        const serialized = JSON.stringify(value, null, 2);
        return {
          key,
          label,
          value: <pre className="detail-json">{serialized}</pre>,
          copyValue: serialized,
        };
      } catch {
        return {
          key,
          label,
          value: <span>{String(value)}</span>,
        };
      }
    });
}

function DetailList({
  rows,
  emptyMessage,
  onCopy,
}: {
  rows: DetailRow[];
  emptyMessage: string;
  onCopy: (value: string, label: string) => Promise<void> | void;
}) {
  if (!rows.length) {
    return <p className="detail-empty">{emptyMessage}</p>;
  }

  return (
    <dl className="detail-list">
      {rows.map((row) => (
        <div key={row.key} className="detail-row">
          <dt>{row.label}</dt>
          <dd>
            <div className="detail-value">
              <div className="detail-value-content">{row.value}</div>
              {row.copyValue ? (
                <button
                  type="button"
                  className="detail-copy"
                  onClick={async (event) => {
                    event.stopPropagation();
                    await onCopy(row.copyValue!, row.label);
                  }}
                >
                  Copy
                </button>
              ) : null}
            </div>
          </dd>
        </div>
      ))}
    </dl>
  );
}

function JobTypeIcon({ type }: { type: string }) {
  if (type === 'inference') {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ minWidth: 16 }}>
        <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
        <line x1="7" y1="2" x2="7" y2="22"></line>
        <line x1="17" y1="2" x2="17" y2="22"></line>
        <line x1="2" y1="12" x2="22" y2="12"></line>
        <line x1="2" y1="7" x2="7" y2="7"></line>
        <line x1="2" y1="17" x2="7" y2="17"></line>
        <line x1="17" y1="17" x2="22" y2="17"></line>
        <line x1="17" y1="7" x2="22" y2="7"></line>
      </svg>
    );
  }
  if (type === 'training_pair') {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ minWidth: 16 }}>
         <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
         <polyline points="14 2 14 8 20 8"></polyline>
         <line x1="16" y1="13" x2="8" y2="13"></line>
         <line x1="16" y1="17" x2="8" y2="17"></line>
         <polyline points="10 9 9 9 8 9"></polyline>
      </svg>
    );
  }
  if (type === 'model_training') {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ minWidth: 16 }}>
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
        <polyline points="17 6 23 6 23 12"></polyline>
      </svg>
    );
  }
  return null;
}

function StatusIcon({ status }: { status: string }) {
  const style = { minWidth: 12 };
  if (status === 'succeeded') {
    return (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={style}>
        <polyline points="20 6 9 17 4 12"></polyline>
      </svg>
    );
  }
  if (status === 'failed') {
    return (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={style}>
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
    );
  }
  if (status === 'running') {
     return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={style}>
           <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
           <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite"/>
        </svg>
     );
  }
  if (status === 'pending') {
     return (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={style}>
           <circle cx="12" cy="12" r="10"></circle>
           <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
     );
  }
  return null;
}

/**
 * A comprehensive component for monitoring and inspecting pipeline jobs.
 *
 * This component renders a list of all jobs on the left and a detailed view
 * of the selected job on the right. The detailed view includes parameters,
 * results, and a live-streaming log viewer that uses Server-Sent Events (SSE).
 *
 * @returns {JSX.Element} The rendered job board.
 */
export function JobBoard() {
  const jobsQuery = useJobs();
  // Memoize jobs list to prevent unstable dependency warning
  const jobs = useMemo(() => jobsQuery.data ?? [], [jobsQuery.data]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [actionJobId, setActionJobId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<'all' | JobStatus>('all');

  const orderedJobs = useMemo(() => {
    let filtered = jobs;
    if (statusFilter !== 'all') {
      filtered = jobs.filter(j => j.status === statusFilter);
    }
    return filtered.slice().sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }, [jobs, statusFilter]);

  useEffect(() => {
    if (!orderedJobs.length) {
      if (statusFilter === 'all') {
        setSelectedId(null);
      }
      return;
    }
    // Only select default if nothing is selected or current selection is gone
    if (!selectedId || !orderedJobs.some((job) => job.id === selectedId)) {
      setSelectedId(orderedJobs[0].id);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [orderedJobs]);

  const selectedJob = jobs.find((job) => job.id === selectedId) ?? null;
  const [logText, setLogText] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const [completedStatus, setCompletedStatus] = useState<string | null>(null);
  const logViewRef = useRef<HTMLPreElement | null>(null);
  const streamResetRef = useRef(true);

  useEffect(() => {
    setLogText('');
    setAutoScroll(true);
    setCompletedStatus(null);
    streamResetRef.current = true;
  }, [selectedJob?.id]);

  const streamUrl = selectedJob ? `/api/jobs/${selectedJob.id}/logs/stream` : null;

  const stream = useEventStream(streamUrl, {
    enabled: Boolean(selectedJob),
    eventTypes: ['complete'],
    onOpen: () => {
      setCompletedStatus(null);
      streamResetRef.current = true;
    },
    onMessage: (event) => {
      if (!event.data) return;
      const chunk = `${event.data}\n`;
      setLogText((prev) => {
        const next = streamResetRef.current ? chunk : prev ? prev + chunk : chunk;
        streamResetRef.current = false;
        return next;
      });
    },
    onEvent: (type, event) => {
      if (type === 'complete') {
        setCompletedStatus(event.data);
      }
    },
  });

  const { status: streamStatus, supported: streamSupported, disconnect: closeStream } = stream;

  useEffect(() => {
    if (completedStatus) {
      closeStream();
    }
  }, [closeStream, completedStatus]);

  const shouldPoll = !streamSupported || streamStatus === 'error';
  const { data: logData } = useJobLog(selectedJob?.id ?? null, {
    enabled: Boolean(selectedJob),
    refetchInterval: shouldPoll ? 4000 : false,
  });

  useEffect(() => {
    if (!selectedJob || !logData?.log) {
      return;
    }
    if (shouldPoll || !logText) {
      setLogText(logData.log);
    }
  }, [logData?.log, logText, selectedJob, shouldPoll]);

  useEffect(() => {
    if (!autoScroll || !logViewRef.current) {
      return;
    }
    logViewRef.current.scrollTop = logViewRef.current.scrollHeight;
  }, [autoScroll, logText]);

  const streamStatusLabel = useMemo(() => {
    if (!selectedJob) return '';
    if (!streamSupported) return 'Polling logs (SSE not supported)';
    if (streamStatus === 'connecting') return 'Connecting to log stream…';
    if (streamStatus === 'open') return 'Streaming live logs';
    if (streamStatus === 'error') return 'Stream interrupted — retrying, polling enabled';
    if (streamStatus === 'closed') {
      return completedStatus ? `Stream ended (${completedStatus})` : 'Stream closed';
    }
    return '';
  }, [completedStatus, selectedJob, streamStatus, streamSupported]);

  const copyText = async (value: string, label: string) => {
    if (!value) {
      toast.error('Nothing to copy');
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      toast.success(`${label} copied`);
    } catch {
      toast.error('Clipboard copy failed');
    }
  };

  const copyJson = async (payload: unknown, label: string) => {
    try {
      const serialized = JSON.stringify(payload ?? {}, null, 2);
      await copyText(serialized, label);
    } catch {
      toast.error('Clipboard copy failed');
    }
  };

  const copyLog = async () => {
    if (!logText) {
      toast.error('No log output yet');
      return;
    }
    await copyText(logText, 'Log');
  };

  const toggleAutoScroll = () => {
    setAutoScroll((value) => !value);
  };

  const handleCancel = async (jobId: string) => {
    setActionJobId(jobId);
    try {
      await client.post(`/jobs/${jobId}/cancel`);
      toast.success('Cancellation requested');
      await jobsQuery.refetch();
    } catch (error) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const detail = (error as any)?.response?.data?.detail;
      toast.error(detail ?? 'Failed to cancel job');
    } finally {
      setActionJobId(null);
    }
  };

  const parameterRows = useMemo(() => buildDetailRows(selectedJob?.params as Record<string, unknown> ?? null, selectedJob?.id), [selectedJob]);
  const resultRows = useMemo(() => buildDetailRows(selectedJob?.result as Record<string, unknown> ?? null, selectedJob?.id), [selectedJob]);
  const detailRows = useMemo(() => {
    if (!selectedJob) {
      return [];
    }
    return [
      {
        key: 'jobId',
        label: 'Job ID',
        value: <code className="path-value">{selectedJob.id}</code>,
        copyValue: selectedJob.id,
      },
      {
        key: 'workspace',
        label: 'Workspace',
        value: <code className="path-value">{selectedJob.workspacePath}</code>,
        copyValue: selectedJob.workspacePath,
      },
      {
        key: 'workflow',
        label: 'Workflow',
        value: <span>{jobTitle(selectedJob)}</span>,
      },
      {
        key: 'status',
        label: 'Status',
        value: <span className="status-pill-inline" style={{ color: STATUS_COLORS[selectedJob.status] ?? '#94a3b8' }}>{selectedJob.status}</span>,
      },
      {
        key: 'message',
        label: 'Message',
        value: <span>{selectedJob.message || '—'}</span>,
      },
    ];
  }, [selectedJob]);

  return (
    <div className="job-board">
      <header>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
          <h2>Job monitor</h2>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.85rem' }}>{orderedJobs.length} job(s)</p>
        </div>
        <div className="job-filters">
           <select
             value={statusFilter}
             onChange={(e) => setStatusFilter(e.target.value as JobStatus | 'all')}
             className="status-select"
             style={{
               background: 'rgba(15, 23, 42, 0.5)',
               border: '1px solid rgba(148, 163, 184, 0.2)',
               color: '#e2e8f0',
               padding: '0.3rem',
               borderRadius: '6px',
               fontSize: '0.8rem',
               cursor: 'pointer'
             }}
            >
             <option value="all">All</option>
             <option value="running">Running</option>
             <option value="pending">Pending</option>
             <option value="succeeded">Succeeded</option>
             <option value="failed">Failed</option>
           </select>
        </div>
      </header>
      {orderedJobs.length === 0 ? (
        <div className="empty-state">No jobs found.</div>
      ) : (
        <>
          <div className="job-list">
            {orderedJobs.map((job) => {
              const percent = Math.round((job.progress ?? 0) * 100);
              const canCancel = job.status === 'pending' || job.status === 'running';
              const isCancelling = actionJobId === job.id;
              return (
                <div
                  key={job.id}
                  className={clsx('job-row', { active: job.id === selectedId })}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedId(job.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setSelectedId(job.id);
                    }
                  }}
                >
                  <div className="job-row-header">
                    <div className="job-row-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <JobTypeIcon type={job.jobType} />
                      <strong>{jobTitle(job)}</strong>
                      <span
                        className="status-pill"
                        style={{
                           background: `${STATUS_COLORS[job.status] ?? '#94a3b8'}22`,
                           color: STATUS_COLORS[job.status] ?? '#94a3b8',
                           display: 'flex',
                           alignItems: 'center',
                           gap: '0.3rem',
                           paddingLeft: '0.4rem',
                           paddingRight: '0.5rem'
                        }}
                      >
                        <StatusIcon status={job.status} />
                        {job.status}
                      </span>
                    </div>
                    <div className="job-row-actions">
                      <button
                        type="button"
                        className="job-row-action"
                        onClick={async (event) => {
                          event.stopPropagation();
                          await copyText(job.workspacePath, 'Workspace path');
                        }}
                      >
                        Copy workspace
                      </button>
                      {canCancel ? (
                        <button
                          type="button"
                          className="job-row-action cancel-action"
                          disabled={isCancelling}
                          onClick={async (event) => {
                            event.stopPropagation();
                            await handleCancel(job.id);
                          }}
                        >
                          {isCancelling ? 'Cancelling…' : 'Cancel'}
                        </button>
                      ) : null}
                    </div>
                  </div>
                  <div className="job-row-meta">
                    <span>{job.message || '—'}</span>
                    <span title={formatTimestamp(job.updatedAt)} style={{ cursor: 'help' }}>{relativeTime(job.updatedAt)}</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-bar" style={{ width: `${percent}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
          {selectedJob ? (
            <div className="job-details">
              <div className="detail-card">
                <div className="detail-card-header">
                  <div>
                    <h3>Details</h3>
                    <p className="detail-subtext">
                      Started {formatTimestamp(selectedJob.createdAt)} · Last update {relativeTime(selectedJob.updatedAt)}
                    </p>
                  </div>
                  <div className="detail-header-actions">
                    <button
                      type="button"
                      className="copy-button"
                      onClick={() => copyText(selectedJob.workspacePath, 'Workspace path')}
                    >
                      Copy workspace
                    </button>
                    <button
                      type="button"
                      className="copy-button"
                      onClick={() => copyJson(selectedJob.params, 'Parameters JSON')}
                    >
                      Copy params JSON
                    </button>
                  </div>
                </div>
                <DetailList rows={detailRows} emptyMessage="No metadata recorded" onCopy={copyText} />
              </div>
              <div className="detail-card">
                <div className="detail-card-header">
                  <h3>Runtime parameters</h3>
                  <button type="button" className="copy-button" onClick={() => copyJson(selectedJob.params, 'Parameters JSON')}>
                    Copy all
                  </button>
                </div>
                <DetailList rows={parameterRows} emptyMessage="No parameters supplied" onCopy={copyText} />
              </div>
              {selectedJob.result ? (
                <div className="detail-card">
                  <div className="detail-card-header">
                    <h3>Results</h3>
                    <div className="detail-header-actions">
                      {selectedJob.jobType === 'training_pair' && !!selectedJob.result['training_json'] && !!selectedJob.result['asr_reference'] && (
                        <Link
                          to={`/jobs/alignment?train=${encodeURIComponent(selectedJob.result['training_json'] as string)}&asr=${encodeURIComponent(selectedJob.result['asr_reference'] as string)}`}
                          className="action-button primary"
                          style={{ fontSize: '0.8rem', padding: '0.2rem 0.6rem', textDecoration: 'none' }}
                        >
                          Visualise Alignment
                        </Link>
                      )}
                      <button type="button" className="copy-button" onClick={() => copyJson(selectedJob.result, 'Result payload')}>
                        Copy all
                      </button>
                    </div>
                  </div>
                  <DetailList rows={resultRows} emptyMessage="No result payload" onCopy={copyText} />
                </div>
              ) : null}
              <div className="detail-card">
                <div className="log-card-header">
                  <div>
                    <h3>Logs</h3>
                    {streamStatusLabel ? <p className="log-status">{streamStatusLabel}</p> : null}
                  </div>
                  <div className="log-controls">
                    <button type="button" className="copy-button" onClick={toggleAutoScroll}>
                      {autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}
                    </button>
                    <button type="button" className="copy-button" onClick={copyLog}>
                      Copy log
                    </button>
                  </div>
                </div>
                <pre ref={logViewRef} className="log-viewer">
                  {logText || 'Waiting for log output…'}
                </pre>
              </div>
              {selectedJob.error ? (
                <div className="detail-card" style={{ borderColor: 'rgba(248, 113, 113, 0.4)' }}>
                  <div className="detail-card-header">
                    <h3>Error</h3>
                    <button type="button" className="copy-button" onClick={() => copyText(selectedJob.error ?? '', 'Error message')}>
                      Copy error
                    </button>
                  </div>
                  <pre>{selectedJob.error}</pre>
                </div>
              ) : null}
            </div>
          ) : null}
        </>
      )}
    </div>
  );
}
