import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { useJobLog, useJobs } from '../hooks/useJobs';
import { JobRecord } from '../types';
import '../styles/jobs.css';

const STATUS_COLORS: Record<string, string> = {
  pending: '#fbbf24',
  running: '#38bdf8',
  succeeded: '#34d399',
  failed: '#f87171',
  cancelled: '#cbd5f5',
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

export function JobBoard() {
  const { data: jobs } = useJobs();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const orderedJobs = useMemo(() => {
    return (jobs ?? []).slice().sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }, [jobs]);

  useEffect(() => {
    if (!orderedJobs.length) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !orderedJobs.some((job) => job.id === selectedId)) {
      setSelectedId(orderedJobs[0].id);
    }
  }, [orderedJobs, selectedId]);

  const selectedJob = orderedJobs.find((job) => job.id === selectedId) ?? null;
  const { data: logData } = useJobLog(selectedJob?.id ?? null);

  const copyLog = async () => {
    if (!logData?.log) return;
    try {
      await navigator.clipboard.writeText(logData.log);
      toast.success('Log copied to clipboard');
    } catch (error) {
      toast.error('Clipboard copy failed');
    }
  };

  const copyJson = async (payload: unknown, label: string) => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      toast.success(`${label} copied`);
    } catch (error) {
      toast.error('Clipboard copy failed');
    }
  };

  return (
    <div className="job-board">
      <header>
        <div>
          <h2>Job monitor</h2>
          <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.85rem' }}>{orderedJobs.length} job(s) tracked</p>
        </div>
      </header>
      {orderedJobs.length === 0 ? (
        <div className="empty-state">No jobs yet. Launch a workflow to see progress here.</div>
      ) : (
        <>
          <div className="job-list">
            {orderedJobs.map((job) => {
              const percent = Math.round((job.progress ?? 0) * 100);
              return (
                <button
                  type="button"
                  key={job.id}
                  className={clsx('job-row', { active: job.id === selectedId })}
                  onClick={() => setSelectedId(job.id)}
                >
                  <div className="job-row-title">
                    <strong>{jobTitle(job)}</strong>
                    <span
                      className="status-pill"
                      style={{ background: `${STATUS_COLORS[job.status] ?? '#94a3b8'}22`, color: STATUS_COLORS[job.status] ?? '#94a3b8' }}
                    >
                      {job.status}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8' }}>
                    <span>{job.message || '—'}</span>
                    <span>{relativeTime(job.updatedAt)}</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-bar" style={{ width: `${percent}%` }} />
                  </div>
                </button>
              );
            })}
          </div>
          {selectedJob ? (
            <div className="job-details">
              <div className="detail-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <h3>Details</h3>
                    <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.8rem' }}>
                      Started {formatTimestamp(selectedJob.createdAt)} · Last update {relativeTime(selectedJob.updatedAt)}
                    </p>
                  </div>
                  <button type="button" className="copy-button" onClick={() => copyJson(selectedJob.params, 'Parameters')}>
                    Copy params
                  </button>
                </div>
                <pre>{JSON.stringify(selectedJob.params, null, 2)}</pre>
              </div>
              {selectedJob.result ? (
                <div className="detail-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3>Results</h3>
                    <button type="button" className="copy-button" onClick={() => copyJson(selectedJob.result, 'Result payload')}>
                      Copy results
                    </button>
                  </div>
                  <pre>{JSON.stringify(selectedJob.result, null, 2)}</pre>
                </div>
              ) : null}
              <div className="detail-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3>Logs</h3>
                  <button type="button" className="copy-button" onClick={copyLog}>
                    Copy log
                  </button>
                </div>
                <pre>{logData?.log ?? 'Loading logs…'}</pre>
              </div>
              {selectedJob.error ? (
                <div className="detail-card" style={{ borderColor: 'rgba(248, 113, 113, 0.4)' }}>
                  <h3>Error</h3>
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
