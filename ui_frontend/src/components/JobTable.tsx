import { JobSummary } from '../lib/api';
import { formatDistanceToNow } from '../lib/time';
import clsx from 'clsx';

interface JobTableProps {
  jobs: JobSummary[];
  selectedJobId?: string;
  onSelect(jobId: string): void;
  onCopyWorkspace(job: JobSummary): void;
  onCancel(job: JobSummary): void;
  pendingCancelId?: string;
}

const statusLabel: Record<string, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
};

const cancellableStatuses = new Set(['pending', 'running']);

export function JobTable({
  jobs,
  selectedJobId,
  onSelect,
  onCopyWorkspace,
  onCancel,
  pendingCancelId,
}: JobTableProps) {
  return (
    <table className="job-table">
      <thead>
        <tr>
          <th>Job</th>
          <th>Status</th>
          <th>Progress</th>
          <th>Updated</th>
          <th aria-label="Actions" />
        </tr>
      </thead>
      <tbody>
        {jobs.map((job) => (
          <tr
            key={job.id}
            className={clsx({ selected: job.id === selectedJobId })}
            onClick={() => onSelect(job.id)}
          >
            <td>
              <div style={{ fontWeight: 600 }}>{job.name}</div>
              <div style={{ fontSize: '0.8rem', color: '#64748b' }}>{job.job_type}</div>
            </td>
            <td>
              <span className={clsx('status-pill', job.status)}>
                {statusLabel[job.status] ?? job.status}
              </span>
            </td>
            <td style={{ width: '160px' }}>
              <div className="progress-bar">
                <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
              </div>
            </td>
            <td style={{ fontSize: '0.8rem', color: '#64748b' }}>
              {job.finished_at
                ? formatDistanceToNow(job.finished_at)
                : job.started_at
                ? formatDistanceToNow(job.started_at)
                : formatDistanceToNow(job.created_at)}
            </td>
            <td className="job-actions">
              <button
                type="button"
                className="row-action"
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(job.id);
                }}
                title="Inspect job"
              >
                Inspect
              </button>
              <button
                type="button"
                className="row-action"
                onClick={(event) => {
                  event.stopPropagation();
                  onCopyWorkspace(job);
                }}
                title="Copy workspace path"
              >
                Copy path
              </button>
              <button
                type="button"
                className="row-action destructive"
                disabled={
                  !cancellableStatuses.has(job.status) || job.cancel_requested || pendingCancelId === job.id
                }
                onClick={(event) => {
                  event.stopPropagation();
                  onCancel(job);
                }}
                title={job.cancel_requested ? 'Cancellation requested' : 'Cancel job'}
              >
                {job.cancel_requested || pendingCancelId === job.id ? 'Cancelling…' : 'Cancel'}
              </button>
            </td>
          </tr>
        ))}
        {jobs.length === 0 && (
          <tr>
            <td colSpan={4} style={{ padding: '1rem 0', color: '#64748b' }}>
              No jobs yet. Launch a workflow to get started.
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

export default JobTable;
