import { JobInfo } from "../hooks/useJobs";
import { clsx } from "clsx";

interface Props {
  jobs: JobInfo[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const statusClass: Record<string, string> = {
  running: "status-pill running",
  queued: "status-pill running",
  succeeded: "status-pill succeeded",
  failed: "status-pill failed",
};

const statusLabel: Record<string, string> = {
  running: "Running",
  queued: "Queued",
  succeeded: "Completed",
  failed: "Failed",
};

export function JobTable({ jobs, selectedId, onSelect }: Props) {
  return (
    <div className="card">
      <div className="inline-controls" style={{ justifyContent: "space-between" }}>
        <h2>Activity</h2>
        <span className="muted">{jobs.length} jobs</span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table className="table">
          <thead>
            <tr>
              <th>Job</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr
                key={job.id}
                onClick={() => onSelect(job.id)}
                style={{ cursor: "pointer", background: selectedId === job.id ? "rgba(56,189,248,0.1)" : "transparent" }}
              >
                <td>
                  <strong>{job.jobType}</strong>
                  <div className="muted" style={{ fontSize: "0.75rem" }}>
                    {Object.entries(job.params)
                      .map(([key, value]) => `${key}: ${String(value)}`)
                      .join(", ")}
                  </div>
                </td>
                <td>
                  <span className={clsx("status-pill", statusClass[job.status] ?? "status-pill")}>{statusLabel[job.status] ?? job.status}</span>
                </td>
                <td style={{ width: "180px" }}>
                  <div className="progress-track">
                    <div className="progress-bar" style={{ width: `${Math.round(job.progress * 100)}%` }} />
                  </div>
                </td>
                <td>{new Date(job.createdAt * 1000).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default JobTable;
