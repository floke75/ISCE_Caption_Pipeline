import { useEffect, useMemo, useState } from "react";
import { JobRecord } from "../types";
import Card from "./Card";
import LogViewer from "./LogViewer";

interface JobMonitorProps {
  jobs: JobRecord[];
  onRefresh: () => void;
}

function formatDate(value: string | null): string {
  if (!value) {
    return "—";
  }
  return new Date(value).toLocaleString();
}

function formatProgress(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function formatDuration(job: JobRecord): string {
  if (!job.startedAt) {
    return "—";
  }
  const end = job.finishedAt ? new Date(job.finishedAt).getTime() : Date.now();
  const start = new Date(job.startedAt).getTime();
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder}s`;
}

function statusClass(status: string): string {
  switch (status) {
    case "completed":
      return "status status--success";
    case "failed":
      return "status status--danger";
    case "running":
      return "status status--warning";
    default:
      return "status";
  }
}

function copyToClipboard(text: string) {
  void navigator.clipboard.writeText(text);
}

export function JobMonitor({ jobs, onRefresh }: JobMonitorProps) {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  useEffect(() => {
    if (!jobs.length) {
      setSelectedJobId(null);
      return;
    }
    if (!selectedJobId || !jobs.some((job) => job.id === selectedJobId)) {
      setSelectedJobId(jobs[0].id);
    }
  }, [jobs, selectedJobId]);

  const selectedJob = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);

  return (
    <Card
      title="Job monitor"
      description="Track active jobs, inspect parameters, and follow live logs."
      actions={
        <button type="button" className="button button--secondary" onClick={onRefresh}>
          Refresh
        </button>
      }
    >
      <div className="job-monitor">
        <div className="job-monitor__list">
          {jobs.length === 0 ? (
            <p className="muted">No jobs yet. Launch an inference, training pair, or training run to populate history.</p>
          ) : (
            <table className="job-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Stage</th>
                  <th>Progress</th>
                  <th>Started</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr
                    key={job.id}
                    className={job.id === selectedJobId ? "job-table__row job-table__row--active" : "job-table__row"}
                    onClick={() => setSelectedJobId(job.id)}
                  >
                    <td>{job.id.slice(0, 8)}</td>
                    <td>{job.type.replace(/_/g, " ")}</td>
                    <td>
                      <span className={statusClass(job.status)}>{job.status}</span>
                    </td>
                    <td>{job.stage ?? "—"}</td>
                    <td>
                      <div className="progress">
                        <div className="progress__bar" style={{ width: `${Math.min(100, Math.max(0, job.progress * 100))}%` }} />
                      </div>
                      <span className="progress__value">{formatProgress(job.progress)}</span>
                    </td>
                    <td>{formatDate(job.startedAt)}</td>
                    <td>{formatDuration(job)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="job-monitor__details">
          {selectedJob ? (
            <div className="job-details">
              <h3>Job {selectedJob.id}</h3>
              <div className="job-details__grid">
                <div>
                  <h4>Status</h4>
                  <p>
                    <span className={statusClass(selectedJob.status)}>{selectedJob.status}</span>
                  </p>
                  <h4>Stage</h4>
                  <p>{selectedJob.stage ?? "—"}</p>
                  {selectedJob.message && (
                    <>
                      <h4>Message</h4>
                      <p>{selectedJob.message}</p>
                    </>
                  )}
                  <h4>Artifacts</h4>
                  {selectedJob.artifacts.length === 0 ? (
                    <p className="muted">No artifacts yet.</p>
                  ) : (
                    <ul className="artifact-list">
                      {selectedJob.artifacts.map((artifact) => (
                        <li key={`${artifact.name}-${artifact.path}`}>
                          <span>{artifact.name}</span>
                          <code>{artifact.path}</code>
                          <button
                            type="button"
                            className="button button--tiny"
                            onClick={() => copyToClipboard(artifact.path)}
                          >
                            Copy
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                <div>
                  <h4>Parameters</h4>
                  <dl className="kv-list">
                    {Object.entries(selectedJob.params).map(([key, value]) => (
                      <div key={key} className="kv-list__row">
                        <dt>{key}</dt>
                        <dd>{String(value ?? "")}</dd>
                      </div>
                    ))}
                  </dl>
                  {Object.keys(selectedJob.result).length > 0 && (
                    <>
                      <h4>Result</h4>
                      <dl className="kv-list">
                        {Object.entries(selectedJob.result).map(([key, value]) => (
                          <div key={key} className="kv-list__row">
                            <dt>{key}</dt>
                            <dd>{String(value ?? "")}</dd>
                          </div>
                        ))}
                      </dl>
                    </>
                  )}
                </div>
              </div>
              <LogViewer jobId={selectedJob.id} status={selectedJob.status} />
            </div>
          ) : (
            <p className="muted">Select a job to view details and logs.</p>
          )}
        </div>
      </div>
    </Card>
  );
}

export default JobMonitor;
