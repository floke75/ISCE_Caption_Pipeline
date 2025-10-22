import { useEffect, useMemo, useState } from "react";
import type { MouseEvent } from "react";
import { cancelJob } from "../api";
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
  const [pendingCancelId, setPendingCancelId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

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

  const handleCancel = async (event: MouseEvent<HTMLButtonElement>, job: JobRecord) => {
    event.stopPropagation();
    if (job.status !== "pending") {
      setActionError("Only pending jobs can be cancelled.");
      return;
    }
    setActionError(null);
    setPendingCancelId(job.id);
    try {
      await cancelJob(job.id);
      onRefresh();
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setPendingCancelId(null);
    }
  };

  const handleCopyWorkspace = (event: MouseEvent<HTMLButtonElement>, job: JobRecord) => {
    event.stopPropagation();
    if (!job.workspacePath) {
      setActionError("Workspace path is not available yet.");
      return;
    }
    setActionError(null);
    copyToClipboard(job.workspacePath);
  };

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
          {actionError && <div className="form__message form__message--error">{actionError}</div>}
          {jobs.length === 0 ? (
            <p className="muted">No jobs yet. Launch an inference, training pair, or training run to populate history.</p>
          ) : (
            <table className="job-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Queue</th>
                  <th>Stage</th>
                  <th>Progress</th>
                  <th>Started</th>
                  <th>Duration</th>
                  <th>Actions</th>
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
                    <td>{job.queuePosition && job.status === "pending" ? `#${job.queuePosition}` : "—"}</td>
                    <td>{job.stage ?? "—"}</td>
                    <td>
                      <div className="progress">
                        <div className="progress__bar" style={{ width: `${Math.min(100, Math.max(0, job.progress * 100))}%` }} />
                      </div>
                      <span className="progress__value">{formatProgress(job.progress)}</span>
                    </td>
                    <td>{formatDate(job.startedAt)}</td>
                    <td>{formatDuration(job)}</td>
                    <td>
                      <div className="job-table__actions">
                        <button
                          type="button"
                          className="button button--tiny"
                          disabled={job.status !== "pending" || pendingCancelId === job.id}
                          onClick={(event) => handleCancel(event, job)}
                        >
                          {pendingCancelId === job.id ? "Cancelling…" : "Cancel"}
                        </button>
                        <button
                          type="button"
                          className="button button--tiny button--secondary"
                          onClick={(event) => handleCopyWorkspace(event, job)}
                          disabled={!job.workspacePath}
                        >
                          Copy path
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="job-monitor__details">
          {selectedJob ? (
            <div className="job-details">
              <header className="job-details__header">
                <div>
                  <h3>{selectedJob.type.replace(/_/g, " ")}</h3>
                  <p className="muted">
                    {selectedJob.startedAt ? `Started ${formatDate(selectedJob.startedAt)}` : "Start time pending"}
                    {selectedJob.finishedAt ? ` · Finished ${formatDate(selectedJob.finishedAt)}` : ""}
                  </p>
                </div>
                <div className="job-details__id">
                  <code>{selectedJob.id}</code>
                  <button
                    type="button"
                    className="button button--tiny button--secondary"
                    onClick={() => copyToClipboard(selectedJob.id)}
                  >
                    Copy ID
                  </button>
                </div>
              </header>
              <div className="job-details__grid">
                <section className="job-details__section job-details__section--summary">
                  <h4>Summary</h4>
                  <dl className="kv-list kv-list--details">
                    <div className="kv-list__row">
                      <dt>Status</dt>
                      <dd>
                        <span className={statusClass(selectedJob.status)}>{selectedJob.status}</span>
                      </dd>
                    </div>
                    <div className="kv-list__row">
                      <dt>Queue position</dt>
                      <dd>{selectedJob.queuePosition && selectedJob.status === "pending" ? `#${selectedJob.queuePosition}` : "—"}</dd>
                    </div>
                    <div className="kv-list__row">
                      <dt>Stage</dt>
                      <dd>{selectedJob.stage ?? "—"}</dd>
                    </div>
                    <div className="kv-list__row">
                      <dt>Started</dt>
                      <dd>{formatDate(selectedJob.startedAt)}</dd>
                    </div>
                    <div className="kv-list__row">
                      <dt>Duration</dt>
                      <dd>{formatDuration(selectedJob)}</dd>
                    </div>
                    <div className="kv-list__row">
                      <dt>Workspace</dt>
                      <dd>
                        {selectedJob.workspacePath ? (
                          <div className="job-details__value">
                            <code>{selectedJob.workspacePath}</code>
                            <button
                              type="button"
                              className="button button--tiny"
                              onClick={() => copyToClipboard(selectedJob.workspacePath ?? "")}
                            >
                              Copy path
                            </button>
                          </div>
                        ) : (
                          <span className="muted">Not assigned yet</span>
                        )}
                      </dd>
                    </div>
                  </dl>
                  <div className="job-details__progress">
                    <div className="progress">
                      <div
                        className="progress__bar"
                        style={{ width: `${Math.min(100, Math.max(0, selectedJob.progress * 100))}%` }}
                      />
                    </div>
                    <span className="progress__value">{formatProgress(selectedJob.progress)}</span>
                  </div>
                </section>
                <section className="job-details__section">
                  <h4>Parameters</h4>
                  <dl className="kv-list kv-list--details">
                    {Object.entries(selectedJob.params).map(([key, value]) => (
                      <div key={key} className="kv-list__row">
                        <dt>{key}</dt>
                        <dd>{String(value ?? "")}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              </div>
              {Object.keys(selectedJob.result).length > 0 && (
                <section className="job-details__section">
                  <h4>Result</h4>
                  <dl className="kv-list kv-list--details">
                    {Object.entries(selectedJob.result).map(([key, value]) => (
                      <div key={key} className="kv-list__row">
                        <dt>{key}</dt>
                        <dd>{String(value ?? "")}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              )}
              {selectedJob.message && (
                <section className="job-details__section">
                  <h4>Message</h4>
                  <p className="job-details__message">{selectedJob.message}</p>
                </section>
              )}
              <section className="job-details__section">
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
              </section>
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
