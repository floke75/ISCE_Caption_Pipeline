import { JobResponse } from "../api";
import { formatDistanceToNow } from "date-fns";

interface Props {
  jobs: JobResponse[];
  onSelect: (job: JobResponse) => void;
  selectedJobId?: string | null;
}

const statusStyles: Record<string, string> = {
  queued: "bg-amber-500/20 text-amber-300",
  running: "bg-sky-500/20 text-sky-300",
  succeeded: "bg-emerald-500/20 text-emerald-300",
  failed: "bg-rose-500/20 text-rose-300",
};

export function JobTable({ jobs, onSelect, selectedJobId }: Props) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-800">
      <table className="min-w-full divide-y divide-slate-800 text-sm">
        <thead className="bg-slate-900/80 text-left uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-3">Job</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3">Started</th>
            <th className="px-4 py-3">Details</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {jobs.map((job) => {
            const statusClass = statusStyles[job.status] ?? "bg-slate-500/20 text-slate-300";
            const started = job.started_at ? formatDistanceToNow(job.started_at * 1000, { addSuffix: true }) : "Pending";
            const isSelected = selectedJobId === job.id;
            return (
              <tr
                key={job.id}
                onClick={() => onSelect(job)}
                className={`cursor-pointer bg-slate-900/40 transition hover:bg-slate-800/60 ${
                  isSelected ? "ring-2 ring-brand-400" : ""
                }`}
              >
                <td className="px-4 py-3 font-medium text-slate-100">{job.job_type.replace("_", " ")}</td>
                <td className="px-4 py-3">
                  <span className={`rounded-full px-2 py-1 text-xs font-semibold ${statusClass}`}>{job.status}</span>
                </td>
                <td className="px-4 py-3 text-slate-300">{started}</td>
                <td className="px-4 py-3 text-xs text-slate-400">
                  {Object.entries(job.metadata || {})
                    .filter(([, value]) => value !== undefined && value !== null && value !== "")
                    .map(([key, value]) => {
                      const rendered =
                        value && typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : String(value);
                      return (
                        <div key={key} className="truncate" title={rendered}>
                          <span className="font-semibold uppercase text-slate-500">{key}: </span>
                          <span className="text-slate-300">{rendered}</span>
                        </div>
                      );
                    })}
                  {job.error && <div className="text-rose-300">Error: {job.error}</div>}
                </td>
              </tr>
            );
          })}
          {jobs.length === 0 && (
            <tr>
              <td colSpan={4} className="px-4 py-6 text-center text-slate-500">
                No jobs launched yet. Submit a run to get started.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
