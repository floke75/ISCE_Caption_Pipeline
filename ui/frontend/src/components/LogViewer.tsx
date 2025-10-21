import { useJobLogs } from "../api";

interface Props {
  jobId: string | null;
}

export function LogViewer({ jobId }: Props) {
  const { data, isLoading, error } = useJobLogs(jobId);

  if (!jobId) {
    return <div className="text-sm text-slate-500">Select a job to inspect live logs.</div>;
  }

  if (isLoading) {
    return <div className="text-sm text-slate-400">Loading logs…</div>;
  }

  if (error) {
    return <div className="text-sm text-rose-300">Failed to load logs: {(error as Error).message}</div>;
  }

  return (
    <pre className="h-96 overflow-auto rounded-lg bg-slate-950/80 p-4 text-xs text-slate-200">
      {data?.lines?.length ? data.lines.join("") : "No log output yet."}
    </pre>
  );
}
