import { useQuery } from "@tanstack/react-query";

async function fetchJobLog(jobId: string): Promise<string> {
  const response = await fetch(`/api/jobs/${jobId}/log?tail=2000`);
  if (!response.ok) {
    throw new Error("Failed to load job log");
  }
  const data = await response.json();
  return data.log ?? "";
}

export function useJobLog(jobId: string | null) {
  return useQuery({
    queryKey: ["job-log", jobId],
    queryFn: () => fetchJobLog(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: 3000,
  });
}
