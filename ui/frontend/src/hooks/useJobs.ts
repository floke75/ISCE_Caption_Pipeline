import { useQuery } from "@tanstack/react-query";

export interface JobInfo {
  id: string;
  jobType: string;
  params: Record<string, unknown>;
  createdAt: number;
  status: string;
  startedAt?: number;
  finishedAt?: number;
  progress: number;
  message?: string;
  extra: Record<string, unknown>;
}

async function fetchJobs(): Promise<JobInfo[]> {
  const response = await fetch("/api/jobs");
  if (!response.ok) {
    throw new Error("Failed to fetch jobs");
  }
  return response.json();
}

export function useJobs(pollInterval = 4000) {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    refetchInterval: pollInterval,
    staleTime: 0,
  });
}
