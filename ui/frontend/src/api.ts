import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

const API_BASE = "/api";

type HttpMethod = "GET" | "POST" | "PUT";

async function request<T>(path: string, method: HttpMethod = "GET", body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function useJobs() {
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => request<JobListResponse>("/jobs"),
    refetchInterval: 2500,
  });
}

export function useConfig(kind: "pipeline" | "core") {
  return useQuery({
    queryKey: ["config", kind],
    queryFn: () => request<ConfigEnvelope>(`/config/${kind}`),
    refetchInterval: false,
  });
}

export function useUpdateConfig(kind: "pipeline" | "core") {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (content: Record<string, unknown>) =>
      request<ConfigWriteResult>(`/config/${kind}`, "PUT", { content }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["config", kind] }),
  });
}

export function useLaunchInference() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: InferenceRequest) => request<JobResponse>("/jobs/inference", "POST", payload),
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useLaunchTrainingPair() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: TrainingPairRequest) =>
      request<JobResponse>("/jobs/training-pair", "POST", payload),
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useLaunchModelTraining() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: TrainModelRequest) => request<JobResponse>("/jobs/model-training", "POST", payload),
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

export function useJobLogs(jobId: string | null) {
  return useQuery({
    enabled: Boolean(jobId),
    queryKey: ["job-logs", jobId],
    queryFn: async () => {
      if (!jobId) return null;
      return await request<JobLogsResponse>(`/jobs/${jobId}/logs`);
    },
    refetchInterval: 2000,
  });
}

export interface ConfigEnvelope {
  defaults: Record<string, unknown>;
  overrides: Record<string, unknown>;
  resolved: Record<string, unknown>;
}

export interface ConfigWriteResult {
  path: string;
  updated: boolean;
}

export interface JobResponse {
  id: string;
  job_type: string;
  status: string;
  created_at: number;
  started_at?: number;
  finished_at?: number;
  metadata: Record<string, unknown>;
  error?: string | null;
}

export interface JobListResponse {
  jobs: JobResponse[];
}

export interface JobLogsResponse {
  lines: string[];
  next_index: number;
  total: number;
}

export interface InferenceRequest {
  media_path: string;
  transcript_path?: string | null;
  pipeline_overrides?: Record<string, unknown> | null;
}

export interface TrainingPairRequest {
  media_path: string;
  srt_path: string;
  pipeline_overrides?: Record<string, unknown> | null;
}

export interface TrainModelRequest {
  corpus_dir: string;
  constraints_path: string;
  weights_path: string;
  iterations: number;
  error_boost_factor: number;
  config_path: string;
}
