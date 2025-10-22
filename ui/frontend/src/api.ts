import { ConfigMap, ConfigResponse, JobRecord, LogChunk } from "./types";

const API_BASE = "/api";

function transformJob(data: any): JobRecord {
  return {
    id: data.id,
    type: data.type,
    status: data.status,
    createdAt: data.created_at,
    startedAt: data.started_at ?? null,
    finishedAt: data.finished_at ?? null,
    progress: typeof data.progress === "number" ? data.progress : 0,
    stage: data.stage ?? null,
    message: data.message ?? null,
    artifacts: Array.isArray(data.artifacts) ? data.artifacts : [],
    params: data.params ?? {},
    result: data.result ?? {}
  };
}

export async function fetchJobs(): Promise<JobRecord[]> {
  const res = await fetch(`${API_BASE}/jobs`);
  if (!res.ok) {
    throw new Error(`Failed to load jobs (${res.status})`);
  }
  const payload = await res.json();
  return payload.map(transformJob);
}

export async function fetchConfig(kind: "pipeline" | "model"): Promise<ConfigResponse> {
  const res = await fetch(`${API_BASE}/config/${kind}`);
  if (!res.ok) {
    throw new Error(`Failed to load ${kind} config`);
  }
  return res.json();
}

export async function updateConfig(kind: "pipeline" | "model", config: ConfigMap): Promise<ConfigResponse> {
  const res = await fetch(`${API_BASE}/config/${kind}` , {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config })
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Failed to save ${kind} config`);
  }
  return res.json();
}

export interface InferencePayload {
  mediaPath: string;
  transcriptPath: string;
  outputPath?: string | null;
  outputBasename?: string | null;
  pipelineOverrides?: ConfigMap | null;
  modelOverrides?: ConfigMap | null;
}

export interface TrainingPairsPayload {
  transcriptPath: string;
  asrReferencePath: string;
  outputBasename?: string | null;
  asrOnlyMode?: boolean;
  pipelineOverrides?: ConfigMap | null;
}

export interface ModelTrainingPayload {
  corpusDir: string;
  constraintsOutput?: string | null;
  weightsOutput?: string | null;
  iterations: number;
  errorBoostFactor: number;
  pipelineOverrides?: ConfigMap | null;
  modelOverrides?: ConfigMap | null;
}

async function postJob<T>(path: string, payload: T): Promise<JobRecord> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed with status ${res.status}`);
  }
  const job = await res.json();
  return transformJob(job);
}

export function createInferenceJob(payload: InferencePayload): Promise<JobRecord> {
  return postJob("/jobs/inference", payload);
}

export function createTrainingPairsJob(payload: TrainingPairsPayload): Promise<JobRecord> {
  return postJob("/jobs/training-pairs", payload);
}

export function createModelTrainingJob(payload: ModelTrainingPayload): Promise<JobRecord> {
  return postJob("/jobs/model-training", payload);
}

export async function fetchJobLog(jobId: string, offset: number): Promise<LogChunk> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/log?offset=${offset}`);
  if (!res.ok) {
    throw new Error(`Failed to read logs for job ${jobId}`);
  }
  return res.json();
}
