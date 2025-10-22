export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface JobArtifact {
  name: string;
  path: string;
}

export interface JobRecord {
  id: string;
  type: string;
  status: JobStatus;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progress: number;
  stage?: string | null;
  message?: string | null;
  artifacts: JobArtifact[];
  params: Record<string, unknown>;
  result: Record<string, unknown>;
}

export type ConfigValue = string | number | boolean | null | ConfigMap;
export interface ConfigMap {
  [key: string]: ConfigValue;
}

export interface ConfigResponse {
  config: ConfigMap;
  yaml: string;
}

export interface LogChunk {
  content: string;
  offset: number;
  complete: boolean;
}
