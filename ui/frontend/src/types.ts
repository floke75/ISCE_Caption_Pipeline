export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface JobRecord {
  id: string;
  jobType: 'inference' | 'training_pair' | 'model_training' | string;
  status: JobStatus;
  progress: number;
  message: string;
  createdAt: string;
  updatedAt: string;
  params: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ConfigField {
  path: string[];
  label: string;
  fieldType: 'string' | 'number' | 'boolean' | 'path' | 'list' | 'select';
  section: string;
  description?: string;
  options?: string[];
  advanced?: boolean;
}

export interface PipelineConfigSnapshot {
  effective: Record<string, unknown>;
  overrides: Record<string, unknown>;
  fields: ConfigField[];
}

export interface OverrideEntry {
  path: string;
  value: string;
}
