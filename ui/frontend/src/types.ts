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
  workspacePath: string;
}

export interface ConfigField {
  path: string[];
  label: string;
  fieldType: 'string' | 'number' | 'boolean' | 'path' | 'list' | 'select';
  section: string;
  description?: string;
  options?: string[];
  advanced?: boolean;
  readOnly?: boolean;
}

export interface ConfigNode {
  key: string;
  path: string[];
  label: string;
  valueType: 'string' | 'number' | 'boolean' | 'path' | 'list' | 'select' | 'object';
  description?: string;
  default?: unknown;
  current?: unknown;
  options?: unknown[];
  advanced?: boolean;
  overridden?: boolean;
  children?: ConfigNode[];
}

export interface ConfigSnapshot {
  effective: Record<string, unknown>;
  overrides: Record<string, unknown>;
  fields: ConfigField[];
  schema: ConfigNode[];
}

export interface FileRoot {
  id: string;
  label: string;
  path: string;
}

export interface FileBreadcrumb {
  label: string;
  path: string;
}

export interface FileEntry {
  name: string;
  path: string;
  isDir: boolean;
  isFile: boolean;
}

export interface FileListing {
  root: FileRoot;
  path: string;
  parent?: string | null;
  breadcrumbs: FileBreadcrumb[];
  entries: FileEntry[];
}

export interface FileValidation {
  path: string;
  exists: boolean;
  isDir: boolean;
  isFile: boolean;
  allowed: boolean;
  root?: FileRoot | null;
  detail?: string | null;
}
