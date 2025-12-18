/**
 * @file This file centralizes all TypeScript type definitions used throughout the
 * frontend application. These types ensure consistency and provide static type
 * checking for data structures received from the backend API.
 */

/**
 * Represents the possible lifecycle states of a background job.
 */
export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/**
 * Represents the available navigation tabs in the frontend shell.
 */
export type TabId = 'inference' | 'trainingPair' | 'modelTraining' | 'config';

/**
 * Represents the full record of a single background job, including its state,
 * parameters, and results.
 */
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

/**
 * Describes the metadata for a single, editable configuration field, used by the
 * UI to render the correct form input.
 */
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

/**
 * Represents a node in the hierarchical configuration tree, used for rendering
 * nested configuration forms.
 */
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

/**
 * A complete snapshot of a configuration, including the effective (merged)
 * values, the user-defined overrides, and the schema metadata.
 */
export interface ConfigSnapshot {
  effective: Record<string, unknown>;
  overrides: Record<string, unknown>;
  fields: ConfigField[];
  schema: ConfigNode[];
}

/**
 * Represents an allowlisted root directory that can be browsed by the UI.
 */
export interface FileRoot {
  id: string;
  label: string;
  path: string;
}

/**
 * Represents a single segment in a file path breadcrumb trail.
 */
export interface FileBreadcrumb {
  label: string;
  path: string;
}

/**
 * Represents a single entry (file or directory) within a file listing.
 */
export interface FileEntry {
  name: string;
  path: string;
  isDir: boolean;
  isFile: boolean;
}

/**
 * Represents the complete content of a directory listing for a specific path.
 */
export interface FileListing {
  root: FileRoot;
  path: string;
  parent?: string | null;
  breadcrumbs: FileBreadcrumb[];
  entries: FileEntry[];
}

/**
 * Represents the result of a file path validation check from the backend.
 */
export interface FileValidation {
  path: string;
  exists: boolean;
  isDir: boolean;
  isFile: boolean;
  allowed: boolean;
  root?: FileRoot | null;
  detail?: string | null;
}
