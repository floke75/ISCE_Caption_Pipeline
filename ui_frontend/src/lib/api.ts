import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export interface JobSummary {
  id: string;
  job_type: string;
  name: string;
  status: string;
  progress: number;
  stage?: string | null;
  message?: string | null;
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  result?: Record<string, unknown> | null;
  metrics: Record<string, unknown>;
}

export interface JobDetail extends JobSummary {
  parameters: Record<string, unknown>;
  workspace: string;
}

export const fetchJobs = async (): Promise<JobSummary[]> => {
  const { data } = await api.get('/api/jobs');
  return data;
};

export const fetchJob = async (jobId: string): Promise<JobDetail> => {
  const { data } = await api.get(`/api/jobs/${jobId}`);
  return data;
};

export const createInferenceJob = async (payload: Record<string, unknown>): Promise<JobDetail> => {
  const { data } = await api.post('/api/jobs/inference', payload);
  return data;
};

export const createTrainingPairJob = async (payload: Record<string, unknown>): Promise<JobDetail> => {
  const { data } = await api.post('/api/jobs/training-pair', payload);
  return data;
};

export const createModelTrainingJob = async (payload: Record<string, unknown>): Promise<JobDetail> => {
  const { data } = await api.post('/api/jobs/model-training', payload);
  return data;
};

export const fetchPipelineConfig = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.get('/api/config/pipeline');
  return data;
};

export const savePipelineConfig = async (config: Record<string, unknown>): Promise<Record<string, unknown>> => {
  const { data } = await api.put('/api/config/pipeline', config);
  return data;
};

export const fetchPipelineYaml = async (): Promise<string> => {
  const { data } = await api.get('/api/config/pipeline/yaml', { responseType: 'text' });
  return data;
};

export const fetchModelConfig = async (): Promise<Record<string, unknown>> => {
  const { data } = await api.get('/api/config/model');
  return data;
};

export const saveModelConfig = async (config: Record<string, unknown>): Promise<Record<string, unknown>> => {
  const { data } = await api.put('/api/config/model', config);
  return data;
};

export const fetchModelYaml = async (): Promise<string> => {
  const { data } = await api.get('/api/config/model/yaml', { responseType: 'text' });
  return data;
};
