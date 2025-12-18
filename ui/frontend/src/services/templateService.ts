export type JobType = 'inference' | 'training_pair' | 'model_training';

export interface TemplateData {
  notes?: string;
  overrides?: {
    pipeline?: Record<string, unknown>;
    segmentation?: Record<string, unknown>;
  };
  // Inference
  outputDir?: string;
  modelConfigPath?: string;
  preset?: string;
  beamWidth?: number;
  diarization?: boolean;
  // Model Training
  iterations?: number;
  errorBoost?: number;
}

export interface JobTemplate {
  id: string;
  name: string;
  type: JobType;
  createdAt: string;
  data: TemplateData;
}

const STORAGE_KEY = 'isce_job_templates';

export const templateService = {
  getTemplates(type: JobType): JobTemplate[] {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    try {
      const all: JobTemplate[] = JSON.parse(raw);
      return all.filter((t) => t.type === type).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    } catch (e) {
      console.error('Failed to parse templates', e);
      return [];
    }
  },

  saveTemplate(name: string, type: JobType, data: TemplateData): JobTemplate {
    const raw = localStorage.getItem(STORAGE_KEY);
    const all: JobTemplate[] = raw ? JSON.parse(raw) : [];

    const newTemplate: JobTemplate = {
      id: crypto.randomUUID(),
      name,
      type,
      createdAt: new Date().toISOString(),
      data,
    };

    all.push(newTemplate);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
    return newTemplate;
  },

  deleteTemplate(id: string): void {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const all: JobTemplate[] = JSON.parse(raw);
    const filtered = all.filter((t) => t.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  }
};
