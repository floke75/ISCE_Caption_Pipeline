import { useQuery } from '@tanstack/react-query';
import client from '../api/client';
import { JobRecord } from '../types';

interface JobLogOptions {
  enabled?: boolean;
  refetchInterval?: number | false;
}

export function useJobs() {
  return useQuery<JobRecord[]>({
    queryKey: ['jobs'],
    queryFn: async () => {
      const { data } = await client.get<JobRecord[]>('/jobs');
      return data;
    },
    refetchInterval: 5000,
  });
}

export function useJobLog(jobId: string | null, options: JobLogOptions = {}) {
  const enabled = Boolean(jobId) && (options.enabled ?? true);
  return useQuery<{ log: string }>({
    queryKey: ['jobs', jobId, 'log'],
    enabled,
    refetchInterval: options.refetchInterval ?? 4000,
    queryFn: async () => {
      const { data } = await client.get<{ log: string }>(`/jobs/${jobId}/logs`, {
        params: { tail: 12000 },
      });
      return data;
    },
  });
}
