/**
 * @file This file contains custom React Query hooks for fetching job-related
 * data from the backend API, including the list of all jobs and individual
 * job logs.
 */
import { useQuery } from '@tanstack/react-query';
import client from '../api/client';
import { JobRecord } from '../types';

interface JobLogOptions {
  enabled?: boolean;
  refetchInterval?: number | false;
}

/**
 * A React Query hook for fetching the list of all jobs.
 *
 * This hook retrieves the full list of jobs from the `/api/jobs` endpoint and
 * automatically refetches the data every 5 seconds to keep the UI updated.
 *
 * @returns {QueryResult<JobRecord[]>} The result of the query.
 */
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

/**
 * A React Query hook for fetching the log of a single job.
 *
 * This hook retrieves the log content for a specific job ID. It can be configured
 * to poll for updates, which is used as a fallback when the real-time SSE
 * (Server-Sent Events) stream is unavailable.
 *
 * @param {string | null} jobId The ID of the job to fetch the log for. The query is disabled if this is null.
 * @param {JobLogOptions} options Configuration for the query, such as `enabled` and `refetchInterval`.
 * @returns {QueryResult<{ log: string }>} The result of the query.
 */
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
