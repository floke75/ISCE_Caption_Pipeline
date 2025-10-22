import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchJobLogs } from '../lib/api';

export const useJobLogs = (jobId?: string) => {
  const [offset, setOffset] = useState(0);
  const [log, setLog] = useState('');

  useEffect(() => {
    setOffset(0);
    setLog('');
  }, [jobId]);

  const query = useQuery({
    queryKey: ['job-log', jobId, offset],
    queryFn: () => fetchJobLogs(jobId as string, offset),
    enabled: Boolean(jobId),
    refetchInterval: jobId ? 2000 : false,
  });

  useEffect(() => {
    if (query.data) {
      setLog((prev) => (offset === 0 ? query.data.content : prev + query.data.content));
      if (query.data.next_offset !== offset) {
        setOffset(query.data.next_offset);
      }
    }
  }, [query.data, offset]);

  return {
    log,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    refresh: query.refetch,
  };
};
