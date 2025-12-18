import { useQuery } from '@tanstack/react-query';
import client from '../api/client';

export function useArtifact<T = unknown>(path: string) {
  return useQuery({
    queryKey: ['artifact', path],
    queryFn: async () => {
      const response = await client.get<T>(`/files/content?path=${encodeURIComponent(path)}`);
      return response.data;
    },
    enabled: !!path,
    staleTime: 1000 * 60 * 60, // 1 hour
  });
}
