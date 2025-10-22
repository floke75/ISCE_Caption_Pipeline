import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import client from '../api/client';
import { PipelineConfigSnapshot } from '../types';
import toast from 'react-hot-toast';

export function usePipelineConfig() {
  return useQuery<PipelineConfigSnapshot>({
    queryKey: ['config', 'pipeline'],
    queryFn: async () => {
      const { data } = await client.get<PipelineConfigSnapshot>('/config/pipeline');
      return data;
    },
  });
}

export function useUpdateConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (updates: Record<string, unknown>) => {
      const { data } = await client.put<PipelineConfigSnapshot>('/config/pipeline', {
        updates,
      });
      return data;
    },
    onSuccess: () => {
      toast.success('Configuration updated');
      queryClient.invalidateQueries({ queryKey: ['config', 'pipeline'] });
    },
  });
}

export function useReplaceConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (overrides: Record<string, unknown>) => {
      const { data } = await client.put<PipelineConfigSnapshot>('/config/pipeline/replace', {
        overrides,
      });
      return data;
    },
    onSuccess: () => {
      toast.success('Overrides saved');
      queryClient.invalidateQueries({ queryKey: ['config', 'pipeline'] });
    },
  });
}

export function useUpdateConfigYaml() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (yaml: string) => {
      const { data } = await client.put<PipelineConfigSnapshot>('/config/pipeline/raw', {
        yaml,
      });
      return data;
    },
    onSuccess: () => {
      toast.success('YAML overrides saved');
      queryClient.invalidateQueries({ queryKey: ['config', 'pipeline'] });
    },
  });
}
