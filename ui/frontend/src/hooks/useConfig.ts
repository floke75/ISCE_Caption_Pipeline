import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import client from '../api/client';
import { ConfigSnapshot } from '../types';

type ConfigResource = 'pipeline' | 'segmentation';

const RESOURCE_LABEL: Record<ConfigResource, string> = {
  pipeline: 'Pipeline configuration',
  segmentation: 'Segmentation configuration',
};

function basePath(resource: ConfigResource): string {
  return `/config/${resource}`;
}

function useConfigQuery(resource: ConfigResource) {
  return useQuery<ConfigSnapshot>({
    queryKey: ['config', resource],
    queryFn: async () => {
      const { data } = await client.get<ConfigSnapshot>(basePath(resource));
      return data;
    },
  });
}

function useConfigMutation<TInput, TPayload>(
  resource: ConfigResource,
  pathSuffix: string,
  successMessage: string,
  buildPayload: (input: TInput) => TPayload
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (input: TInput) => {
      const payload = buildPayload(input);
      const { data } = await client.put<ConfigSnapshot>(`${basePath(resource)}${pathSuffix}`, payload);
      return data;
    },
    onSuccess: () => {
      toast.success(successMessage);
      queryClient.invalidateQueries({ queryKey: ['config', resource] });
    },
  });
}

export function usePipelineConfig() {
  return useConfigQuery('pipeline');
}

export function useSegmentationConfig() {
  return useConfigQuery('segmentation');
}

export function useUpdateConfig(resource: ConfigResource = 'pipeline') {
  return useConfigMutation(resource, '', `${RESOURCE_LABEL[resource]} updated`, (updates: Record<string, unknown>) => ({
    updates,
  }));
}

export function useReplaceConfig(resource: ConfigResource = 'pipeline') {
  return useConfigMutation(
    resource,
    '/replace',
    'Overrides saved',
    (overrides: Record<string, unknown>) => ({ overrides })
  );
}

export function useUpdateConfigYaml(resource: ConfigResource = 'pipeline') {
  return useConfigMutation(resource, '/raw', 'YAML overrides saved', (yaml: string) => ({ yaml }));
}
