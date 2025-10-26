/**
 * @file This file contains custom React Query hooks for fetching and updating
 * the application's configuration from the backend API.
 */
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

/**
 * A React Query hook for fetching the main pipeline configuration.
 * @returns {QueryResult<ConfigSnapshot>} The result of the query.
 */
export function usePipelineConfig() {
  return useConfigQuery('pipeline');
}

/**
 * A React Query hook for fetching the segmentation model configuration.
 * @returns {QueryResult<ConfigSnapshot>} The result of the query.
 */
export function useSegmentationConfig() {
  return useConfigQuery('segmentation');
}

/**
 * A React Query mutation hook for applying partial updates to a configuration.
 * @param {ConfigResource} resource The type of configuration to update.
 * @returns {MutationResult<ConfigSnapshot, unknown, Record<string, unknown>>} The result of the mutation.
 */
export function useUpdateConfig(resource: ConfigResource = 'pipeline') {
  return useConfigMutation(resource, '', `${RESOURCE_LABEL[resource]} updated`, (updates: Record<string, unknown>) => ({
    updates,
  }));
}

/**
 * A React Query mutation hook for replacing the entire override set for a configuration.
 * @param {ConfigResource} resource The type of configuration to replace.
 * @returns {MutationResult<ConfigSnapshot, unknown, Record<string, unknown>>} The result of the mutation.
 */
export function useReplaceConfig(resource: ConfigResource = 'pipeline') {
  return useConfigMutation(
    resource,
    '/replace',
    'Overrides saved',
    (overrides: Record<string, unknown>) => ({ overrides })
  );
}

/**
 * A React Query mutation hook for updating a configuration from a raw YAML string.
 * @param {ConfigResource} resource The type of configuration to update.
 * @returns {MutationResult<ConfigSnapshot, unknown, string>} The result of the mutation.
 */
export function useUpdateConfigYaml(resource: ConfigResource = 'pipeline') {
  return useConfigMutation(resource, '/raw', 'YAML overrides saved', (yaml: string) => ({ yaml }));
}
