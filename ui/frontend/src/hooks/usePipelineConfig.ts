import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export type PipelineConfig = Record<string, unknown>;

async function fetchConfig(): Promise<PipelineConfig> {
  const response = await fetch("/api/config/pipeline");
  if (!response.ok) {
    throw new Error("Unable to load pipeline configuration");
  }
  const payload = await response.json();
  return payload.data ?? {};
}

async function updateConfig(data: PipelineConfig): Promise<PipelineConfig> {
  const response = await fetch("/api/config/pipeline", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });
  if (!response.ok) {
    throw new Error("Failed to update configuration");
  }
  const payload = await response.json();
  return payload.data ?? {};
}

export function usePipelineConfig() {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: fetchConfig,
  });

  const mutation = useMutation({
    mutationFn: updateConfig,
    onSuccess: (data) => {
      queryClient.setQueryData(["pipeline-config"], data);
    },
  });

  return { ...query, updateConfig: mutation.mutateAsync, isUpdating: mutation.isPending };
}
