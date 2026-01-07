import { useQuery } from "@tanstack/react-query"
import {
  fetchDaemons,
  fetchEpicGraph,
  fetchEpics,
  fetchHosts,
  fetchLinearStatus,
  fetchOrchestrationDefaults,
} from "@/api"
import { queryKeys } from "@/api/queryKeys"

export function useEpicsQuery() {
  return useQuery({
    queryKey: queryKeys.epics(),
    queryFn: ({ signal }) => fetchEpics({ signal }),
  })
}

export function useEpicGraphQuery(epicId: number | null) {
  return useQuery({
    queryKey:
      epicId === null
        ? ["epics", "none", "graph"]
        : queryKeys.epicGraph(epicId),
    queryFn: ({ signal }) => {
      if (epicId === null) {
        throw new Error("Epic id missing for graph query.")
      }
      return fetchEpicGraph(epicId, { signal })
    },
    enabled: epicId !== null,
  })
}

export function useHostsQuery() {
  return useQuery({
    queryKey: queryKeys.hosts(),
    queryFn: ({ signal }) => fetchHosts({ signal }),
  })
}

export function useDaemonsQuery(options?: { pollIntervalMs?: number }) {
  const pollIntervalMs = options?.pollIntervalMs ?? 15_000

  return useQuery({
    queryKey: queryKeys.daemons(),
    queryFn: ({ signal }) => fetchDaemons({ signal }),
    refetchInterval:
      Number.isFinite(pollIntervalMs) && pollIntervalMs > 0
        ? pollIntervalMs
        : false,
    refetchIntervalInBackground: false,
  })
}

export function useOrchestrationDefaultsQuery() {
  return useQuery({
    queryKey: queryKeys.orchestrationDefaults(),
    queryFn: ({ signal }) => fetchOrchestrationDefaults({ signal }),
  })
}

export function useLinearStatusQuery() {
  return useQuery({
    queryKey: queryKeys.linearStatus(),
    queryFn: ({ signal }) => fetchLinearStatus({ signal }),
  })
}
