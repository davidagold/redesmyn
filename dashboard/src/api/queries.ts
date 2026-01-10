import { useQuery } from "@tanstack/react-query"
import {
  fetchDaemons,
  fetchEpicGraph,
  fetchEpicGithubRepoConfig,
  fetchEpicLinearConfig,
  fetchEpics,
  fetchGitHubStatus,
  fetchHosts,
  fetchLinearMilestones,
  fetchLinearProjects,
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

export function useGitHubStatusQuery() {
  return useQuery({
    queryKey: queryKeys.githubStatus(),
    queryFn: ({ signal }) => fetchGitHubStatus({ signal }),
  })
}

export function useLinearProjectsQuery(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: queryKeys.linearProjects(),
    queryFn: ({ signal }) => fetchLinearProjects({ signal }),
    enabled: options?.enabled ?? true,
  })
}

export function useLinearMilestonesQuery(
  projectId: string | null,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey:
      projectId === null
        ? ["linear", "projects", "none", "milestones"]
        : queryKeys.linearMilestones(projectId),
    queryFn: ({ signal }) => {
      if (projectId === null) {
        throw new Error("Project id missing for milestones query.")
      }
      return fetchLinearMilestones(projectId, { signal })
    },
    enabled: (options?.enabled ?? true) && projectId !== null,
  })
}

export function useEpicLinearConfigQuery(
  epicSlug: string | null,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey:
      epicSlug === null
        ? ["epics", "none", "linear", "config"]
        : queryKeys.epicLinearConfig(epicSlug),
    queryFn: ({ signal }) => {
      if (epicSlug === null) {
        throw new Error("Epic slug missing for Linear config query.")
      }
      return fetchEpicLinearConfig(epicSlug, { signal })
    },
    enabled: (options?.enabled ?? true) && epicSlug !== null,
  })
}

export function useEpicGithubRepoConfigQuery(
  epicSlug: string | null,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey:
      epicSlug === null
        ? ["epics", "none", "github", "repo"]
        : queryKeys.epicGithubRepoConfig(epicSlug),
    queryFn: ({ signal }) => {
      if (epicSlug === null) {
        throw new Error("Epic slug missing for GitHub repo query.")
      }
      return fetchEpicGithubRepoConfig(epicSlug, { signal })
    },
    enabled: (options?.enabled ?? true) && epicSlug !== null,
  })
}
