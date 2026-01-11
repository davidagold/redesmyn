import { useMutation, useQueryClient } from "@tanstack/react-query"
import type {
  MergeRunCancelRequest,
  MergeRunResumeRequest,
  OrchestrationDefaults,
  OrchestrationDefaultsUpdateRequest,
  Task,
  TaskAgentBulkActionRequest,
  TaskAgentMessageRequest,
  TaskAgentRestartRequest,
  TaskAgentStartRequest,
  TaskMergeRequest,
  TaskRestackRequest,
} from "@/api"
import {
  bulkTaskAgentActions,
  cancelMergeRun,
  mergeTask,
  postLinearLogout,
  postTaskAgentMessage,
  postSyncFromLinear,
  postSyncToLinear,
  resumeMergeRun,
  restackTask,
  restartTaskAgent,
  setTaskMergeReady,
  startTaskAgent,
  stopTaskAgent,
  updateEpicGithubRepo,
  updateEpicLinearConfig,
  updateEpicLinearProject,
  updateOrchestrationDefaults,
  type EpicGraph,
} from "@/api"
import { queryKeys } from "@/api/queryKeys"

type SetTaskMergeReadyVariables = {
  taskId: number
  ready: boolean
  scope?: "task" | "spine"
}

type StopTaskAgentVariables = {
  epicId: number
  taskId: number
}

type LinearSyncVariables = {
  epicId: number | null
  epicSlug: string
}

function updateTaskInGraph(graph: EpicGraph, task: Task): EpicGraph {
  const tasks = graph.tasks ?? []
  const index = tasks.findIndex((t) => t.id === task.id)
  if (index < 0) {
    return graph
  }
  const nextTasks = [...tasks]
  nextTasks[index] = task
  return { ...graph, tasks: nextTasks }
}

function updateTaskInGraphById(
  graph: EpicGraph,
  taskId: number,
  update: (task: Task) => Task,
): EpicGraph {
  const tasks = graph.tasks ?? []
  const index = tasks.findIndex((t) => t.id === taskId)
  if (index < 0) {
    return graph
  }
  const nextTasks = [...tasks]
  nextTasks[index] = update(nextTasks[index]!)
  return { ...graph, tasks: nextTasks }
}

export function useUpdateOrchestrationDefaultsMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: OrchestrationDefaultsUpdateRequest) =>
      updateOrchestrationDefaults(request),
    onSuccess: (defaults: OrchestrationDefaults) => {
      queryClient.setQueryData(queryKeys.orchestrationDefaults(), defaults)
    },
  })
}

export function useSetTaskMergeReadyMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (variables: SetTaskMergeReadyVariables) => {
      return setTaskMergeReady(variables.taskId, variables.ready, {
        scope: variables.scope,
      })
    },
    onMutate: async (variables) => {
      const now = new Date().toISOString()
      const previousGraphEntries: Array<{
        key: readonly unknown[]
        value: EpicGraph
      }> = []

      const epicGraphs = queryClient.getQueriesData<EpicGraph>({
        queryKey: ["epics"],
        exact: false,
      })

      for (const [key, graph] of epicGraphs) {
        if (
          !graph ||
          !Array.isArray(key) ||
          key.length !== 3 ||
          key[0] !== "epics" ||
          key[2] !== "graph"
        ) {
          continue
        }
        const existing = graph.tasks.find((t) => t.id === variables.taskId)
        if (!existing) {
          continue
        }
        previousGraphEntries.push({ key, value: graph })
      }

      for (const entry of previousGraphEntries) {
        queryClient.setQueryData<EpicGraph>(entry.key, (current) => {
          if (!current) {
            return current
          }
          return updateTaskInGraphById(current, variables.taskId, (task) => ({
            ...task,
            mergeReadyAt: variables.ready ? now : null,
          }))
        })
      }

      return { previousGraphEntries }
    },
    onError: (_error, _variables, context) => {
      for (const entry of context?.previousGraphEntries ?? []) {
        queryClient.setQueryData(entry.key, entry.value)
      }
    },
    onSuccess: (task, variables) => {
      if (variables.scope === "spine" && variables.ready) {
        // TODO: For smoother UI, consider having the API return the list of updated
        // task ids (or updated tasks) so we can patch the graph cache precisely.
        void queryClient.invalidateQueries({
          queryKey: queryKeys.epicGraph(task.epicId),
        })
        return
      }

      queryClient.setQueryData<EpicGraph>(
        queryKeys.epicGraph(task.epicId),
        (graph) => (graph ? updateTaskInGraph(graph, task) : graph),
      )
    },
  })
}

export function useStartTaskAgentMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      taskId: number
      request: TaskAgentStartRequest
    }) => startTaskAgent(variables.taskId, variables.request),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useRestartTaskAgentMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      taskId: number
      request: TaskAgentRestartRequest
    }) => restartTaskAgent(variables.taskId, variables.request),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useStopTaskAgentMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: StopTaskAgentVariables) =>
      stopTaskAgent(variables.taskId),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useTaskAgentMessageMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (variables: {
      epicId: number
      taskId: number
      request: TaskAgentMessageRequest
    }) => postTaskAgentMessage(variables.taskId, variables.request),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useMergeTaskMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      taskId: number
      request: TaskMergeRequest
    }) => {
      return mergeTask(variables.taskId, variables.request)
    },
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useRestackTaskMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      taskId: number
      request: TaskRestackRequest
    }) => {
      return restackTask(variables.taskId, variables.request)
    },
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useResumeMergeRunMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      runId: string
      request: MergeRunResumeRequest
    }) => resumeMergeRun(variables.runId, variables.request),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useCancelMergeRunMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      runId: string
      request: MergeRunCancelRequest
    }) => cancelMergeRun(variables.runId, variables.request),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useBulkTaskAgentActionsMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicId: number
      request: TaskAgentBulkActionRequest
    }) => {
      return bulkTaskAgentActions(variables.request)
    },
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(variables.epicId),
      })
    },
  })
}

export function useLinearLogoutMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => postLinearLogout(),
    onSuccess: (status) => {
      queryClient.setQueryData(queryKeys.linearStatus(), status)
    },
  })
}

export function useSyncFromLinearMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: LinearSyncVariables) =>
      postSyncFromLinear(variables.epicSlug),
    onSuccess: (_result, variables) => {
      if (variables.epicId !== null) {
        void queryClient.invalidateQueries({
          queryKey: queryKeys.epicGraph(variables.epicId),
        })
        return
      }
      void queryClient.invalidateQueries({ queryKey: ["epics"], exact: false })
    },
  })
}

export function useSyncToLinearMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: LinearSyncVariables) =>
      postSyncToLinear(variables.epicSlug),
    onSuccess: (_result, variables) => {
      if (variables.epicId !== null) {
        void queryClient.invalidateQueries({
          queryKey: queryKeys.epicGraph(variables.epicId),
        })
        return
      }
      void queryClient.invalidateQueries({ queryKey: ["epics"], exact: false })
    },
  })
}

export function useUpdateEpicLinearProjectMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicSlug: string
      linearProjectId: string | null
    }) =>
      updateEpicLinearProject(variables.epicSlug, variables.linearProjectId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.epics() })
    },
  })
}

export function useUpdateEpicLinearConfigMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: {
      epicSlug: string
      labelId?: string | null
      labelName?: string | null
      milestoneId?: string | null
    }) =>
      updateEpicLinearConfig(variables.epicSlug, {
        labelId: variables.labelId,
        labelName: variables.labelName,
        milestoneId: variables.milestoneId,
      }),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicLinearConfig(variables.epicSlug),
      })
    },
  })
}

type UpdateEpicGithubRepoVariables = {
  epicSlug: string
  repo: string | null
}

export function useUpdateEpicGithubRepoMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (variables: UpdateEpicGithubRepoVariables) =>
      updateEpicGithubRepo(variables.epicSlug, variables.repo),
    onSuccess: (_result, variables) => {
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGithubRepoConfig(variables.epicSlug),
      })
      void queryClient.invalidateQueries({ queryKey: queryKeys.epics() })
    },
  })
}
