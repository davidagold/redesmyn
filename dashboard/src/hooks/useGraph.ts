import { useCallback, useMemo } from "react"
import type { EpicGraph } from "@/api"
import { useEpicGraphQuery } from "@/api/queries"
import {
  buildAgentSessionsByNodeId,
  buildChildrenMap,
  buildNodesMap,
  buildTasksMap,
} from "@/lib/graph-utils"

function isNodeOnSpine(
  nodesById: Map<number, { parentTaskId?: number | null }>,
  leafTaskId: number,
  taskId: number,
): boolean {
  const seen = new Set<number>()
  let cursor: number | null = leafTaskId
  while (cursor !== null && !seen.has(cursor)) {
    if (cursor === taskId) {
      return true
    }
    seen.add(cursor)
    cursor = nodesById.get(cursor)?.parentTaskId ?? null
  }
  return false
}

type MergeRun = NonNullable<EpicGraph["mergeRuns"]>[number]

export function useGraph(epicId: number | null) {
  const {
    data,
    error: queryError,
    isFetching,
    refetch,
  } = useEpicGraphQuery(epicId)
  const graph = epicId === null ? null : (data ?? null)

  const refresh = useCallback(async () => {
    if (epicId === null) {
      return
    }
    await refetch()
  }, [epicId, refetch])

  const tasksById = useMemo(() => buildTasksMap(graph?.tasks ?? []), [graph])

  const agentSessionsByNodeId = useMemo(
    () => buildAgentSessionsByNodeId(graph?.agentSessions ?? []),
    [graph],
  )

  const childrenByParent = useMemo(
    () => buildChildrenMap(graph?.tasks ?? []),
    [graph],
  )

  const nodesById = useMemo(() => buildNodesMap(graph?.tasks ?? []), [graph])

  const mergeRunsByTaskId = useMemo(() => {
    const runs = graph?.mergeRuns ?? []
    const map = new Map<number, MergeRun>()
    for (const run of runs) {
      if (map.has(run.requestedTaskId)) {
        continue
      }

      const blockedTaskId = run.blockedTaskId ?? null
      let blockedOnSpine: boolean | null = null
      if (blockedTaskId !== null && nodesById.size > 0) {
        blockedOnSpine = isNodeOnSpine(
          nodesById,
          run.requestedTaskId,
          blockedTaskId,
        )
      }

      map.set(run.requestedTaskId, { ...run, blockedOnSpine } as MergeRun)
    }
    return map
  }, [graph, nodesById])

  const rootNodes = childrenByParent.get(null) ?? []

  return {
    graph,
    error: queryError
      ? queryError instanceof Error
        ? queryError.message
        : String(queryError)
      : null,
    loading: epicId !== null && isFetching,
    refresh,
    tasksById,
    agentSessionsByNodeId,
    mergeRunsByTaskId,
    childrenByParent,
    nodesById,
    rootNodes,
  }
}
