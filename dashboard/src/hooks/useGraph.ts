import { useCallback, useEffect, useMemo, useState } from "react"
import { fetchEpicGraph, type EpicGraph } from "@/api"
import {
  buildAgentsMap,
  buildChildrenMap,
  buildNodesMap,
  buildTasksMap,
} from "@/lib/graph-utils"

function isNodeOnSpine(
  nodesById: Map<number, { parentNodeId?: number | null }>,
  leafNodeId: number,
  nodeId: number,
): boolean {
  const seen = new Set<number>()
  let cursor: number | null = leafNodeId
  while (cursor !== null && !seen.has(cursor)) {
    if (cursor === nodeId) {
      return true
    }
    seen.add(cursor)
    cursor = nodesById.get(cursor)?.parentNodeId ?? null
  }
  return false
}

type MergeRun = NonNullable<EpicGraph["mergeRuns"]>[number]

export function useGraph(epicId: number | null) {
  const [graph, setGraph] = useState<EpicGraph | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    if (epicId === null) {
      setGraph(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const data = await fetchEpicGraph(epicId)
      setGraph(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [epicId])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const tasksById = useMemo(() => buildTasksMap(graph?.tasks ?? []), [graph])

  const agentsById = useMemo(() => buildAgentsMap(graph?.agents ?? []), [graph])

  const childrenByParent = useMemo(
    () => buildChildrenMap(graph?.nodes ?? []),
    [graph],
  )

  const nodesById = useMemo(() => buildNodesMap(graph?.nodes ?? []), [graph])

  const mergeRunsByTaskId = useMemo(() => {
    const runs = graph?.mergeRuns ?? []
    const map = new Map<number, MergeRun>()
    for (const run of runs) {
      if (map.has(run.requestedTaskId)) {
        continue
      }

      const blockedNodeId = run.blockedNodeId ?? null
      let blockedOnSpine: boolean | null = null
      if (blockedNodeId !== null) {
        const requestedTask = tasksById.get(run.requestedTaskId) ?? null
        const leafNodeId = requestedTask?.nodeId ?? null
        if (leafNodeId !== null && nodesById.size > 0) {
          blockedOnSpine = isNodeOnSpine(nodesById, leafNodeId, blockedNodeId)
        }
      }

      map.set(run.requestedTaskId, { ...run, blockedOnSpine } as MergeRun)
    }
    return map
  }, [graph, nodesById, tasksById])

  const rootNodes = childrenByParent.get(null) ?? []

  return {
    graph,
    error,
    loading,
    refresh,
    tasksById,
    agentsById,
    mergeRunsByTaskId,
    childrenByParent,
    nodesById,
    rootNodes,
  }
}
