import { useCallback, useEffect, useMemo, useState } from "react"
import { fetchEpicGraph, type EpicGraph } from "@/api"
import {
  buildAgentsMap,
  buildChildrenMap,
  buildNodesMap,
  buildSessionsByNodeId,
  buildTasksMap,
} from "@/lib/graph-utils"

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

  const sessionsByNodeId = useMemo(
    () => buildSessionsByNodeId(graph?.sessions ?? []),
    [graph],
  )

  const childrenByParent = useMemo(
    () => buildChildrenMap(graph?.nodes ?? []),
    [graph],
  )

  const nodesById = useMemo(() => buildNodesMap(graph?.nodes ?? []), [graph])

  const rootNodes = childrenByParent.get(null) ?? []

  return {
    graph,
    error,
    loading,
    refresh,
    tasksById,
    agentsById,
    sessionsByNodeId,
    childrenByParent,
    nodesById,
    rootNodes,
  }
}
