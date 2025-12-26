import { useCallback, useEffect, useMemo, useState } from "react"
import { fetchEpicGraph, fetchEpics, type Epic, type EpicGraph } from "@/api"
import {
  buildAgentsMap,
  buildChildrenMap,
  buildNodesMap,
  buildTasksMap,
} from "@/lib/graph-utils"

export function useEpicGraph() {
  const [epics, setEpics] = useState<Epic[]>([])
  const [selectedEpicId, setSelectedEpicId] = useState<number | null>(null)
  const [graph, setGraph] = useState<EpicGraph | null>(null)
  const [selectedNodeId, setSelectedNodeId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async (epicIdOverride?: number | null) => {
    setLoading(true)
    setError(null)
    try {
      const epicResponses = await fetchEpics()
      setEpics(epicResponses)

      let epicId: number | null = epicIdOverride ?? epicResponses[0]?.id ?? null
      if (epicId !== null && !epicResponses.some((e) => e.id === epicId)) {
        epicId = epicResponses[0]?.id ?? null
      }
      setSelectedEpicId(epicId)
      setGraph(epicId === null ? null : await fetchEpicGraph(epicId))
      setSelectedNodeId(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh(null)
  }, [refresh])

  const tasksById = useMemo(() => buildTasksMap(graph?.tasks ?? []), [graph])

  const agentsById = useMemo(() => buildAgentsMap(graph?.agents ?? []), [graph])

  const childrenByParent = useMemo(
    () => buildChildrenMap(graph?.nodes ?? []),
    [graph],
  )

  const nodesById = useMemo(() => buildNodesMap(graph?.nodes ?? []), [graph])

  const rootNodes = childrenByParent.get(null) ?? []

  const selectedEpic = useMemo(
    () =>
      selectedEpicId === null
        ? null
        : (epics.find((e) => e.id === selectedEpicId) ?? null),
    [epics, selectedEpicId],
  )

  const selectedNode = useMemo(
    () =>
      selectedNodeId === null ? null : (nodesById.get(selectedNodeId) ?? null),
    [nodesById, selectedNodeId],
  )

  const selectedTask = useMemo(() => {
    if (!selectedNode || selectedNode.primaryTaskId === null) {
      return null
    }
    return tasksById.get(selectedNode.primaryTaskId) ?? null
  }, [selectedNode, tasksById])

  return {
    epics,
    selectedEpicId,
    graph,
    selectedNodeId,
    setSelectedNodeId,
    error,
    loading,
    refresh,
    tasksById,
    agentsById,
    childrenByParent,
    nodesById,
    rootNodes,
    selectedEpic,
    selectedNode,
    selectedTask,
  }
}
