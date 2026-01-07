import { useCallback, useEffect, useRef } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { queryKeys } from "@/api/queryKeys"
import type { EpicGraph } from "@/api"
import { type StreamEvent, useEventStream } from "@/hooks/useEventStream"

export function useEpicCacheSync(options: {
  epicSlug: string | null
  epicId: number | null
  onEvent?: (event: StreamEvent) => void
}) {
  const { epicSlug, epicId, onEvent } = options
  const queryClient = useQueryClient()

  const graphInvalidateTimerRef = useRef<number | null>(null)
  const daemonsInvalidateTimerRef = useRef<number | null>(null)

  const invalidateEpicGraph = useCallback(
    (delayMs: number) => {
      if (epicId === null) {
        return
      }
      if (graphInvalidateTimerRef.current !== null) {
        return
      }

      graphInvalidateTimerRef.current = window.setTimeout(() => {
        graphInvalidateTimerRef.current = null
        void queryClient.invalidateQueries({
          queryKey: queryKeys.epicGraph(epicId),
        })
      }, delayMs)
    },
    [epicId, queryClient],
  )

  const invalidateDaemons = useCallback(
    (delayMs: number) => {
      if (daemonsInvalidateTimerRef.current !== null) {
        return
      }

      daemonsInvalidateTimerRef.current = window.setTimeout(() => {
        daemonsInvalidateTimerRef.current = null
        void queryClient.invalidateQueries({
          queryKey: queryKeys.daemons(),
        })
      }, delayMs)
    },
    [queryClient],
  )

  const patchMergeRun = useCallback(
    (event: StreamEvent) => {
      const data = event.data
      if (epicId === null || data.type !== "merge.run") {
        return
      }
      if (data.epicId !== epicId) {
        return
      }
      const updatedAt = event.createdAt

      queryClient.setQueryData<EpicGraph>(
        queryKeys.epicGraph(epicId),
        (graph) => {
          if (!graph || !graph.mergeRuns || graph.mergeRuns.length === 0) {
            return graph
          }

          const index = graph.mergeRuns.findIndex(
            (run) => run.runId === data.runId,
          )
          if (index < 0) {
            return graph
          }

          const current = graph.mergeRuns[index]!
          const nextRuns = [...graph.mergeRuns]
          nextRuns[index] = {
            ...current,
            status: data.status,
            operation: data.operation ?? current.operation,
            blockedStepIndex:
              data.blockedStepIndex ?? current.blockedStepIndex ?? null,
            blockedStepKind:
              data.blockedStepKind ?? current.blockedStepKind ?? null,
            blockedBranchName:
              data.blockedBranchName ?? current.blockedBranchName ?? null,
            updatedAt,
          }
          return { ...graph, mergeRuns: nextRuns }
        },
      )
    },
    [epicId, queryClient],
  )

  const handleStreamEvent = useCallback(
    (event: StreamEvent) => {
      if (event.eventType.startsWith("daemon.")) {
        invalidateDaemons(250)
        invalidateEpicGraph(250)
        onEvent?.(event)
        return
      }

      if (event.eventType === "task.stack_in_sync") {
        invalidateEpicGraph(250)
        onEvent?.(event)
        return
      }

      if (event.eventType === "merge.run") {
        patchMergeRun(event)
        invalidateEpicGraph(600)
        onEvent?.(event)
        return
      }

      if (event.eventType === "node.stack_in_sync") {
        invalidateEpicGraph(750)
        onEvent?.(event)
        return
      }

      if (event.eventType === "task.merge") {
        invalidateEpicGraph(250)
        onEvent?.(event)
        return
      }

      if (
        event.eventType === "task.agent_run" ||
        event.eventType === "task.agent_action"
      ) {
        invalidateEpicGraph(750)
        onEvent?.(event)
        return
      }

      onEvent?.(event)
    },
    [invalidateDaemons, invalidateEpicGraph, onEvent, patchMergeRun],
  )

  const { status } = useEventStream({
    epic: epicSlug,
    onEvent: handleStreamEvent,
    onResync: () => invalidateEpicGraph(0),
  })

  useEffect(() => {
    return () => {
      if (graphInvalidateTimerRef.current !== null) {
        window.clearTimeout(graphInvalidateTimerRef.current)
      }
      if (daemonsInvalidateTimerRef.current !== null) {
        window.clearTimeout(daemonsInvalidateTimerRef.current)
      }
    }
  }, [])

  return { status }
}
