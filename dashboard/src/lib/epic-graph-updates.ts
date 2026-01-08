import type { EpicGraph } from "@/api"
import type { TaskAgentSessionUpdateEventData } from "@/hooks/useEventStream"

export function applyTaskAgentSessionUpdate(
  graph: EpicGraph | undefined,
  update: TaskAgentSessionUpdateEventData,
): EpicGraph | undefined {
  if (!graph || graph.agentSessions.length === 0) {
    return graph
  }

  const byIdIndex = graph.agentSessions.findIndex(
    (session) => session.id === update.agentSessionId,
  )
  const byTaskIdIndex =
    byIdIndex >= 0
      ? -1
      : graph.agentSessions.findIndex(
          (session) => session.taskId === update.taskId,
        )

  if (byIdIndex < 0 && byTaskIdIndex < 0) {
    return graph
  }

  // Prefer matching by `agentSessionId` when possible. If we only match by
  // `taskId`, update the cached session id so downstream UI logic that keys off
  // `agentSessions[].id` doesn't get stuck on the stale id after restarts.
  const index = byIdIndex >= 0 ? byIdIndex : byTaskIdIndex
  const current = graph.agentSessions[index]!
  const nextSessions = [...graph.agentSessions]

  // If the update's session id is already present elsewhere (unexpected, but
  // possible during resync/races), update that entry and drop the stale one.
  const hasDuplicateId =
    byIdIndex < 0 &&
    nextSessions.some((session) => session.id === update.agentSessionId)
  if (hasDuplicateId) {
    const canonicalIndex = nextSessions.findIndex(
      (session) => session.id === update.agentSessionId,
    )
    if (canonicalIndex >= 0) {
      const canonicalCurrent = nextSessions[canonicalIndex]!
      nextSessions[canonicalIndex] = {
        ...canonicalCurrent,
        taskId: update.taskId,
        status: update.agentStatus,
        startedAt: update.startedAt ?? canonicalCurrent.startedAt ?? null,
        endedAt: update.endedAt ?? canonicalCurrent.endedAt ?? null,
        agentKindSelection: update.agentKindSelection,
        agentKind: update.agentKind,
        agentInterfaceMode: update.agentInterfaceMode,
        agentCapabilities: update.agentCapabilities,
        agentSemanticStatus: update.agentSemanticStatus,
        externalSessionRef: update.externalSessionRef,
        agentPreview: update.agentPreview ?? canonicalCurrent.agentPreview,
      }

      nextSessions.splice(index, 1)
      return { ...graph, agentSessions: nextSessions }
    }
  }

  nextSessions[index] = {
    ...current,
    id: update.agentSessionId,
    taskId: update.taskId,
    status: update.agentStatus,
    startedAt: update.startedAt ?? current.startedAt ?? null,
    endedAt: update.endedAt ?? current.endedAt ?? null,
    agentKindSelection: update.agentKindSelection,
    agentKind: update.agentKind,
    agentInterfaceMode: update.agentInterfaceMode,
    agentCapabilities: update.agentCapabilities,
    agentSemanticStatus: update.agentSemanticStatus,
    externalSessionRef: update.externalSessionRef,
    agentPreview: update.agentPreview ?? current.agentPreview,
  }

  return { ...graph, agentSessions: nextSessions }
}
