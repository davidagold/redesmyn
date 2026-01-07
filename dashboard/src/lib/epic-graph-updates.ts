import type { EpicGraph } from "@/api"
import type { TaskAgentSessionUpdateEventData } from "@/hooks/useEventStream"

export function applyTaskAgentSessionUpdate(
  graph: EpicGraph | undefined,
  update: TaskAgentSessionUpdateEventData,
): EpicGraph | undefined {
  if (!graph || graph.agentSessions.length === 0) {
    return graph
  }

  const index = graph.agentSessions.findIndex(
    (session) =>
      session.id === update.agentSessionId || session.taskId === update.taskId,
  )
  if (index < 0) {
    return graph
  }

  const current = graph.agentSessions[index]!
  const nextSessions = [...graph.agentSessions]
  nextSessions[index] = {
    ...current,
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
