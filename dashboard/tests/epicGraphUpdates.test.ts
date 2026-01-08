import test from "node:test"
import assert from "node:assert/strict"
import type { EpicGraph } from "../src/api.ts"
import type { TaskAgentSessionUpdateEventData } from "../src/hooks/useEventStream.ts"
import { applyTaskAgentSessionUpdate } from "../src/lib/epic-graph-updates.ts"

type AgentSession = EpicGraph["agentSessions"][number]

function makeAgentSession(overrides: Partial<AgentSession>): AgentSession {
  return {
    id: 10,
    taskId: 95,
    status: "running",
    startedAt: null,
    endedAt: null,
    agentLabel: "a-95",
    agentKindSelection: "auto",
    agentKind: "codex",
    agentInterfaceMode: "interactive",
    agentCapabilities: {
      canContinueInCwd: false,
      canDetectReadyForInput: true,
      canDetectTurnComplete: true,
      canInterrupt: true,
      canReceiveNotifications: false,
      canResumeById: false,
      canSendText: true,
      canStreamSemanticEvents: false,
    },
    agentSemanticStatus: { turnState: "ready", detail: null },
    externalSessionRef: { type: "none" },
    agentPreview: {
      lastAssistantMessageAt: null,
      lastAssistantMessagePreview: null,
      lastMessageTurnId: null,
    },
    launchConfigurationId: null,
    resolvedLaunchConfiguration: null,
    ...overrides,
  }
}

function makeGraph(overrides: Partial<EpicGraph>): EpicGraph {
  return {
    epic: {
      id: 1,
      slug: "harness-interface-v0",
      name: "Harness interface v0",
      repositoryId: 1,
      rootBranch: "main",
      linearProjectId: null,
      createdAt: "2026-01-04T00:00:00Z",
    },
    tasks: [],
    agentSessions: [makeAgentSession({})],
    mergeRuns: [],
    repoExecutor: null,
    trunk: null,
    ...overrides,
  }
}

test("applyTaskAgentSessionUpdate: patches the matching session", () => {
  const graph = makeGraph({
    agentSessions: [
      makeAgentSession({
        id: 11,
        taskId: 123,
        status: "blocked",
        agentPreview: {
          lastAssistantMessageAt: "2026-01-06T12:00:00Z",
          lastAssistantMessagePreview: "Old message",
          lastMessageTurnId: "tu_old",
        },
      }),
    ],
  })

  const update: TaskAgentSessionUpdateEventData = {
    type: "task.agent_session_update",
    taskId: 123,
    agentSessionId: 11,
    agentStatus: "running",
    startedAt: "2026-01-06T12:00:00Z",
    endedAt: null,
    agentKindSelection: "codex",
    agentKind: "codex",
    agentInterfaceMode: "interactive",
    agentCapabilities: graph.agentSessions[0]!.agentCapabilities,
    agentSemanticStatus: { turnState: "busy", detail: "working" },
    externalSessionRef: {
      type: "codex_thread",
      threadId: "th_1",
      turnId: "tu_1",
    },
    agentPreview: {
      lastAssistantMessageAt: "2026-01-06T12:10:00Z",
      lastAssistantMessagePreview: "Hello world",
      lastMessageTurnId: "tu_1",
    },
  }

  const next = applyTaskAgentSessionUpdate(graph, update)!
  assert.equal(next.agentSessions[0]!.status, "running")
  assert.deepEqual(next.agentSessions[0]!.agentPreview, update.agentPreview)
  assert.deepEqual(
    next.agentSessions[0]!.externalSessionRef,
    update.externalSessionRef,
  )
  assert.deepEqual(
    next.agentSessions[0]!.agentSemanticStatus,
    update.agentSemanticStatus,
  )
})

test("applyTaskAgentSessionUpdate: preserves prior preview when update omits it", () => {
  const graph = makeGraph({
    agentSessions: [
      makeAgentSession({
        id: 12,
        taskId: 124,
        agentPreview: {
          lastAssistantMessageAt: "2026-01-06T12:10:00Z",
          lastAssistantMessagePreview: "Hello world",
          lastMessageTurnId: "tu_1",
        },
      }),
    ],
  })

  const update: TaskAgentSessionUpdateEventData = {
    type: "task.agent_session_update",
    taskId: 124,
    agentSessionId: 12,
    agentStatus: "running",
    startedAt: "2026-01-06T12:00:00Z",
    endedAt: null,
    agentKindSelection: "codex",
    agentKind: "codex",
    agentInterfaceMode: "interactive",
    agentCapabilities: graph.agentSessions[0]!.agentCapabilities,
    agentSemanticStatus: { turnState: "ready", detail: null },
    externalSessionRef: { type: "none" },
  }

  const next = applyTaskAgentSessionUpdate(graph, update)!
  assert.deepEqual(
    next.agentSessions[0]!.agentPreview,
    graph.agentSessions[0]!.agentPreview,
  )
})

test("applyTaskAgentSessionUpdate: updates cached session id when matching by taskId", () => {
  const graph = makeGraph({
    agentSessions: [
      makeAgentSession({
        id: 11,
        taskId: 123,
        status: "error",
      }),
    ],
  })

  const update: TaskAgentSessionUpdateEventData = {
    type: "task.agent_session_update",
    taskId: 123,
    agentSessionId: 99,
    agentStatus: "running",
    startedAt: "2026-01-07T12:00:00Z",
    endedAt: null,
    agentKindSelection: "codex",
    agentKind: "codex",
    agentInterfaceMode: "interactive",
    agentCapabilities: graph.agentSessions[0]!.agentCapabilities,
    agentSemanticStatus: { turnState: "busy", detail: null },
    externalSessionRef: { type: "none" },
  }

  const next = applyTaskAgentSessionUpdate(graph, update)!
  assert.equal(next.agentSessions.length, 1)
  assert.equal(next.agentSessions[0]!.taskId, 123)
  assert.equal(next.agentSessions[0]!.id, 99)
})

test("applyTaskAgentSessionUpdate: no-ops when session isn't in graph", () => {
  const graph = makeGraph({
    agentSessions: [makeAgentSession({ id: 13, taskId: 125 })],
  })

  const update: TaskAgentSessionUpdateEventData = {
    type: "task.agent_session_update",
    taskId: 999,
    agentSessionId: 999,
    agentStatus: "running",
    startedAt: null,
    endedAt: null,
    agentKindSelection: "generic",
    agentKind: "generic",
    agentInterfaceMode: "interactive",
    agentCapabilities: graph.agentSessions[0]!.agentCapabilities,
    agentSemanticStatus: { turnState: "unknown" },
    externalSessionRef: { type: "none" },
  }

  const next = applyTaskAgentSessionUpdate(graph, update)
  assert.equal(next, graph)
})
