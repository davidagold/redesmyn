import { useEffect, useRef, useState } from "react"

function cursorStorageKey(epic: string) {
  return `redesmyn:event-stream:cursor:${epic}`
}

function readStoredCursor(epic: string): number | null {
  try {
    const raw = window.sessionStorage.getItem(cursorStorageKey(epic))
    if (!raw) {
      return null
    }
    const parsed = Number(raw)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return null
    }
    return parsed
  } catch {
    return null
  }
}

function storeCursor(epic: string, value: number) {
  try {
    window.sessionStorage.setItem(cursorStorageKey(epic), String(value))
  } catch {
    // ignore
  }
}

export type StreamHelloMessage = {
  type: "hello"
  protocol: number
  serverTime: string
  lastEventId: number
}

export type StreamEvent = {
  id: number
  eventType: string
  data: StreamEventData
  createdAt: string
}

export type GitCommitEventData = {
  type: "git.commit"
  nodeId: number
  branchName: string
  sha: string
  authorName?: string | null
  authorEmail?: string | null
  authoredAt?: string | null
  subject?: string | null
  agentId?: number | null
}

export type WorktreeHealthEventData = {
  type: "worktree.health"
  nodeId: number
  branchName: string
  worktreePath: string
  exists: boolean
  currentBranch?: string | null
  dirty?: boolean | null
  branchMismatch?: boolean | null
}

export type TaskAgentRunEventData = {
  type: "task.agent_run"
  runId: string
  nodeId: number
  taskId: number
  action: "start" | "restart"
  phase: "requested" | "started" | "failed"
  agentId?: number | null
  warnings?: string[]
  error?: string | null
}

export type NodeAgentSetEventData = {
  type: "node.agent_set"
  nodeId: number
  agentId: number | null
  previousAgentId: number | null
}

export type BlockSetEventData = {
  type: "block.set"
  scope: Record<string, unknown>
  policy: string
  mode: string
  release: Record<string, unknown>
  reason?: string | null
}

export type BlockClearedEventData = {
  type: "block.cleared"
  scope: Record<string, unknown>
  policy: string
  reason?: string | null
}

export type BlockAckEventData = {
  type: "block.ack"
  scope: Record<string, unknown>
  policy: string
  agentId: number
}

export type UnknownEventData = {
  type: "unknown"
  eventType: string
  data: Record<string, unknown>
}

export type StreamEventData = GitCommitEventData | WorktreeHealthEventData | TaskAgentRunEventData | NodeAgentSetEventData | BlockSetEventData | BlockClearedEventData | BlockAckEventData | UnknownEventData

export type StreamEventMessage = {
  type: "event"
  event: StreamEvent
}

export type StreamErrorMessage = {
  type: "error"
  message: string
}

export type StreamResyncMessage = {
  type: "resync"
  reason: string
}

export type StreamPongMessage = {
  type: "pong"
  serverTime: string
}

export type StreamMessage = StreamHelloMessage | StreamEventMessage | StreamErrorMessage | StreamResyncMessage | StreamPongMessage

function toWebSocketUrl(path: string) {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws"
  return `${protocol}://${window.location.host}${path}`
}

type DisconnectedEventStreamStatus = {
  state: "disconnected"
  reason?: string
}

type EventStreamStatus = { state: "idle" } | { state: "connecting" } | {
  state: "connected"
} | DisconnectedEventStreamStatus

export function useEventStream(options: {
  epic: string | null
  onEvent?: (event: StreamEvent) => void
  onResync?: (reason: string) => void
}) {
  const { epic, onEvent, onResync } = options
  const [status, setStatus] = useState<EventStreamStatus>({ state: "idle" })
  const lastEventIdRef = useRef<number | null>(null)
  const retryRef = useRef(0)

  useEffect(() => {
    if (!epic) {
      setStatus({ state: "idle" })
      return
    }

    const epicValue = epic
    lastEventIdRef.current = readStoredCursor(epicValue)

    let closed = false
    let socket: WebSocket | null = null
    let reconnectTimer: number | null = null

    function clearReconnectTimer() {
      if (reconnectTimer === null) {
        return
      }
      window.clearTimeout(reconnectTimer)
      reconnectTimer = null
    }

    function scheduleReconnect(reason?: string) {
      clearReconnectTimer()
      if (closed) {
        return
      }

      retryRef.current += 1
      const attempt = retryRef.current
      const delayMs = Math.min(10_000, 250 * 2 ** Math.min(attempt, 6))
      setStatus({ state: "disconnected", reason })
      reconnectTimer = window.setTimeout(connect, delayMs)
    }

    function connect() {
      if (closed) {
        return
      }
      clearReconnectTimer()
      setStatus({ state: "connecting" })

      const params = new URLSearchParams()
      params.set("epic", epicValue)
      if (lastEventIdRef.current !== null) {
        params.set("after_id", String(lastEventIdRef.current))
      }

      const url = toWebSocketUrl(`/v1/ws?${params.toString()}`)
      socket = new WebSocket(url)

      socket.addEventListener("open", () => {
        retryRef.current = 0
        setStatus({ state: "connected" })
      })

      socket.addEventListener("message", (e) => {
        try {
          const msg = JSON.parse(String(e.data)) as StreamMessage
          if (!msg || typeof msg !== "object" || !("type" in msg)) {
            return
          }

          if (msg.type === "hello") {
            lastEventIdRef.current = msg.lastEventId
            storeCursor(epicValue, msg.lastEventId)
            return
          }

          if (msg.type === "event") {
            lastEventIdRef.current = msg.event.id
            storeCursor(epicValue, msg.event.id)
            onEvent?.(msg.event)
            return
          }

          if (msg.type === "resync") {
            onResync?.(msg.reason)
            return
          }
        } catch {
          // Ignore malformed messages for v0.
        }
      })

      socket.addEventListener("close", () => scheduleReconnect("closed"))
      socket.addEventListener("error", () => scheduleReconnect("error"))
    }

    connect()

    return () => {
      closed = true
      clearReconnectTimer()
      try {
        socket?.close()
      } catch {
        // ignore
      }
      socket = null
    }
  }, [epic, onEvent, onResync])

  return { status }
}
