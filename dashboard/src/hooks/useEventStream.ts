import { useEffect, useRef, useState } from "react"

export type StreamHelloMessage = {
  type: "hello"
  protocol: number
  serverTime: string
  lastEventId: number
}

export type StreamEvent = {
  id: number
  eventType: string
  data: Record<string, unknown>
  createdAt: string
}

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
            return
          }

          if (msg.type === "event") {
            lastEventIdRef.current = msg.event.id
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
