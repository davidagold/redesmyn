import { useCallback, useEffect, useRef, useState } from "react"
import { fetchDaemons, type DaemonPresence } from "@/api"

export function useDaemons(options?: { pollIntervalMs?: number }) {
  const pollIntervalMs = options?.pollIntervalMs ?? 15_000
  const pollTimerRef = useRef<number | null>(null)

  const [daemons, setDaemons] = useState<DaemonPresence[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchDaemons()
      setDaemons(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  useEffect(() => {
    if (!Number.isFinite(pollIntervalMs) || pollIntervalMs <= 0) {
      return
    }

    function clearTimer() {
      if (pollTimerRef.current === null) {
        return
      }
      window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }

    pollTimerRef.current = window.setInterval(() => {
      if (typeof document !== "undefined" && document.hidden) {
        return
      }
      void refresh()
    }, pollIntervalMs)

    return clearTimer
  }, [pollIntervalMs, refresh])

  return { daemons, error, loading, refresh }
}
