import { useCallback, useEffect, useState } from "react"
import { fetchEpics, type Epic } from "@/api"

export function useEpics() {
  const [epics, setEpics] = useState<Epic[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchEpics()
      setEpics(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { epics, error, loading, refresh }
}
