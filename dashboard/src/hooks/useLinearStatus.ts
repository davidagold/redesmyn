import { useCallback, useEffect, useState } from "react"
import { fetchLinearStatus, type LinearStatus } from "@/api"

export function useLinearStatus() {
  const [status, setStatus] = useState<LinearStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchLinearStatus()
      setStatus(data)
      return data
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus(null)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { status, error, loading, refresh }
}
