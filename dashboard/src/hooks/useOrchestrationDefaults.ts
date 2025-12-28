import { useCallback, useEffect, useState } from "react"
import { fetchOrchestrationDefaults, type OrchestrationDefaults } from "@/api"

export function useOrchestrationDefaults() {
  const [defaults, setDefaults] = useState<OrchestrationDefaults | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchOrchestrationDefaults()
      setDefaults(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { defaults, loading, error, refresh }
}
