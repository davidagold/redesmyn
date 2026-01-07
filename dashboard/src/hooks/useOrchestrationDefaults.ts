import { useOrchestrationDefaultsQuery } from "@/api/queries"

export function useOrchestrationDefaults() {
  const query = useOrchestrationDefaultsQuery()

  return {
    defaults: query.data ?? null,
    loading: query.isFetching,
    error: query.error
      ? query.error instanceof Error
        ? query.error.message
        : String(query.error)
      : null,
    refresh: async () => {
      await query.refetch()
    },
  }
}
