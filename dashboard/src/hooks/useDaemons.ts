import { useDaemonsQuery } from "@/api/queries"

export function useDaemons(options?: { pollIntervalMs?: number }) {
  const query = useDaemonsQuery(options)

  return {
    daemons: query.data ?? [],
    error: query.error
      ? query.error instanceof Error
        ? query.error.message
        : String(query.error)
      : null,
    loading: query.isFetching,
    refresh: async () => {
      await query.refetch()
    },
  }
}
