import { useEpicsQuery } from "@/api/queries"

export function useEpics() {
  const query = useEpicsQuery()

  return {
    epics: query.data ?? [],
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
