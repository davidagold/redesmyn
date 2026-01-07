import { useHostsQuery } from "@/api/queries"

export function useHosts() {
  const query = useHostsQuery()

  return {
    hosts: query.data ?? [],
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
