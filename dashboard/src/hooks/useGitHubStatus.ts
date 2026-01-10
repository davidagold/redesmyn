import { useGitHubStatusQuery } from "@/api/queries"

export function useGitHubStatus() {
  const query = useGitHubStatusQuery()

  return {
    status: query.data ?? null,
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
