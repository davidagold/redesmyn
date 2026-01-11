import { useGitHubPullRequestQuery } from "@/api/queries"

export type GitHubPullRequestRef = {
  owner: string
  repo: string
  number: number
}

const PR_ID_RE = /^(?<owner>[^/]+)\/(?<repo>[^#]+)#(?<number>\d+)$/

export function parseGitHubPullRequestId(
  value: string,
): GitHubPullRequestRef | null {
  const match = PR_ID_RE.exec(value.trim())
  const groups = match?.groups
  if (!groups) {
    return null
  }
  const number = Number(groups.number)
  if (!Number.isInteger(number) || number <= 0) {
    return null
  }
  return { owner: groups.owner, repo: groups.repo, number }
}

export function gitHubPullRequestUrl(ref: GitHubPullRequestRef) {
  return `https://github.com/${ref.owner}/${ref.repo}/pull/${ref.number}`
}

export function useGitHubPullRequest(
  prId: string | null,
  options?: { enabled?: boolean },
) {
  const ref = prId ? parseGitHubPullRequestId(prId) : null

  const query = useGitHubPullRequestQuery(
    ref?.owner ?? null,
    ref?.repo ?? null,
    ref?.number ?? null,
    { enabled: options?.enabled ?? true },
  )

  return {
    ref,
    pullRequest: query.data ?? null,
    error: query.error
      ? query.error instanceof Error
        ? query.error.message
        : String(query.error)
      : null,
    loading: query.isFetching,
  }
}
