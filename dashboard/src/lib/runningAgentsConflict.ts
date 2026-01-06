import { ApiHttpError } from "@/api"

const RUNNING_AGENTS_PREFIX = "RUNNING_AGENTS:"

export function isRunningAgentsConflict(error: ApiHttpError) {
  const detail = error.detail
  return (
    typeof detail === "string" &&
    detail.trimStart().startsWith(RUNNING_AGENTS_PREFIX)
  )
}

export function runningAgentsSummary(error: ApiHttpError) {
  const detail = error.detail
  if (typeof detail !== "string") {
    return null
  }
  if (!detail.trimStart().startsWith(RUNNING_AGENTS_PREFIX)) {
    return null
  }
  const summary = detail.replace(RUNNING_AGENTS_PREFIX, "").trim()
  if (!summary) {
    return null
  }
  if (/allowRunning\s*=?/i.test(summary)) {
    return null
  }
  return summary
}
