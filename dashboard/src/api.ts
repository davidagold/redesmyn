export type PauseStatus = {
  mode: string
  scope: string
  reason: string | null
}

export type ApiStatus = {
  repoRoot: string
  dbPath: string
  defaultBranch: string | null
  pause: PauseStatus | null
}

export async function fetchStatus(): Promise<ApiStatus> {
  const response = await fetch("/api/status", {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    throw new Error(`GET /api/status failed (${response.status})`)
  }
  return response.json() as Promise<ApiStatus>
}
