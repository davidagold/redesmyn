export type NodeActivity = {
  lastCommitAt?: number
  lastWorktreeAt?: number
}

export const RECENT_ACTIVITY_WINDOW_MS = 8_000

export function isRecentActivity(ts: number | undefined, now = Date.now()) {
  return ts !== undefined && now - ts <= RECENT_ACTIVITY_WINDOW_MS
}
