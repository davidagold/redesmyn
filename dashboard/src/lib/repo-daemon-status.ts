import type { DaemonPresence, EpicGraph, Host } from "@/api"

export const DEFAULT_TELEMETRY_STALE_MS = 90_000

export function computeGitMutationsDisabledReason(
  repoExecutor: EpicGraph["repoExecutor"] | null | undefined,
): string | null {
  if (!repoExecutor) {
    return null
  }

  const attachedHostKeys = repoExecutor.attachedHostKeys ?? []
  const primaryHostKey = repoExecutor.primaryHostKey ?? null

  if (!primaryHostKey) {
    return attachedHostKeys.length > 0
      ? "Merge/restack is disabled because no executor currently holds the primary lease for this repo."
      : "No executor is attached to this repo. Start a daemon in this repo to enable merge/restack."
  }

  if (!attachedHostKeys.includes(primaryHostKey)) {
    return "Merge/restack is disabled because the primary executor is not attached to this repo."
  }

  return null
}

export type RepoDaemonHost = {
  hostKey: string
  displayName: string
  connected: boolean
  lastSeenAt: string | null
  isLocal: boolean
  isPrimary: boolean
  isAttached: boolean
}

export type RepoDaemonStatus = {
  kind: "ok" | "stale" | "offline" | "degraded" | "unknown"
  label: string
  gitMutationsDisabledReason: string | null
  telemetryFresh: boolean
  primaryHostKey: string | null
  primaryDisplayName: string | null
  primaryConnected: boolean
  primaryIsLocal: boolean
  hosts: RepoDaemonHost[]
  staleThresholdMs: number
}

function daemonByHostKey(daemons: DaemonPresence[]) {
  const map = new Map<string, DaemonPresence>()
  for (const daemon of daemons) {
    map.set(daemon.hostKey, daemon)
  }
  return map
}

function hostByHostKey(hosts: Host[]) {
  const map = new Map<string, Host>()
  for (const host of hosts) {
    map.set(host.hostKey, host)
  }
  return map
}

function uniqueHostKeys(hostKeys: (string | null | undefined)[]) {
  const seen = new Set<string>()
  const result: string[] = []
  for (const hostKey of hostKeys) {
    if (!hostKey) {
      continue
    }
    if (seen.has(hostKey)) {
      continue
    }
    seen.add(hostKey)
    result.push(hostKey)
  }
  return result
}

export function computeRepoDaemonStatus(options: {
  repoExecutor: EpicGraph["repoExecutor"] | null | undefined
  daemons: DaemonPresence[]
  hosts?: Host[]
  nowMs?: number
  staleThresholdMs?: number
}): RepoDaemonStatus {
  const {
    repoExecutor,
    daemons,
    hosts: hostRows = [],
    nowMs = Date.now(),
    staleThresholdMs = DEFAULT_TELEMETRY_STALE_MS,
  } = options

  if (!repoExecutor) {
    return {
      kind: "unknown",
      label: "Executor",
      gitMutationsDisabledReason: null,
      telemetryFresh: false,
      primaryHostKey: null,
      primaryDisplayName: null,
      primaryConnected: false,
      primaryIsLocal: false,
      hosts: [],
      staleThresholdMs,
    }
  }

  const attachedHostKeys = repoExecutor.attachedHostKeys ?? []
  const primaryHostKey = repoExecutor.primaryHostKey ?? null
  const daemonMap = daemonByHostKey(daemons)
  const hostMap = hostByHostKey(hostRows)

  const primaryDaemon = primaryHostKey
    ? (daemonMap.get(primaryHostKey) ?? null)
    : null
  const primaryHost = primaryHostKey
    ? (hostMap.get(primaryHostKey) ?? null)
    : null
  const primaryConnected = primaryDaemon?.connected ?? false
  const primaryAttached = primaryHostKey
    ? attachedHostKeys.includes(primaryHostKey)
    : false
  const primaryIsLocal = primaryAttached && !primaryConnected

  const primaryDisplayName =
    primaryHost?.displayName ??
    primaryDaemon?.displayName ??
    (primaryHostKey ? primaryHostKey : null)

  let telemetryFresh = false
  if (primaryHostKey && primaryAttached) {
    if (primaryIsLocal) {
      telemetryFresh = true
    } else {
      const lastSeenAt = primaryDaemon?.lastSeenAt ?? null
      const lastSeenMs = lastSeenAt ? Date.parse(lastSeenAt) : Number.NaN
      telemetryFresh =
        Number.isFinite(lastSeenMs) && nowMs - lastSeenMs <= staleThresholdMs
    }
  }

  const gitMutationsDisabledReason =
    computeGitMutationsDisabledReason(repoExecutor)

  const relevantHostKeys = uniqueHostKeys([...attachedHostKeys, primaryHostKey])
  const repoHosts: RepoDaemonHost[] = relevantHostKeys.map((hostKey) => {
    const daemon = daemonMap.get(hostKey) ?? null
    const host = hostMap.get(hostKey) ?? null
    const connected = daemon?.connected ?? false
    const isAttached = attachedHostKeys.includes(hostKey)
    const isPrimary = primaryHostKey === hostKey
    const isLocal = isAttached && !connected
    const displayName = host?.displayName ?? daemon?.displayName ?? hostKey
    return {
      hostKey,
      displayName,
      connected,
      lastSeenAt: daemon?.lastSeenAt ?? null,
      isLocal,
      isPrimary,
      isAttached,
    }
  })

  let kind: RepoDaemonStatus["kind"] = "ok"
  let label = "Executor online"

  if (gitMutationsDisabledReason) {
    kind = attachedHostKeys.length > 0 ? "degraded" : "offline"
    label = !primaryHostKey
      ? attachedHostKeys.length > 0
        ? "No primary executor"
        : "Executor offline"
      : "Primary not attached"
  } else if (!telemetryFresh && !primaryIsLocal) {
    kind = "stale"
    label = "Telemetry stale"
  } else if (primaryIsLocal) {
    label = "Local"
  }

  return {
    kind,
    label,
    gitMutationsDisabledReason,
    telemetryFresh,
    primaryHostKey,
    primaryDisplayName,
    primaryConnected,
    primaryIsLocal,
    hosts: repoHosts,
    staleThresholdMs,
  }
}
