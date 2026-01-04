import test from "node:test"
import assert from "node:assert/strict"
import {
  DEFAULT_TELEMETRY_STALE_MS,
  computeRepoDaemonStatus,
} from "../src/lib/repo-daemon-status.ts"
import type { DaemonPresence, EpicGraph } from "../src/api.ts"

type RepoExecutor = NonNullable<EpicGraph["repoExecutor"]>

function makeDaemonPresence(
  overrides: Partial<DaemonPresence> & Pick<DaemonPresence, "hostKey">,
): DaemonPresence {
  return {
    hostKey: overrides.hostKey,
    displayName: null,
    capabilities: {},
    attachedRepos: [],
    connected: false,
    lastSeenAt: null,
    connectedAt: null,
    disconnectedAt: null,
    createdAt: "2026-01-04T00:00:00Z",
    updatedAt: "2026-01-04T00:00:00Z",
    ...overrides,
  }
}

test("computeRepoDaemonStatus: unknown without repo executor info", () => {
  const status = computeRepoDaemonStatus({
    repoExecutor: null,
    daemons: [],
    nowMs: Date.parse("2026-01-04T00:00:00Z"),
  })
  assert.equal(status.kind, "unknown")
  assert.equal(status.telemetryFresh, false)
  assert.equal(status.gitMutationsDisabledReason, null)
})

test("computeRepoDaemonStatus: local executor counts as fresh", () => {
  const nowMs = Date.parse("2026-01-04T00:00:00Z")
  const repoExecutor: RepoExecutor = {
    workspaceId: "w",
    repoId: "r",
    primaryHostKey: "local",
    attachedHostKeys: ["local"],
  }
  const daemons = [
    makeDaemonPresence({
      hostKey: "local",
      displayName: "Local",
      connected: false,
      lastSeenAt: "2025-01-01T00:00:00Z",
    }),
  ]

  const status = computeRepoDaemonStatus({ repoExecutor, daemons, nowMs })
  assert.equal(status.telemetryFresh, true)
  assert.equal(status.gitMutationsDisabledReason, null)
  assert.equal(status.label, "Local executor")
})

test("computeRepoDaemonStatus: no primary disables git actions", () => {
  const repoExecutor: RepoExecutor = {
    workspaceId: "w",
    repoId: "r",
    primaryHostKey: null,
    attachedHostKeys: ["host-a"],
  }
  const status = computeRepoDaemonStatus({
    repoExecutor,
    daemons: [makeDaemonPresence({ hostKey: "host-a", connected: true })],
    nowMs: Date.parse("2026-01-04T00:00:00Z"),
  })
  assert.equal(status.kind, "degraded")
  assert.equal(status.gitMutationsDisabledReason?.includes("No primary"), true)
})

test("computeRepoDaemonStatus: primary not attached disables git actions", () => {
  const repoExecutor: RepoExecutor = {
    workspaceId: "w",
    repoId: "r",
    primaryHostKey: "host-a",
    attachedHostKeys: [],
  }
  const status = computeRepoDaemonStatus({
    repoExecutor,
    daemons: [makeDaemonPresence({ hostKey: "host-a", connected: false })],
    nowMs: Date.parse("2026-01-04T00:00:00Z"),
  })
  assert.equal(status.kind, "offline")
  assert.equal(
    status.gitMutationsDisabledReason?.includes("is not attached"),
    true,
  )
})

test("computeRepoDaemonStatus: connected daemon uses lastSeen to determine freshness", () => {
  const nowMs = Date.parse("2026-01-04T00:01:40Z")
  const repoExecutor: RepoExecutor = {
    workspaceId: "w",
    repoId: "r",
    primaryHostKey: "host-a",
    attachedHostKeys: ["host-a"],
  }

  const fresh = computeRepoDaemonStatus({
    repoExecutor,
    daemons: [
      makeDaemonPresence({
        hostKey: "host-a",
        connected: true,
        lastSeenAt: "2026-01-04T00:01:00Z",
      }),
    ],
    nowMs,
    staleThresholdMs: DEFAULT_TELEMETRY_STALE_MS,
  })
  assert.equal(fresh.telemetryFresh, true)
  assert.equal(fresh.kind, "ok")

  const stale = computeRepoDaemonStatus({
    repoExecutor,
    daemons: [
      makeDaemonPresence({
        hostKey: "host-a",
        connected: true,
        lastSeenAt: "2026-01-04T00:00:00Z",
      }),
    ],
    nowMs,
    staleThresholdMs: 60_000,
  })
  assert.equal(stale.telemetryFresh, false)
  assert.equal(stale.kind, "stale")
  assert.equal(stale.label, "Telemetry stale")
})
