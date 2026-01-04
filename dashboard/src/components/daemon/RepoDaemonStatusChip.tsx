import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
import type { RepoDaemonStatus } from "@/lib/repo-daemon-status"

function kindDotClass(kind: RepoDaemonStatus["kind"]) {
  if (kind === "ok") {
    return "bg-emerald-400"
  }
  if (kind === "stale") {
    return "bg-amber-300"
  }
  if (kind === "unknown") {
    return "bg-muted-foreground/50"
  }
  return "bg-destructive/80"
}

function hostDotClass(host: { connected: boolean isLocal: boolean }) {
  if (host.isLocal) {
    return "bg-emerald-400"
  }
  if (host.connected) {
    return "bg-emerald-400"
  }
  return "bg-muted-foreground/40"
}

function formatAgeMs(ageMs: number) {
  if (!Number.isFinite(ageMs) || ageMs < 0) {
    return "—"
  }
  const seconds = Math.floor(ageMs / 1000)
  if (seconds < 5) {
    return "just now"
  }
  if (seconds < 60) {
    return `${seconds}s`
  }
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) {
    return `${minutes}m`
  }
  const hours = Math.floor(minutes / 60)
  if (hours < 48) {
    return `${hours}h`
  }
  const days = Math.floor(hours / 24)
  return `${days}d`
}

function formatLastSeen(nowMs: number, lastSeenAt: string | null) {
  if (!lastSeenAt) {
    return null
  }
  const ms = Date.parse(lastSeenAt)
  if (!Number.isFinite(ms)) {
    return null
  }
  return { label: formatAgeMs(nowMs - ms), raw: lastSeenAt }
}

export function RepoDaemonStatusChip(props: {
  status: RepoDaemonStatus
  startCommand?: string
}) {
  const { status, startCommand = "rn daemon run" } = props
  const nowMs = Date.now()
  const [notice, setNotice] = useState<string | null>(null)

  const primaryHost = useMemo(() => {
    if (!status.primaryHostKey) {
      return null
    }
    return (
      status.hosts.find(
        (candidate) => candidate.hostKey === status.primaryHostKey,
      ) ?? null
    )
  }, [status.hosts, status.primaryHostKey])

  const primaryLastSeen = useMemo(
    () => formatLastSeen(nowMs, primaryHost?.lastSeenAt ?? null),
    [nowMs, primaryHost?.lastSeenAt],
  )

  const tooltipSummary = useMemo(() => {
    if (status.kind === "unknown") {
      return "Daemon status unavailable."
    }
    if (status.kind === "stale") {
      return primaryLastSeen
        ? `Telemetry stale (last seen ${primaryLastSeen.label}).`
        : "Telemetry stale."
    }
    if (status.primaryIsLocal) {
      return "Local repo executor active."
    }
    if (status.primaryHostKey && status.primaryConnected) {
      return primaryLastSeen
        ? `Daemon online (last seen ${primaryLastSeen.label}).`
        : "Daemon online."
    }
    if (status.primaryHostKey && !status.primaryConnected) {
      return "Primary executor not attached."
    }
    if (status.hosts.some((host) => host.isAttached)) {
      return "Repo executor attached (no primary)."
    }
    return "No repo executor attached."
  }, [
    primaryLastSeen,
    status.hosts,
    status.kind,
    status.primaryConnected,
    status.primaryHostKey,
    status.primaryIsLocal,
  ])

  return (
    <Popover
      onOpenChange={(open) => {
        if (!open) {
          setNotice(null)
        }
      }}
    >
      <Tooltip>
        <TooltipTrigger
          render={(triggerProps) => (
            <span
              {...triggerProps}
              className={cn("inline-flex", triggerProps.className)}
            >
              <PopoverTrigger
                render={(popoverProps) => (
                  <Button
                    {...popoverProps}
                    variant="ghost"
                    size="sm"
                    className="h-7 gap-2 px-2"
                  >
                    <span className="inline-flex items-center gap-2">
                      <span
                        className={cn(
                          "size-2 rounded-full",
                          kindDotClass(status.kind),
                        )}
                        aria-hidden="true"
                      />
                      <span className="hidden sm:inline">{status.label}</span>
                      <span className="sm:hidden">Daemon</span>
                    </span>
                  </Button>
                )}
              />
            </span>
          )}
        />
        <TooltipContent side="bottom" align="center" showArrow={false}>
          {tooltipSummary}
        </TooltipContent>
      </Tooltip>

      <PopoverContent
        align="end"
        sideOffset={10}
        className="w-[22rem] space-y-3"
      >
        <div className="space-y-1">
          <div className="text-xs font-medium text-foreground">
            Repo executor
          </div>
          {status.primaryHostKey ? (
            <div className="text-xs text-muted-foreground">
              Primary:{" "}
              <span className="text-foreground/90">
                {status.primaryDisplayName ?? status.primaryHostKey}
              </span>
              {status.primaryIsLocal
                ? " (local)"
                : status.primaryConnected
                  ? " (online)"
                  : " (offline)"}
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">Primary: none</div>
          )}
          <div className="text-xs text-muted-foreground">
            Attached:{" "}
            {status.hosts.filter((host) => host.isAttached).length > 0
              ? status.hosts
                  .filter((host) => host.isAttached)
                  .map((host) => host.displayName)
                  .join(", ")
              : "none"}
          </div>
        </div>

        {status.hosts.length > 0 ? (
          <div className="space-y-1">
            <div className="text-xs font-medium text-foreground">Hosts</div>
            <div className="grid gap-1">
              {status.hosts.map((host) => (
                <div
                  key={host.hostKey}
                  className="flex items-center justify-between gap-2"
                >
                  <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-full",
                        hostDotClass(host),
                      )}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 truncate">
                      {host.displayName}
                      <span className="text-muted-foreground/70">
                        {" "}
                        <span className="font-mono">{host.hostKey}</span>
                      </span>
                    </span>
                    {host.isPrimary ? (
                      <span className="shrink-0 text-[0.625rem] text-muted-foreground/70">
                        primary
                      </span>
                    ) : null}
                    {host.isAttached ? (
                      <span className="shrink-0 text-[0.625rem] text-muted-foreground/70">
                        attached
                      </span>
                    ) : null}
                  </div>
                  <Button
                    variant="ghost"
                    size="xs"
                    className="h-5 px-2"
                    onClick={() => {
                      void copyToClipboard(host.hostKey)
                        .then(() => setNotice("Host key copied"))
                        .catch((e: unknown) =>
                          setNotice(e instanceof Error ? e.message : String(e)),
                        )
                    }}
                  >
                    Copy
                  </Button>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="space-y-1">
          <div className="text-xs font-medium text-foreground">Telemetry</div>
          {status.primaryIsLocal ? (
            <div className="text-xs text-muted-foreground">Fresh (local)</div>
          ) : status.primaryHostKey && status.primaryConnected ? (
            <div className="text-xs text-muted-foreground">
              {status.telemetryFresh ? "Fresh" : "Stale"}
              {primaryLastSeen ? (
                <>
                  {" "}
                  (last seen {primaryLastSeen.label}; threshold{" "}
                  {formatAgeMs(status.staleThresholdMs)})
                </>
              ) : (
                <> (last seen unknown)</>
              )}
            </div>
          ) : status.primaryHostKey ? (
            <div className="text-xs text-muted-foreground">
              Offline (primary not attached)
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">Unavailable</div>
          )}

          {status.telemetryFresh ? null : (
            <div className="text-xs text-muted-foreground">
              Sync projections (like “out of sync with upstream”) may be
              unavailable.
            </div>
          )}
        </div>

        {status.gitMutationsDisabledReason ? (
          <div className="rounded-md border border-border/60 bg-background/30 p-2 text-xs text-muted-foreground">
            {status.gitMutationsDisabledReason}
          </div>
        ) : null}

        <div className="space-y-2">
          <div className="text-xs font-medium text-foreground">Fix</div>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                void copyToClipboard(startCommand)
                  .then(() => setNotice("Start command copied"))
                  .catch((e: unknown) =>
                    setNotice(e instanceof Error ? e.message : String(e)),
                  )
              }}
            >
              Copy start
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                void copyToClipboard("rn dev")
                  .then(() => setNotice("`rn dev` copied"))
                  .catch((e: unknown) =>
                    setNotice(e instanceof Error ? e.message : String(e)),
                  )
              }}
            >
              Copy dev
            </Button>
          </div>
          {notice ? (
            <div className="text-xs text-muted-foreground">{notice}</div>
          ) : null}
          <div className="text-xs text-muted-foreground">
            Troubleshooting: run{" "}
            <span className="font-mono text-foreground/80">rn status</span> to
            confirm repo state, then restart the daemon if needed.
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
