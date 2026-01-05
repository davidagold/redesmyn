import { useMemo } from "react"
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
import { cn } from "@/lib/utils"
import type { RepoDaemonStatus } from "@/lib/repo-daemon-status"

type HostDotInput = {
  connected: boolean
  isLocal: boolean
}

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

function hostDotClass(host: HostDotInput) {
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
  showStatusDot?: boolean
  buttonClassName?: string
}) {
  const {
    status,
    startCommand = "rn daemon run",
    showStatusDot = true,
    buttonClassName,
  } = props
  const nowMs = Date.now()

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

  const attachedHosts = useMemo(
    () => status.hosts.filter((host) => host.isAttached),
    [status.hosts],
  )

  const primaryAttached = primaryHost?.isAttached ?? false

  const tooltipSummary = useMemo(() => {
    if (status.kind === "unknown") {
      return "Executor status unavailable."
    }
    if (status.kind === "stale") {
      return primaryLastSeen
        ? `Telemetry stale (last seen ${primaryLastSeen.label}).`
        : "Telemetry stale."
    }
    if (status.primaryIsLocal) {
      return "Local executor active."
    }
    if (status.kind === "ok") {
      return "Executor online."
    }
    if (status.kind === "offline") {
      return "No executor attached."
    }
    if (status.kind === "degraded") {
      return status.label
    }
    return "Executor status unavailable."
  }, [primaryLastSeen, status.kind, status.label, status.primaryIsLocal])

  return (
    <Popover>
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
                    className={cn("h-7 gap-2 px-2", buttonClassName)}
                  >
                    <span className="inline-flex items-center gap-2">
                      {showStatusDot ? (
                        <span
                          className={cn(
                            "size-2 rounded-full",
                            kindDotClass(status.kind),
                          )}
                          aria-hidden="true"
                        />
                      ) : null}
                      <span className="hidden sm:inline">{status.label}</span>
                      <span className="sm:hidden">Executor</span>
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
          <div className="text-xs font-medium text-foreground">Executor</div>
          <div className="text-xs text-muted-foreground">
            <span className="text-muted-foreground/70">
              Primary executor (lease):
            </span>{" "}
            <span className="text-foreground/90">
              {status.primaryHostKey
                ? (status.primaryDisplayName ?? status.primaryHostKey)
                : "none"}
            </span>{" "}
            {status.primaryHostKey ? (
              <span className="text-muted-foreground/70">
                {status.primaryIsLocal
                  ? "(local)"
                  : primaryAttached
                    ? "(online)"
                    : status.primaryConnected
                      ? "(connected; not attached)"
                      : "(not attached)"}
              </span>
            ) : null}
          </div>
          {attachedHosts.length > 0 &&
          !(
            attachedHosts.length === 1 &&
            primaryHost &&
            attachedHosts[0]?.hostKey === primaryHost.hostKey
          ) ? (
            <div className="text-xs text-muted-foreground">
              Attached executors:{" "}
              <span className="text-foreground/90">
                {attachedHosts.map((host) => host.displayName).join(", ")}
              </span>
            </div>
          ) : null}
          <div className="text-xs text-muted-foreground">
            The primary executor holds the repo lease used for merge/restack.
            “Attached” means a daemon has this repo attached (it can run
            repo-scoped commands and report telemetry).
          </div>
        </div>

        {status.hosts.length > 0 ? (
          <div className="space-y-1">
            <div className="text-xs font-medium text-foreground">Executors</div>
            <div className="grid gap-1">
              {status.hosts.map((host) => (
                <div
                  key={host.hostKey}
                  className="flex min-w-0 items-center justify-between gap-2"
                >
                  <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-full",
                        hostDotClass(host),
                      )}
                      aria-hidden="true"
                    />
                    <span className="min-w-0 truncate">{host.displayName}</span>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    {host.isPrimary ? (
                      <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/70">
                        primary
                      </span>
                    ) : null}
                    {host.isAttached ? (
                      <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/70">
                        attached
                      </span>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="space-y-1">
          <div className="text-xs font-medium text-foreground">Git actions</div>
          <div className="text-xs text-muted-foreground">
            {status.gitMutationsDisabledReason ? (
              <>
                <span className="text-foreground/90">Disabled.</span>{" "}
                {status.gitMutationsDisabledReason}
              </>
            ) : (
              <span className="text-muted-foreground">Enabled.</span>
            )}
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-xs font-medium text-foreground">Telemetry</div>
          {status.primaryIsLocal ? (
            <div className="text-xs text-muted-foreground">Fresh (local)</div>
          ) : status.primaryHostKey &&
            primaryAttached &&
            status.primaryConnected ? (
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
          ) : status.primaryHostKey && primaryAttached ? (
            <div className="text-xs text-muted-foreground">
              Attached (offline)
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">Unavailable</div>
          )}

          {status.telemetryFresh ? null : (
            <div className="text-xs text-muted-foreground">
              Sync projections stay visible, but may be stale.
            </div>
          )}
        </div>

        <div className="space-y-2">
          <div className="text-xs font-medium text-foreground">
            Troubleshooting
          </div>
          <div className="text-xs text-muted-foreground">
            If no executor is attached, start one with{" "}
            <span className="font-mono text-foreground/80">{startCommand}</span>
            . If things look wrong, run{" "}
            <span className="font-mono text-foreground/80">rn status</span> then
            restart the daemon.
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
