import { useCallback, useMemo, useState } from "react"
import { RepoDaemonStatusChip } from "@/components/daemon/RepoDaemonStatusChip"
import { LinearSyncMenuButton } from "@/components/linear/LinearSyncMenuButton"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import type { RepoDaemonStatus } from "@/lib/repo-daemon-status"

type LinearStatusSummary = {
  connected: boolean
  loading: boolean
}

function linearLabel(status: LinearStatusSummary) {
  if (status.loading) {
    return "Checking…"
  }
  return status.connected ? "Connected" : "Not connected"
}

function connectionsDotClass(options: {
  repoDaemonStatus: RepoDaemonStatus
  linearStatus: LinearStatusSummary
}) {
  const { repoDaemonStatus, linearStatus } = options

  if (
    repoDaemonStatus.kind === "offline" ||
    repoDaemonStatus.kind === "degraded"
  ) {
    return "bg-destructive/80"
  }
  if (repoDaemonStatus.kind === "stale") {
    return "bg-amber-300"
  }
  if (repoDaemonStatus.kind === "unknown") {
    return "bg-muted-foreground/50"
  }
  if (linearStatus.loading) {
    return "bg-muted-foreground/40"
  }
  return linearStatus.connected ? "bg-emerald-400" : "bg-muted-foreground/40"
}

export type ConnectionsClusterProps = {
  repoDaemonStatus: RepoDaemonStatus
  epicId?: number | null
  epicSlug: string
  onSynced?: () => Promise<void> | void
  startCommand?: string
}

export function ConnectionsCluster({
  repoDaemonStatus,
  epicId = null,
  epicSlug,
  onSynced,
  startCommand = "rn daemon run",
}: ConnectionsClusterProps) {
  const [linearStatus, setLinearStatus] = useState<LinearStatusSummary>({
    connected: false,
    loading: true,
  })

  const handleLinearStatusChange = useCallback((next: LinearStatusSummary) => {
    setLinearStatus(next)
  }, [])

  const tooltipSummary = useMemo(
    () =>
      `Executor: ${repoDaemonStatus.label}. Linear: ${linearLabel(linearStatus)}.`,
    [linearStatus, repoDaemonStatus.label],
  )

  const dotClass = useMemo(
    () =>
      connectionsDotClass({
        repoDaemonStatus,
        linearStatus,
      }),
    [linearStatus, repoDaemonStatus],
  )

  return (
    <div className="inline-flex items-center">
      <Tooltip>
        <TooltipTrigger
          render={(triggerProps) => (
            <span
              {...triggerProps}
              className={cn(
                "flex h-7 items-center pl-2 pr-1",
                triggerProps.className,
              )}
            >
              <span
                className={cn("size-2 rounded-full", dotClass)}
                aria-hidden="true"
              />
              <span className="sr-only">{tooltipSummary}</span>
            </span>
          )}
        />
        <TooltipContent side="bottom" align="center" showArrow={false}>
          {tooltipSummary}
        </TooltipContent>
      </Tooltip>

      <div className="mx-1 h-4 w-px bg-border/60" aria-hidden="true" />

      <RepoDaemonStatusChip
        status={repoDaemonStatus}
        startCommand={startCommand}
        showStatusDot={false}
        buttonClassName="border-0"
      />

      <LinearSyncMenuButton
        epicId={epicId}
        epicSlug={epicSlug}
        variant="epic"
        onSynced={onSynced}
        showStatusDot={false}
        buttonClassName="border-0"
        onStatusChange={handleLinearStatusChange}
      />
    </div>
  )
}
