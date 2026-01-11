import { useCallback, useMemo, useState } from "react"
import { RepoDaemonStatusChip } from "@/components/daemon/RepoDaemonStatusChip"
import {
  GitHubMenuButton,
  type GitHubStatusChange,
} from "@/components/github/GitHubMenuButton"
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

type GitHubStatusSummary = GitHubStatusChange

function linearLabel(status: LinearStatusSummary) {
  if (status.loading) {
    return "Checking…"
  }
  return status.connected ? "Connected" : "Not connected"
}

function githubLabel(status: GitHubStatusSummary) {
  if (status.loading) {
    return "Checking…"
  }
  if (!status.connected) {
    return "Not connected"
  }
  return status.warning ? "Connected (missing scopes)" : "Connected"
}

function connectionsDotClass(options: {
  repoDaemonStatus: RepoDaemonStatus
  linearStatus: LinearStatusSummary
  githubStatus: GitHubStatusSummary
}) {
  const { repoDaemonStatus, linearStatus, githubStatus } = options

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
  if (linearStatus.loading || githubStatus.loading) {
    return "bg-muted-foreground/40"
  }
  if (githubStatus.warning) {
    return "bg-amber-300"
  }
  return linearStatus.connected || githubStatus.connected
    ? "bg-emerald-400"
    : "bg-muted-foreground/40"
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
  const [githubStatus, setGitHubStatus] = useState<GitHubStatusSummary>({
    connected: false,
    loading: true,
    warning: false,
  })

  const handleLinearStatusChange = useCallback((next: LinearStatusSummary) => {
    setLinearStatus(next)
  }, [])

  const handleGitHubStatusChange = useCallback((next: GitHubStatusSummary) => {
    setGitHubStatus(next)
  }, [])

  const tooltipSummary = useMemo(
    () =>
      `Executor: ${repoDaemonStatus.label}. Linear: ${linearLabel(
        linearStatus,
      )}. GitHub: ${githubLabel(githubStatus)}.`,
    [githubStatus, linearStatus, repoDaemonStatus.label],
  )

  const dotClass = useMemo(
    () =>
      connectionsDotClass({
        repoDaemonStatus,
        linearStatus,
        githubStatus,
      }),
    [githubStatus, linearStatus, repoDaemonStatus],
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
        showStatusDot
        buttonClassName="border-0"
        onStatusChange={handleLinearStatusChange}
      />

      <GitHubMenuButton
        showStatusDot
        buttonClassName="border-0"
        onStatusChange={handleGitHubStatusChange}
      />
    </div>
  )
}
