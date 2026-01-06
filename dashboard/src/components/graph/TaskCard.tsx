import { useEffect, useRef, useState, type ComponentProps } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ResourceBadge } from "@/components/ui/resource-badge"
import {
  ApiHttpError,
  mergeTask,
  restackTask,
  resumeMergeRun,
  restartTaskAgent,
  setTaskMergeReady,
  startTaskAgent,
  stopTaskAgent,
} from "@/api"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
import type { AgentSession, GraphNode, MergeRun, Task } from "@/lib/graph-utils"
import { getRebaseRemediation } from "@/lib/merge-remediation"
import {
  isRunningAgentsConflict,
  runningAgentsSummary,
} from "@/lib/runningAgentsConflict"
import { AgentStatusIcon } from "@/components/agents/AgentStatusIcon"
import { LinearIcon } from "@/components/linear/LinearIcon"
import { ProceedAnywayDialog } from "@/components/ui/proceed-anyway-dialog"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  ChevronDown,
  ChevronUp,
  ChevronRight,
  EllipsisVertical,
  AlertTriangle,
  GitBranch,
  GitMerge,
  Layers,
  Loader2,
  MessageSquareText,
  Pause,
  Play,
  RotateCcw,
  Square,
  Terminal,
  X,
} from "lucide-react"

interface TaskCardProps {
  node: GraphNode
  task?: Task
  agentSession?: AgentSession
  mergeRun?: MergeRun
  blockingMergeRun?: MergeRun
  branchLabel: string
  branchLabelProvisional?: boolean
  gitMutationsDisabledReason?: string | null
  stackProjectionsFresh: boolean
  harnessCommand: string
  detach: boolean
  isSelected: boolean
  isHighlighted?: boolean
  onSelect: (options: { additive: boolean }) => void
  onRequestRefresh?: () => void
}

interface MergeAllowRunningPrompt {
  kind: "merge"
  cascade: boolean
  restackMode: "strict" | "merge_then_restack"
  detail?: string | null
}

interface ResumeAllowRunningPrompt {
  kind: "resume"
  runId: string
  operation: "merge" | "restack"
  detail?: string | null
}

interface RestackAllowRunningPrompt {
  kind: "restack"
  scope: "descendants" | "spine"
  detail?: string | null
}

type AllowRunningPrompt = MergeAllowRunningPrompt | ResumeAllowRunningPrompt | RestackAllowRunningPrompt

function linearStateTypeFromTaskState(state: Task["state"]) {
  switch (state) {
    case "todo":
      return "unstarted"
    case "in_progress":
      return "started"
    case "blocked":
      return "blocked"
    case "done":
      return "completed"
  }
}

function linearStateLabel(stateName: string | null, stateType: string | null) {
  const name = stateName?.trim()
  if (name) {
    return name
  }

  const normalized = stateType?.trim().toLowerCase() ?? ""
  if (normalized === "unstarted") {
    return "Todo"
  }
  if (normalized === "started") {
    return "In progress"
  }
  if (normalized === "blocked") {
    return "Blocked"
  }
  if (normalized === "completed") {
    return "Done"
  }
  return "Unknown"
}

function formatObservedAt(value: string | null) {
  if (!value) {
    return null
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return null
  }
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "numeric",
    minute: "2-digit",
  }).format(date)
}

function statusSummary(
  task: Task | undefined,
  agentSession: AgentSession | undefined,
) {
  if (task?.state === "blocked") {
    return "blocked"
  }
  if (task?.state === "done") {
    return "done"
  }
  if (!agentSession) {
    return "not started"
  }
  return agentSession.status
}

function agentStatusTooltip(
  task: Task | undefined,
  agentSession: AgentSession | undefined,
) {
  if (!agentSession) {
    return "Agent not started"
  }

  if (agentSession.status === "running") {
    return "Agent running"
  }
  if (agentSession.status === "blocked") {
    return "Agent blocked"
  }
  if (agentSession.status === "error") {
    return "Agent error"
  }
  if (agentSession.status === "stopped") {
    return "Agent stopped"
  }

  const summary = statusSummary(task, agentSession)
  return summary === "not started" ? "Agent not started" : `Agent ${summary}`
}

export function TaskCard({
  node,
  task,
  mergeRun,
  blockingMergeRun,
  agentSession,
  branchLabel,
  branchLabelProvisional = false,
  gitMutationsDisabledReason,
  stackProjectionsFresh,
  harnessCommand,
  detach,
  isSelected,
  isHighlighted = false,
  onSelect,
  onRequestRefresh,
}: TaskCardProps) {
  const now = Date.now()
  const stackInSync = stackProjectionsFresh ? (node.stackInSync ?? null) : null
  const outOfSync = stackInSync === false
  const mergeRunStatus = mergeRun?.status ?? null
  const mergeRunOperation = mergeRun?.operation ?? "merge"
  const mergeRunBlocked = mergeRunStatus === "blocked"
  const rebaseRemediation = getRebaseRemediation(mergeRun, node.branchName)
  const mergeRunBlockedRebase = rebaseRemediation !== null
  const blockingRebaseRemediation = getRebaseRemediation(
    blockingMergeRun,
    node.branchName,
  )
  const blockingMergeRunBlockedRebase = blockingRebaseRemediation !== null
  const gitDisabledReason = gitMutationsDisabledReason ?? null
  const harnessKind = agentSession?.harnessProfileId?.split("/")[0] ?? null
  const linearIssueId = task?.linearIssueId ?? null
  const linearIdentifier = task?.linearIdentifier ?? null
  const linearPillLabel = linearIdentifier ?? "Linear"
  const linearStateType = task?.linearStateType ?? null
  const linearStateName = task?.linearStateName ?? null
  const linearStateObservedAt = task?.linearStateObservedAt ?? null
  const linearObservedAgeMs = linearStateObservedAt
    ? now - new Date(linearStateObservedAt).getTime()
    : null
  const linearObservedOld =
    linearObservedAgeMs !== null &&
    Number.isFinite(linearObservedAgeMs) &&
    linearObservedAgeMs > 30 * 60 * 1000
  const linearObservedAtLabel = formatObservedAt(linearStateObservedAt)

  const [pendingAction, setPendingAction] =
    useState<"start" | "stop" | "restart" | "attach" | null>(null)
  const [pendingMerge, setPendingMerge] =
    useState<"ready" | "merge" | "mergeStack" | "restack" | "resume" | null>(
      null,
    )
  const [actionsMenuOpen, setActionsMenuOpen] = useState(false)
  const [mergeReady, setMergeReady] = useState(Boolean(task?.mergeReadyAt))
  const mergeReadyRequestInFlightRef = useRef(false)
  const mergeReadyRefreshTimersRef = useRef<number[]>([])

  function clearMergeReadyRefreshTimers() {
    for (const timer of mergeReadyRefreshTimersRef.current) {
      window.clearTimeout(timer)
    }
    mergeReadyRefreshTimersRef.current = []
  }

  useEffect(() => {
    return () => {
      clearMergeReadyRefreshTimers()
    }
  }, [])
  const expectedLinearStateType =
    task === undefined
      ? null
      : task.state === "done"
        ? "completed"
        : mergeReady
          ? "started"
          : linearStateTypeFromTaskState(task.state)
  const linearStateMatches =
    task !== undefined &&
    linearIssueId !== null &&
    linearStateType !== null &&
    expectedLinearStateType !== null &&
    linearStateType.toLowerCase() === expectedLinearStateType.toLowerCase()
  const linearStateMismatch =
    task !== undefined &&
    linearIssueId !== null &&
    linearStateType !== null &&
    expectedLinearStateType !== null &&
    linearStateType.toLowerCase() !== expectedLinearStateType.toLowerCase()
  const linearPillBorderClass = (() => {
    if (task?.state === "done") {
      return "border-green-950/80"
    }
    if (mergeReady && linearStateMatches) {
      return "border-emerald-400/70"
    }

    const normalized = linearStateType?.trim().toLowerCase() ?? ""
    if (normalized === "unstarted") {
      return "border-foreground/10 border-dashed"
    }
    if (normalized === "started") {
      return "border-foreground/10"
    }
    if (normalized === "blocked") {
      return "border-amber-400/80"
    }
    if (normalized === "completed") {
      return "border-green-950/70"
    }
    return "border-border/60"
  })()
  const [refreshPendingAfterMenuClose, setRefreshPendingAfterMenuClose] =
    useState(false)
  const [actionErrorExpanded, setActionErrorExpanded] = useState(false)
  const [actionError, setActionError] = useState<{
    title: string
    summary: string
    raw: string
  } | null>(null)
  const [allowRunningPrompt, setAllowRunningPrompt] =
    useState<AllowRunningPrompt | null>(null)
  const [allowRunningConfirming, setAllowRunningConfirming] = useState(false)
  const [blockedRebaseExpanded, setBlockedRebaseExpanded] = useState(false)

  useEffect(() => {
    if (!blockingMergeRunBlockedRebase) {
      setBlockedRebaseExpanded(false)
    }
  }, [blockingMergeRunBlockedRebase])

  function clearActionError() {
    setActionError(null)
    setActionErrorExpanded(false)
  }

  function setActionErrorFromException(
    actionLabel: string,
    exception: unknown,
  ) {
    setActionErrorExpanded(false)

    if (exception instanceof ApiHttpError) {
      const summary = exception.detail?.trim() || exception.message
      setActionError({
        title: `${actionLabel} failed`,
        summary,
        raw: exception.message,
      })
      return
    }

    const raw =
      exception instanceof Error ? exception.message : String(exception)
    const httpDetailMatch = raw.match(/failed \\(\\d+\\): (.+)$/)
    const summary = httpDetailMatch?.[1]?.trim() || raw
    setActionError({
      title: `${actionLabel} failed`,
      summary,
      raw,
    })
  }

  const agentStatus = agentSession?.status ?? null
  const taskId = task?.id ?? null
  const canResumeMerge = mergeRunStatus === "resumable"

  useEffect(() => {
    setMergeReady(Boolean(task?.mergeReadyAt))
  }, [taskId, task?.mergeReadyAt])

  useEffect(() => {
    if (!refreshPendingAfterMenuClose) {
      return
    }
    if (actionsMenuOpen) {
      return
    }
    if (!onRequestRefresh) {
      return
    }
    onRequestRefresh()
    setRefreshPendingAfterMenuClose(false)
  }, [actionsMenuOpen, onRequestRefresh, refreshPendingAfterMenuClose])

  const quickActionsEnabled =
    taskId !== null &&
    !!onRequestRefresh &&
    task?.state !== "blocked" &&
    task?.state !== "done"

  const isRunning = agentStatus === "running" || agentStatus === "blocked"
  const canStart =
    (!agentSession || agentStatus === "stopped") && harnessCommand.trim()
  const canRestart = isRunning || agentStatus === "error"
  const canMerge =
    taskId !== null && task?.state !== "blocked" && task?.state !== "done"

  async function handleAttach() {
    if (taskId === null) {
      return
    }
    setPendingAction("attach")
    clearActionError()
    try {
      await copyToClipboard(`rn agent attach --task ${taskId}`)
    } catch (e) {
      setActionErrorFromException("Copy attach command", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function handleStart() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    if (!harnessCommand.trim()) {
      return
    }
    setPendingAction("start")
    clearActionError()
    try {
      await startTaskAgent(taskId, { harness: harnessCommand, detach })
      onRequestRefresh()
    } catch (e) {
      setActionErrorFromException("Start agent", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function handleStop() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setPendingAction("stop")
    clearActionError()
    try {
      await stopTaskAgent(taskId)
      onRequestRefresh()
    } catch (e) {
      setActionErrorFromException("Stop agent", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function handleRestart() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setPendingAction("restart")
    clearActionError()
    try {
      await restartTaskAgent(taskId, { detach: true })
      onRequestRefresh()
    } catch (e) {
      setActionErrorFromException("Restart agent", e)
    } finally {
      setPendingAction(null)
    }
  }

  async function handleToggleMergeReady(next: boolean) {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    if (next && !task?.branchName) {
      return
    }
    if (mergeReadyRequestInFlightRef.current) {
      return
    }
    const previous = mergeReady
    setPendingMerge("ready")
    setMergeReady(next)
    clearActionError()
    clearMergeReadyRefreshTimers()
    mergeReadyRequestInFlightRef.current = true
    try {
      await setTaskMergeReady(taskId, next)
      if (actionsMenuOpen) {
        setRefreshPendingAfterMenuClose(true)
      } else {
        onRequestRefresh()
      }
      if (next) {
        mergeReadyRefreshTimersRef.current.push(
          window.setTimeout(() => onRequestRefresh(), 1250),
        )
        mergeReadyRefreshTimersRef.current.push(
          window.setTimeout(() => onRequestRefresh(), 3500),
        )
      }
    } catch (e) {
      setMergeReady(previous)
      setActionErrorFromException("Set merge readiness", e)
    } finally {
      mergeReadyRequestInFlightRef.current = false
      setPendingMerge(null)
    }
  }

  async function handleMerge({ cascade }: { cascade: boolean }) {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setRefreshPendingAfterMenuClose(false)
    setPendingMerge(cascade ? "mergeStack" : "merge")
    clearActionError()

    const restackMode: "strict" | "merge_then_restack" = "strict"
    const actionLabel = cascade ? "Merge and Restack" : "Merge"
    try {
      await mergeTask(taskId, { cascade, restackMode })
      onRequestRefresh()
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "merge",
            cascade,
            restackMode,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(actionLabel, e)
      return
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleMergeThenRestack() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setRefreshPendingAfterMenuClose(false)
    setPendingMerge("mergeStack")
    clearActionError()
    const actionLabel = "Merge then Restack"
    const restackMode: "strict" | "merge_then_restack" = "merge_then_restack"
    try {
      await mergeTask(taskId, { cascade: true, restackMode })
      onRequestRefresh()
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "merge",
            cascade: true,
            restackMode,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(actionLabel, e)
      return
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleResumeMerge() {
    if (!mergeRun?.runId || !onRequestRefresh) {
      return
    }
    setRefreshPendingAfterMenuClose(false)
    setPendingMerge("resume")
    clearActionError()
    try {
      await resumeMergeRun(mergeRun.runId, {})
      onRequestRefresh()
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "resume",
            runId: mergeRun.runId,
            operation: mergeRunOperation,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(
        mergeRunOperation === "restack" ? "Resume restack" : "Resume merge",
        e,
      )
    } finally {
      setPendingMerge(null)
    }
  }

  async function handleRestack({ scope }: { scope: "descendants" | "spine" }) {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setRefreshPendingAfterMenuClose(false)
    setPendingMerge("restack")
    clearActionError()
    const actionLabel = "Restack"
    try {
      await restackTask(taskId, { scope })
      onRequestRefresh()
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409) {
        if (isRunningAgentsConflict(e)) {
          setAllowRunningPrompt({
            kind: "restack",
            scope,
            detail: runningAgentsSummary(e),
          })
          return
        }
      }
      setActionErrorFromException(actionLabel, e)
      return
    } finally {
      setPendingMerge(null)
    }
  }

  async function confirmAllowRunning() {
    if (!allowRunningPrompt || !onRequestRefresh) {
      return
    }

    setAllowRunningConfirming(true)
    clearActionError()

    const actionLabel =
      allowRunningPrompt.kind === "merge"
        ? allowRunningPrompt.cascade
          ? allowRunningPrompt.restackMode === "merge_then_restack"
            ? "Merge then Restack"
            : "Merge and Restack"
          : "Merge"
        : allowRunningPrompt.kind === "restack"
          ? "Restack"
          : allowRunningPrompt.operation === "restack"
            ? "Resume restack"
            : "Resume merge"

    try {
      if (allowRunningPrompt.kind === "merge") {
        if (taskId === null) {
          throw new Error("Task id missing for merge confirmation.")
        }
        setPendingMerge(allowRunningPrompt.cascade ? "mergeStack" : "merge")
        await mergeTask(taskId, {
          cascade: allowRunningPrompt.cascade,
          restackMode: allowRunningPrompt.restackMode,
          allowRunning: true,
        })
      } else if (allowRunningPrompt.kind === "restack") {
        if (taskId === null) {
          throw new Error("Task id missing for restack confirmation.")
        }
        setPendingMerge("restack")
        await restackTask(taskId, {
          scope: allowRunningPrompt.scope,
          allowRunning: true,
        })
      } else {
        setPendingMerge("resume")
        await resumeMergeRun(allowRunningPrompt.runId, { allowRunning: true })
      }
      setAllowRunningPrompt(null)
      onRequestRefresh()
    } catch (e) {
      setAllowRunningPrompt(null)
      setActionErrorFromException(actionLabel, e)
    } finally {
      setPendingMerge(null)
      setAllowRunningConfirming(false)
    }
  }

  function renderQuickActionButton({
    tooltip,
    disabledReason,
    ...props
  }: { tooltip: string } & ComponentProps<typeof Button>) {
    if (disabledReason) {
      return <Button {...props} disabledReason={disabledReason} />
    }

    return (
      <Tooltip>
        <TooltipTrigger
          render={(triggerProps) => (
            <Button
              {...triggerProps}
              {...props}
              className={cn(props.className, triggerProps.className)}
            />
          )}
        />
        <TooltipContent side="bottom" sideOffset={10} showArrow={false}>
          {tooltip}
        </TooltipContent>
      </Tooltip>
    )
  }

  const quickActions =
    quickActionsEnabled && taskId !== null ? (
      isRunning ? (
        <>
          {renderQuickActionButton({
            tooltip: "Copy attach command",
            variant: "ghost",
            size: "icon-xs",
            "aria-label": "Copy attach command",
            disabledReason:
              pendingAction !== null ? "Action in progress" : null,
            onClick: (e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleAttach()
            },
            children: <Terminal className="size-3" />,
          })}
          {renderQuickActionButton({
            tooltip: "Restart agent",
            variant: "ghost",
            size: "icon-xs",
            "aria-label": "Restart agent",
            disabledReason:
              pendingAction !== null ? "Action in progress" : null,
            onClick: (e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleRestart()
            },
            children: <RotateCcw className="size-3" />,
          })}
          {renderQuickActionButton({
            tooltip: "Stop agent",
            variant: "ghost",
            size: "icon-xs",
            className: "text-destructive hover:bg-destructive/10",
            "aria-label": "Stop agent",
            disabledReason:
              pendingAction !== null ? "Action in progress" : null,
            onClick: (e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleStop()
            },
            children: <Square className="size-3" />,
          })}
        </>
      ) : canRestart ? (
        renderQuickActionButton({
          tooltip: "Restart agent",
          variant: "ghost",
          size: "icon-xs",
          "aria-label": "Restart agent",
          disabledReason: pendingAction !== null ? "Action in progress" : null,
          onClick: (e) => {
            e.preventDefault()
            e.stopPropagation()
            void handleRestart()
          },
          children: <RotateCcw className="size-3" />,
        })
      ) : canStart ? (
        renderQuickActionButton({
          tooltip: "Start agent",
          variant: "ghost",
          size: "icon-xs",
          "aria-label": "Start agent",
          disabledReason:
            pendingAction !== null
              ? "Action in progress"
              : !harnessCommand.trim()
                ? "Set a harness command in Configure"
                : null,
          onClick: (e) => {
            e.preventDefault()
            e.stopPropagation()
            void handleStart()
          },
          children: <Play className="size-3" />,
        })
      ) : null
    ) : null

  const tooltip = agentStatusTooltip(task, agentSession)

  const rebaseAttachCommand = rebaseRemediation?.attachCommand ?? null
  const rebaseRemediationMessage = rebaseRemediation?.message ?? null
  const blockingAttachCommand = blockingRebaseRemediation?.attachCommand ?? null
  const blockingRemediationMessage = blockingRebaseRemediation?.message ?? null

  const gitAttentionTooltipLines: string[] = []
  if (mergeRunBlocked) {
    const blockedOnSpine =
      (mergeRun as { blockedOnSpine?: boolean | null } | null | undefined)
        ?.blockedOnSpine ?? null
    gitAttentionTooltipLines.push(
      blockedOnSpine === false
        ? "Restack blocked (conflicts)."
        : "Merge blocked (conflicts).",
    )
    if (mergeRun?.blockedBranchName) {
      gitAttentionTooltipLines.push(`Branch: ${mergeRun.blockedBranchName}`)
    }
  }
  if (outOfSync) {
    gitAttentionTooltipLines.push(
      "Branch is out of sync with its effective upstream (ignores merged ancestors).",
    )
  }

  const mergeAttentionIcon = mergeRunBlocked ? (
    <AlertTriangle className="size-4 text-destructive/80" />
  ) : null

  const syncAttentionIcon = outOfSync ? (
    <GitBranch
      className={cn(
        "size-4",
        task?.state === "done"
          ? "text-muted-foreground/60"
          : "text-amber-300/70",
      )}
    />
  ) : null

  const gitAttentionIcon = mergeAttentionIcon || syncAttentionIcon

  const shouldShowResumeButton = canResumeMerge && !!onRequestRefresh

  const allowRunningActionLabel =
    allowRunningPrompt?.kind === "merge"
      ? allowRunningPrompt.cascade
        ? allowRunningPrompt.restackMode === "merge_then_restack"
          ? "Merge then Restack"
          : "Merge and Restack"
        : "Merge"
      : allowRunningPrompt?.kind === "restack"
        ? "Restack"
        : allowRunningPrompt?.kind === "resume"
          ? allowRunningPrompt.operation === "restack"
            ? "Resume restack"
            : "Resume merge"
          : null

  return (
    <Card
      data-node-card
      className={cn(
        "group",
        "relative overflow-visible",
        "py-0",
        "cursor-pointer transition-[background-color,box-shadow] duration-200 hover:bg-accent/40",
        actionsMenuOpen ? "bg-accent/40" : null,
        task?.state === "done"
          ? "ring-green-950/80"
          : mergeReady
            ? "ring-emerald-400/50"
            : null,
        isSelected
          ? "ring-[3px] ring-ring"
          : isHighlighted
            ? "ring-1 ring-ring/60"
            : null,
      )}
      onClick={(e) => onSelect({ additive: e.metaKey || e.ctrlKey })}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onSelect({ additive: false })
        }
      }}
    >
      {linearIssueId ? (
        <div
          className={cn(
            "nodrag nopan absolute left-0 top-0 z-40 flex items-center gap-1 -translate-y-[calc(100%+8px)]",
          )}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <Tooltip>
            <TooltipTrigger
              render={(tooltipTriggerProps) => (
                <button
                  {...tooltipTriggerProps}
                  type="button"
                  className={cn(
                    "group/linear inline-flex h-6 max-w-48 items-center overflow-hidden whitespace-nowrap rounded-full border-2 shadow-sm backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30",
                    "transition-colors duration-200",
                    actionsMenuOpen
                      ? "bg-accent/40"
                      : "bg-transparent group-hover:bg-accent/40 group-focus-within:bg-accent/40",
                    linearPillBorderClass,
                    linearObservedOld ? "border-opacity-70" : null,
                    linearStateMismatch ? "ring-1 ring-amber-300/25" : null,
                  )}
                  aria-label={linearPillLabel}
                  onClick={() => {
                    window.open(
                      `/v1/linear/issues/${linearIssueId}/open`,
                      "_blank",
                      "noreferrer",
                    )
                  }}
                >
                  <span className="relative inline-flex size-6 shrink-0 items-center justify-center">
                    <LinearIcon className="size-3.5 text-muted-foreground" />
                  </span>
                  <span className="min-w-0 truncate pr-2 font-mono text-[0.625rem] leading-none text-muted-foreground">
                    {linearPillLabel}
                  </span>
                </button>
              )}
            />
            <TooltipContent side="bottom" sideOffset={10}>
              <div className="space-y-1">
                <div className="text-xs">
                  Linear: {linearStateLabel(linearStateName, linearStateType)}
                </div>
                {linearStateMismatch ? (
                  <div className="text-xs text-muted-foreground">
                    Out of sync with local
                  </div>
                ) : null}
                {linearObservedAtLabel ? (
                  <div className="text-xs text-muted-foreground">
                    Last observed {linearObservedAtLabel}
                  </div>
                ) : null}
              </div>
            </TooltipContent>
          </Tooltip>
        </div>
      ) : null}
      {taskId !== null && onRequestRefresh ? (
        <div
          className={cn(
            "nodrag nopan absolute right-0 top-2.5 z-40 translate-x-[calc(100%+8px)] transition-opacity",
            isSelected || actionsMenuOpen
              ? "opacity-100"
              : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
          )}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <DropdownMenu onOpenChange={(open) => setActionsMenuOpen(open)}>
            <DropdownMenuTrigger
              render={(triggerProps) => (
                <Tooltip>
                  <TooltipTrigger
                    render={(tooltipTriggerProps) => (
                      <Button
                        {...tooltipTriggerProps}
                        {...triggerProps}
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Task actions"
                        className={cn(
                          "rounded-full border border-border/60 bg-accent/40 shadow-sm backdrop-blur hover:bg-accent/60",
                          triggerProps.className,
                          tooltipTriggerProps.className,
                        )}
                      >
                        <EllipsisVertical className="size-4" />
                      </Button>
                    )}
                  />
                  <TooltipContent side="bottom" sideOffset={10}>
                    Task actions
                  </TooltipContent>
                </Tooltip>
              )}
            />
            <DropdownMenuContent align="end" side="bottom" sideOffset={10}>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleRestack({ scope: "descendants" })
                      }}
                    >
                      <GitBranch className="size-3.5" />
                      Restack
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Rebase this branch and downstream branches to keep the stack intact."}
                </TooltipContent>
              </Tooltip>
              <DropdownMenuSeparator />
              <DropdownMenuCheckboxItem
                checked={mergeReady}
                disabled={
                  !canMerge ||
                  pendingMerge !== null ||
                  canResumeMerge ||
                  (!task?.branchName && !mergeReady)
                }
                closeOnClick={false}
                onClick={(e) => e.stopPropagation()}
                onCheckedChange={(checked) =>
                  void handleToggleMergeReady(checked)
                }
              >
                Ready to merge
              </DropdownMenuCheckboxItem>
              <DropdownMenuSeparator />
              {canResumeMerge ? (
                <>
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <DropdownMenuItem
                          {...triggerProps}
                          disabled={
                            pendingMerge !== null || gitDisabledReason !== null
                          }
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            void handleResumeMerge()
                          }}
                        >
                          <Play className="size-3.5" />
                          {mergeRunOperation === "restack"
                            ? "Resume restack"
                            : "Resume merge"}
                        </DropdownMenuItem>
                      )}
                    />
                    <TooltipContent side="right" sideOffset={12} align="center">
                      {gitDisabledReason ??
                        `Continue a previously-blocked ${
                          mergeRunOperation === "restack" ? "restack" : "merge"
                        } run after resolving conflicts.`}
                    </TooltipContent>
                  </Tooltip>
                  <DropdownMenuSeparator />
                </>
              ) : null}
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        !canMerge ||
                        !mergeReady ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleMerge({ cascade: false })
                      }}
                    >
                      <GitMerge className="size-3.5" />
                      Merge
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Fast-forward merge the task + its unmerged ancestors into the epic base branch."}
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        !canMerge ||
                        !mergeReady ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleMerge({ cascade: true })
                      }}
                    >
                      <Layers className="size-3.5" />
                      Merge and Restack
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Rebase downstream branches to keep the stack intact, then fast-forward merge the task + its unmerged ancestors into the epic base branch."}
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <DropdownMenuItem
                      {...triggerProps}
                      disabled={
                        !canMerge ||
                        !mergeReady ||
                        pendingMerge !== null ||
                        canResumeMerge ||
                        gitDisabledReason !== null
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        void handleMergeThenRestack()
                      }}
                    >
                      <Layers className="size-3.5" />
                      Merge then Restack
                    </DropdownMenuItem>
                  )}
                />
                <TooltipContent side="right" sideOffset={12} align="center">
                  {gitDisabledReason ??
                    "Fast-forward merge the task + its unmerged ancestors as in [Merge], then rebase downstream branches to keep the stack intact."}
                </TooltipContent>
              </Tooltip>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      ) : null}
      <CardContent className="flex h-full flex-col gap-2 p-3">
        <div className="flex min-w-0 items-center justify-between gap-2">
          <div
            className={cn(
              "min-w-0 truncate font-mono text-xs leading-none text-muted-foreground",
              branchLabelProvisional ? "opacity-60" : null,
            )}
          >
            {branchLabel}
          </div>
          <div className="flex items-center gap-1">
            {quickActions ? (
              <div
                className={cn(
                  "flex items-center gap-1 transition-opacity",
                  isSelected || actionsMenuOpen
                    ? "opacity-100"
                    : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100",
                )}
              >
                {quickActions}
              </div>
            ) : null}
            <div className="inline-flex size-5 items-center justify-center">
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <span
                      {...triggerProps}
                      className={cn("inline-flex", triggerProps.className)}
                      aria-label={tooltip}
                    >
                      <AgentStatusIcon
                        status={agentSession?.status ?? null}
                        taskState={task?.state}
                        className="size-3"
                        label={tooltip}
                      />
                    </span>
                  )}
                />
                <TooltipContent side="bottom" sideOffset={10}>
                  {tooltip}
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </div>

        <div className="text-sm font-medium leading-tight">
          {task?.title ?? "—"}
        </div>
        <div className="mt-auto flex items-end justify-between gap-2 pt-1">
          {agentSession ? (
            <div className="flex flex-wrap items-end justify-start gap-2">
              <ResourceBadge
                label={agentSession.agentLabel}
                value={harnessKind}
                extendBackground
              />
            </div>
          ) : (
            <div />
          )}
          {gitAttentionIcon ? (
            <Tooltip>
              <TooltipTrigger
                render={(triggerProps) => (
                  <span
                    {...triggerProps}
                    className={cn(
                      "inline-flex items-center gap-1",
                      triggerProps.className,
                    )}
                    aria-label="Git status attention"
                  >
                    {syncAttentionIcon}
                    {mergeAttentionIcon}
                  </span>
                )}
              />
              <TooltipContent side="left" sideOffset={12} align="center">
                <div className="whitespace-pre-line">
                  {gitAttentionTooltipLines.join("\n")}
                </div>
              </TooltipContent>
            </Tooltip>
          ) : null}
        </div>
      </CardContent>
      {actionError ||
      shouldShowResumeButton ||
      blockingMergeRunBlockedRebase ? (
        <div
          className="nodrag nopan absolute left-0 top-full z-50 mt-3 w-full space-y-2"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          {actionError ? (
            <Card
              size="sm"
              className="gap-2 border border-destructive/25 bg-destructive/10 py-2 shadow-lg ring-destructive/10 backdrop-blur"
            >
              <CardContent className="px-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <AlertTriangle className="size-3.5 text-destructive" />
                    <Tooltip>
                      <TooltipTrigger
                        render={(triggerProps) => (
                          <div
                            {...triggerProps}
                            className={cn(
                              "truncate text-xs font-medium text-foreground",
                              triggerProps.className,
                            )}
                          >
                            {actionError.title}
                          </div>
                        )}
                      />
                      <TooltipContent side="top" sideOffset={8}>
                        {actionError.title}
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant="ghost"
                          size="icon-xs"
                          aria-label="Dismiss error"
                          className={cn(triggerProps.className)}
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            clearActionError()
                          }}
                        >
                          <X className="size-3" />
                        </Button>
                      )}
                    />
                    <TooltipContent side="bottom" sideOffset={10}>
                      Dismiss error
                    </TooltipContent>
                  </Tooltip>
                </div>

                <div
                  className={cn(
                    "break-words text-xs text-foreground/80",
                    actionErrorExpanded
                      ? "whitespace-pre-wrap"
                      : "line-clamp-2",
                  )}
                >
                  {actionError.summary}
                </div>

                {mergeRunBlockedRebase ? (
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <Button
                      variant="ghost"
                      size="xs"
                      disabledReason={
                        rebaseAttachCommand
                          ? null
                          : "No task id recorded for this blocked step."
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        if (!rebaseAttachCommand) {
                          return
                        }
                        void copyToClipboard(rebaseAttachCommand)
                      }}
                    >
                      <Terminal className="size-3" />
                      Copy attach
                    </Button>
                    <Button
                      variant="ghost"
                      size="xs"
                      disabledReason={
                        rebaseRemediationMessage
                          ? null
                          : "No remediation message available."
                      }
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        if (!rebaseRemediationMessage) {
                          return
                        }
                        void copyToClipboard(rebaseRemediationMessage)
                      }}
                    >
                      <MessageSquareText className="size-3" />
                      Copy agent note
                    </Button>
                  </div>
                ) : null}

                <div className="mt-1 flex items-center justify-between gap-2">
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      setActionErrorExpanded((current) => !current)
                    }}
                  >
                    {actionErrorExpanded ? (
                      <ChevronUp className="size-3" />
                    ) : (
                      <ChevronDown className="size-3" />
                    )}
                    {actionErrorExpanded ? "Hide" : "Show"} details
                  </Button>
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      void copyToClipboard(actionError.raw)
                    }}
                  >
                    Copy
                  </Button>
                </div>

                {actionErrorExpanded ? (
                  <div className="mt-1 max-h-40 overflow-auto rounded-md bg-background/40 px-2 py-1.5 font-mono text-[0.625rem] text-foreground/80">
                    <div className="whitespace-pre-wrap break-words">
                      {actionError.raw}
                    </div>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : null}

          {blockingMergeRunBlockedRebase ? (
            <Card
              size="sm"
              className="gap-2 border border-amber-400/25 bg-amber-400/10 py-2 shadow-lg ring-amber-400/10 backdrop-blur"
            >
              <CardContent
                className={cn(
                  "flex flex-col gap-2 px-3",
                  "cursor-pointer select-none",
                )}
                role="button"
                tabIndex={0}
                aria-expanded={blockedRebaseExpanded}
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  setBlockedRebaseExpanded((current) => !current)
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    e.stopPropagation()
                    setBlockedRebaseExpanded((current) => !current)
                  }
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <Pause className="size-3.5 text-amber-300/80" />
                    <div className="flex min-w-0 items-center gap-1">
                      <div className="truncate text-xs font-medium text-foreground">
                        Rebase blocked
                      </div>
                      <ChevronRight
                        className={cn(
                          "size-3 text-foreground/70 transition-transform",
                          blockedRebaseExpanded ? "rotate-90" : null,
                        )}
                        aria-hidden="true"
                      />
                    </div>
                  </div>
                  <div
                    className="flex shrink-0 items-center gap-1"
                    onClick={(event) => event.stopPropagation()}
                  >
                    {blockingAttachCommand ? (
                      <Tooltip>
                        <TooltipTrigger
                          render={(triggerProps) => (
                            <Button
                              {...triggerProps}
                              variant="ghost"
                              size="icon-sm"
                              aria-label="Copy attach command"
                              onClick={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                                void copyToClipboard(blockingAttachCommand)
                              }}
                            >
                              <Terminal />
                            </Button>
                          )}
                        />
                        <TooltipContent
                          side="bottom"
                          sideOffset={10}
                          showArrow={false}
                        >
                          Copy attach
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        disabledReason="No task id recorded for this blocked step."
                        aria-label="Copy attach command"
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                        }}
                      >
                        <Terminal />
                      </Button>
                    )}

                    {blockingRemediationMessage ? (
                      <Tooltip>
                        <TooltipTrigger
                          render={(triggerProps) => (
                            <Button
                              {...triggerProps}
                              variant="ghost"
                              size="icon-sm"
                              aria-label="Copy agent note"
                              onClick={(e) => {
                                e.preventDefault()
                                e.stopPropagation()
                                void copyToClipboard(blockingRemediationMessage)
                              }}
                            >
                              <MessageSquareText />
                            </Button>
                          )}
                        />
                        <TooltipContent
                          side="bottom"
                          sideOffset={10}
                          showArrow={false}
                        >
                          Copy agent note
                        </TooltipContent>
                      </Tooltip>
                    ) : (
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        disabledReason="No remediation message available."
                        aria-label="Copy agent note"
                        onClick={(e) => {
                          e.preventDefault()
                          e.stopPropagation()
                        }}
                      >
                        <MessageSquareText />
                      </Button>
                    )}
                  </div>
                </div>

                {blockedRebaseExpanded ? (
                  <div className="space-y-1 text-[11px] text-foreground/80">
                    {blockingMergeRun?.blockedBranchName ? (
                      <div className="truncate">
                        Blocked branch:{" "}
                        <span className="font-mono text-foreground/90">
                          {blockingMergeRun.blockedBranchName}
                        </span>
                      </div>
                    ) : null}
                    {blockingMergeRun?.blockedWorktreePath ? (
                      <div className="truncate">
                        Worktree:{" "}
                        <span className="font-mono text-foreground/90">
                          {blockingMergeRun.blockedWorktreePath}
                        </span>
                      </div>
                    ) : null}
                    <div className="text-foreground/70">
                      Resolve conflicts, then resume the run.
                    </div>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : null}

          {shouldShowResumeButton ? (
            <div className="flex justify-end">
              <Button
                variant="outline"
                size="sm"
                className="border-emerald-400/35 text-emerald-100 hover:bg-emerald-400/10 hover:text-emerald-50"
                disabledReason={
                  pendingMerge !== null
                    ? "Action in progress"
                    : gitDisabledReason
                }
                onClick={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  void handleResumeMerge()
                }}
              >
                {pendingMerge === "resume" ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <Play className="size-3" />
                )}
                {mergeRunOperation === "restack"
                  ? "Resume restack"
                  : "Resume merge"}
              </Button>
            </div>
          ) : null}
        </div>
      ) : null}
      <ProceedAnywayDialog
        open={allowRunningPrompt !== null}
        pending={allowRunningConfirming}
        actionLabel={allowRunningActionLabel ?? "merge"}
        detail={allowRunningPrompt?.detail}
        onOpenChange={(open) => {
          if (open) {
            return
          }
          setAllowRunningPrompt(null)
        }}
        onProceed={() => void confirmAllowRunning()}
      />
    </Card>
  )
}
