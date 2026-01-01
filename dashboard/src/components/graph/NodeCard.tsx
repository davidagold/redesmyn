import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import {
  ApiHttpError,
  mergeTask,
  resumeMergeRun,
  restartTaskAgent,
  setTaskMergeReady,
  startTaskAgent,
  stopTaskAgent,
} from "@/api"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
import type { Agent, GraphNode, MergeRun, Task } from "@/lib/graph-utils"
import { isRecentActivity, type NodeActivity } from "@/lib/presence"
import { AgentStatusIcon } from "@/components/agents/AgentStatusIcon"
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
  EllipsisVertical,
  AlertTriangle,
  GitBranch,
  GitMerge,
  Layers,
  Play,
  RotateCcw,
  Square,
  Terminal,
  X,
} from "lucide-react"

interface NodeCardProps {
  node: GraphNode
  task?: Task
  agent?: Agent
  mergeRun?: MergeRun
  activity?: NodeActivity
  branchLabel: string
  harnessCommand: string
  detach: boolean
  isSelected: boolean
  isHighlighted?: boolean
  onSelect: (options: { additive: boolean }) => void
  onRequestRefresh?: () => void
}

function statusSummary(task: Task | undefined, agent: Agent | undefined) {
  if (task?.state === "blocked") {
    return "blocked"
  }
  if (task?.state === "done") {
    return "done"
  }
  if (!agent) {
    return "not started"
  }
  return agent.status
}

export function NodeCard({
  node,
  task,
  agent,
  mergeRun,
  activity,
  branchLabel,
  harnessCommand,
  detach,
  isSelected,
  isHighlighted = false,
  onSelect,
  onRequestRefresh,
}: NodeCardProps) {
  const now = Date.now()
  const commitHot = isRecentActivity(activity?.lastCommitAt, now)
  const worktreeHot = isRecentActivity(activity?.lastWorktreeAt, now)
  const harnessKind = agent?.harnessProfileId?.split("/")[0] ?? null
  const stackInSync = node.stackInSync ?? null
  const outOfSync = stackInSync === false
  const mergeRunStatus = mergeRun?.status ?? null
  const mergeRunBlocked = mergeRunStatus === "blocked"
  const mergeRunResumable = mergeRunStatus === "resumable"

  const [pendingAction, setPendingAction] =
    useState<"start" | "stop" | "restart" | "attach" | null>(null)
  const [pendingMerge, setPendingMerge] =
    useState<"ready" | "merge" | "mergeStack" | "resume" | null>(null)
  const [actionsMenuOpen, setActionsMenuOpen] = useState(false)
  const [mergeReady, setMergeReady] = useState(Boolean(task?.mergeReadyAt))
  const [refreshPendingAfterMenuClose, setRefreshPendingAfterMenuClose] =
    useState(false)
  const [actionErrorExpanded, setActionErrorExpanded] = useState(false)
  const [actionError, setActionError] = useState<{
    title: string
    summary: string
    raw: string
  } | null>(null)

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

  const agentStatus = agent?.status ?? null
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
    (!agent || agentStatus === "stopped") && harnessCommand.trim()
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
    const previous = mergeReady
    setPendingMerge("ready")
    setMergeReady(next)
    clearActionError()
    try {
      await setTaskMergeReady(taskId, next)
      if (actionsMenuOpen) {
        setRefreshPendingAfterMenuClose(true)
      } else {
        onRequestRefresh()
      }
    } catch (e) {
      setMergeReady(previous)
      setActionErrorFromException("Set merge readiness", e)
    } finally {
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
    try {
      await mergeTask(taskId, { cascade })
      onRequestRefresh()
    } catch (e) {
      if (e instanceof ApiHttpError && e.status === 409) {
        const confirmed = window.confirm(
          "This merge affects running tasks/agents.\n\nProceed anyway?",
        )
        if (!confirmed) {
          return
        }
        try {
          await mergeTask(taskId, { cascade, allowRunning: true })
          onRequestRefresh()
        } catch (inner) {
          setActionErrorFromException(
            cascade ? "Merge and Restack" : "Merge",
            inner,
          )
        }
        return
      }
      setActionErrorFromException(cascade ? "Merge and Restack" : "Merge", e)
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
        const confirmed = window.confirm(
          "This merge affects running tasks/agents.\n\nProceed anyway?",
        )
        if (!confirmed) {
          return
        }
        try {
          await resumeMergeRun(mergeRun.runId, { allowRunning: true })
          onRequestRefresh()
        } catch (inner) {
          setActionErrorFromException("Resume merge", inner)
        }
        return
      }
      setActionErrorFromException("Resume merge", e)
    } finally {
      setPendingMerge(null)
    }
  }

  const quickActions =
    quickActionsEnabled && taskId !== null ? (
      isRunning ? (
        <>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Copy attach command"
            title="Copy attach command"
            disabledReason={
              pendingAction !== null ? "Action in progress" : null
            }
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleAttach()
            }}
          >
            <Terminal className="size-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Restart agent"
            title="Restart agent"
            disabledReason={
              pendingAction !== null ? "Action in progress" : null
            }
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleRestart()
            }}
          >
            <RotateCcw className="size-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon-xs"
            className="text-destructive hover:bg-destructive/10"
            aria-label="Stop agent"
            title="Stop agent"
            disabledReason={
              pendingAction !== null ? "Action in progress" : null
            }
            onClick={(e) => {
              e.preventDefault()
              e.stopPropagation()
              void handleStop()
            }}
          >
            <Square className="size-3" />
          </Button>
        </>
      ) : canRestart ? (
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Restart agent"
          title="Restart agent"
          disabledReason={pendingAction !== null ? "Action in progress" : null}
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            void handleRestart()
          }}
        >
          <RotateCcw className="size-3" />
        </Button>
      ) : canStart ? (
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Start agent"
          title="Start agent"
          disabledReason={
            pendingAction !== null
              ? "Action in progress"
              : !harnessCommand.trim()
                ? "Set a harness command in Configure"
                : null
          }
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            void handleStart()
          }}
        >
          <Play className="size-3" />
        </Button>
      ) : null
    ) : null

  const tooltipParts = [
    agent ? `Agent: ${agent.displayName}` : "Agent: (not started)",
    `Status: ${statusSummary(task, agent)}`,
    stackInSync === true
      ? "Stack: in sync"
      : stackInSync === false
        ? "Stack: out of sync"
        : null,
  ]
  const tooltip = tooltipParts.filter(Boolean).join("\n")

  const gitAttentionKind: "mergeBlocked" | "mergeResumable" | "outOfSync" | null =
    mergeRunBlocked
      ? "mergeBlocked"
      : mergeRunResumable
        ? "mergeResumable"
        : outOfSync
          ? "outOfSync"
          : null

  const gitAttentionIcon =
    gitAttentionKind === "mergeBlocked" ? (
      <AlertTriangle className="size-4" />
    ) : gitAttentionKind === "mergeResumable" ? (
      <Play className="size-4" />
    ) : gitAttentionKind === "outOfSync" ? (
      <GitBranch className="size-4" />
    ) : null

  const gitAttentionTextClass =
    gitAttentionKind === "mergeBlocked"
      ? "text-destructive/80"
      : gitAttentionKind === "mergeResumable"
        ? "text-amber-200/80"
        : "text-amber-300/70"

  const gitAttentionTooltipLines: string[] = []
  if (mergeRunBlocked) {
    gitAttentionTooltipLines.push("Merge blocked (conflicts).")
    if (mergeRun?.blockedBranchName) {
      gitAttentionTooltipLines.push(`Branch: ${mergeRun.blockedBranchName}`)
    }
  } else if (mergeRunResumable) {
    gitAttentionTooltipLines.push("Merge ready to resume.")
  }
  if (outOfSync) {
    gitAttentionTooltipLines.push("Branch is out of sync with its upstream.")
  }

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
          ? "ring-emerald-500/35"
          : mergeReady
            ? "ring-emerald-400/50"
            : null,
        isSelected
          ? "ring-2 ring-ring"
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
                <Button
                  {...triggerProps}
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Task actions"
                  title="Task actions"
                  className={cn(
                    "rounded-full border border-border/60 bg-accent/40 shadow-sm backdrop-blur hover:bg-accent/60",
                    triggerProps.className,
                  )}
                >
                  <EllipsisVertical className="size-4" />
                </Button>
              )}
            />
            <DropdownMenuContent align="end" side="bottom" sideOffset={10}>
              <DropdownMenuCheckboxItem
                checked={mergeReady}
                disabled={!canMerge || pendingMerge !== null || canResumeMerge}
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
                          disabled={pendingMerge !== null}
                          onClick={(e) => {
                            e.preventDefault()
                            e.stopPropagation()
                            void handleResumeMerge()
                          }}
                        >
                          <Play className="size-3.5" />
                          Resume merge
                        </DropdownMenuItem>
                      )}
                    />
                    <TooltipContent side="right" sideOffset={12} align="center">
                      Continue a previously-blocked merge run after resolving
                      conflicts.
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
                        canResumeMerge
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
                  Fast-forward merge the task + its ancestors into the epic base
                  branch.
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
                        canResumeMerge
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
                  Fast-forward the task and ancestors as in [Merge], plus rebase
                  downstream branches to keep the stack intact.
                </TooltipContent>
              </Tooltip>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      ) : null}
      <CardContent className="flex h-full flex-col gap-2 p-3">
        <div className="flex min-w-0 items-center justify-between gap-2">
          <div
            className="min-w-0 truncate font-mono text-xs leading-none text-muted-foreground"
            title={node.branchName}
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
            <div
              className="relative inline-flex size-5 items-center justify-center"
              title={tooltip}
            >
              {commitHot ? (
                <span className="absolute inset-0 m-auto inline-flex size-3 animate-ping rounded-full bg-sky-400/60 opacity-75" />
              ) : worktreeHot ? (
                <span className="absolute inset-0 m-auto inline-flex size-3 animate-ping rounded-full bg-amber-400/60 opacity-75" />
              ) : null}
              <AgentStatusIcon
                status={agent?.status ?? null}
                taskState={task?.state}
                className="size-3"
                title={tooltip}
              />
            </div>
          </div>
        </div>

        <div className="text-sm font-medium leading-tight">
          {task?.title ?? "—"}
        </div>

        {agent ? (
          <div className="mt-auto flex flex-wrap items-end justify-start gap-2">
            <span className="rounded-sm bg-accent px-2 py-0.5 font-mono text-xs text-accent-foreground/80 transition-colors group-hover:bg-accent/70 group-focus-within:bg-accent/70">
              {agent.displayName}
            </span>
            {harnessKind ? (
              <span className="rounded-sm bg-muted/60 px-2 py-0.5 font-mono text-xs text-muted-foreground transition-colors group-hover:bg-muted/75 group-focus-within:bg-muted/75">
                {harnessKind}
              </span>
            ) : null}
          </div>
        ) : null}
      </CardContent>
      {actionError ? (
        <div
          className="nodrag nopan absolute left-0 top-full z-50 mt-2 w-full"
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <Card
            size="sm"
            className="gap-2 border border-destructive/25 bg-destructive/10 py-2 shadow-lg ring-destructive/10 backdrop-blur"
          >
            <CardContent className="px-3">
              <div className="flex items-start justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2">
                  <AlertTriangle className="size-3.5 text-destructive" />
                  <div
                    className="truncate text-xs font-medium text-foreground"
                    title={actionError.title}
                  >
                    {actionError.title}
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  aria-label="Dismiss error"
                  title="Dismiss error"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    clearActionError()
                  }}
                >
                  <X className="size-3" />
                </Button>
              </div>

              <div
                className={cn(
                  "break-words text-xs text-foreground/80",
                  actionErrorExpanded ? "whitespace-pre-wrap" : "line-clamp-2",
                )}
                title={actionError.summary}
              >
                {actionError.summary}
              </div>

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
        </div>
      ) : null}
      {gitAttentionKind && gitAttentionIcon ? (
        <Tooltip>
          <TooltipTrigger
            render={(triggerProps) => (
              <span
                {...triggerProps}
                className={cn(
                  "nodrag nopan absolute bottom-3 right-3",
                  gitAttentionTextClass,
                  triggerProps.className,
                )}
                aria-label="Git status attention"
              >
                {gitAttentionIcon}
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
    </Card>
  )
}
