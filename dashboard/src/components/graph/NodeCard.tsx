import { useEffect, useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { restartTaskAgent, startTaskAgent, stopTaskAgent } from "@/api"
import { getStoredHarnessCommand } from "@/lib/agent-settings"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"
import { isRecentActivity, type NodeActivity } from "@/lib/presence"
import { Play, RotateCcw, Square, Terminal } from "lucide-react"

interface NodeCardProps {
  node: GraphNode
  task?: Task
  agent?: Agent
  activity?: NodeActivity
  branchLabel: string
  isSelected: boolean
  isHighlighted?: boolean
  onSelect: (options: { additive: boolean }) => void
  onRequestRefresh?: () => void
}

function statusColor(task: Task | undefined, agent: Agent | undefined) {
  if (task?.state === "blocked") {
    return "border-amber-400 bg-amber-400"
  }
  if (task?.state === "done") {
    return "border-muted-foreground/40 bg-muted-foreground/40"
  }
  if (!agent) {
    return "border-muted-foreground/60 bg-transparent border-dashed"
  }
  if (agent.status === "running") {
    return "border-emerald-400 bg-emerald-400"
  }
  if (agent.status === "blocked") {
    return "border-amber-400 bg-amber-400"
  }
  if (agent.status === "error") {
    return "border-rose-400 bg-rose-400"
  }
  return "border-muted-foreground/60 bg-transparent"
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
  activity,
  branchLabel,
  isSelected,
  isHighlighted = false,
  onSelect,
  onRequestRefresh,
}: NodeCardProps) {
  const now = Date.now()
  const commitHot = isRecentActivity(activity?.lastCommitAt, now)
  const worktreeHot = isRecentActivity(activity?.lastWorktreeAt, now)
  const harnessKind = agent?.harnessProfileId?.split("/")[0] ?? null

  const [pendingAction, setPendingAction] =
    useState<"start" | "stop" | "restart" | "attach" | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  useEffect(() => {
    if (!actionError) {
      return
    }
    const id = window.setTimeout(() => setActionError(null), 2_000)
    return () => window.clearTimeout(id)
  }, [actionError])

  const agentStatus = agent?.status ?? null
  const taskId = task?.id ?? null
  const quickActionsEnabled =
    taskId !== null &&
    !!onRequestRefresh &&
    task?.state !== "blocked" &&
    task?.state !== "done"

  const isRunning = agentStatus === "running" || agentStatus === "blocked"
  const canStart = !agent || agentStatus === "stopped"
  const canRestart = isRunning || agentStatus === "error"

  async function handleAttach() {
    if (taskId === null) {
      return
    }
    setPendingAction("attach")
    setActionError(null)
    try {
      await copyToClipboard(`rn agent attach --task ${taskId}`)
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setPendingAction(null)
    }
  }

  async function handleStart() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setPendingAction("start")
    setActionError(null)
    try {
      const harness = getStoredHarnessCommand()
      await startTaskAgent(taskId, { harness, detach: true })
      onRequestRefresh()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setPendingAction(null)
    }
  }

  async function handleStop() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setPendingAction("stop")
    setActionError(null)
    try {
      await stopTaskAgent(taskId)
      onRequestRefresh()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setPendingAction(null)
    }
  }

  async function handleRestart() {
    if (taskId === null || !onRequestRefresh) {
      return
    }
    setPendingAction("restart")
    setActionError(null)
    try {
      await restartTaskAgent(taskId, { detach: true })
      onRequestRefresh()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e))
    } finally {
      setPendingAction(null)
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
          disabledReason={pendingAction !== null ? "Action in progress" : null}
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

  const tooltip = [
    agent ? `Agent: ${agent.displayName}` : "Agent: (not started)",
    `Status: ${statusSummary(task, agent)}`,
  ].join("\n")

  return (
    <Card
      data-node-card
      className={cn(
        "group",
        "py-0",
        "cursor-pointer transition-colors hover:bg-accent/40",
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
                  isSelected
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
              <span
                className={cn(
                  "relative inline-flex size-3 rounded-full border",
                  statusColor(task, agent),
                )}
                aria-label={tooltip}
              />
            </div>
          </div>
        </div>

        <div className="text-sm font-medium leading-tight">
          {task?.title ?? "—"}
        </div>

        {actionError ? (
          <div
            className="truncate text-xs text-destructive"
            title={actionError}
          >
            {actionError}
          </div>
        ) : null}

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
    </Card>
  )
}
