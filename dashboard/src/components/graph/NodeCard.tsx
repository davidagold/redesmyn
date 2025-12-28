import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"
import { isRecentActivity, type NodeActivity } from "@/lib/presence"

interface NodeCardProps {
  node: GraphNode
  task?: Task
  agent?: Agent
  activity?: NodeActivity
  branchLabel: string
  isSelected: boolean
  isHighlighted?: boolean
  onSelect: () => void
}

function statusColor(task: Task | undefined, agent: Agent | undefined) {
  if (task?.state === "blocked") {
    return "bg-amber-400"
  }
  if (task?.state === "done") {
    return "bg-muted-foreground/50"
  }
  const status = agent?.status ?? null
  if (!agent) {
    return "bg-muted-foreground/50"
  }
  if (status === "running") {
    return "bg-emerald-400"
  }
  if (status === "blocked") {
    return "bg-amber-400"
  }
  if (status === "error") {
    return "bg-rose-400"
  }
  return "bg-sky-400"
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
}: NodeCardProps) {
  const now = Date.now()
  const commitHot = isRecentActivity(activity?.lastCommitAt, now)
  const worktreeHot = isRecentActivity(activity?.lastWorktreeAt, now)

  const tooltip = [
    agent ? `Agent: ${agent.displayName}` : "Agent: (not started)",
    `Status: ${statusSummary(task, agent)}`,
  ].join("\n")

  return (
    <Card
      data-node-card
      className={cn(
        "cursor-pointer transition-colors hover:bg-accent/40",
        isSelected
          ? "ring-2 ring-ring"
          : isHighlighted
            ? "ring-1 ring-ring/60"
            : null,
      )}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onSelect()
        }
      }}
    >
      <CardContent className="relative flex h-full flex-col gap-2 p-4">
        <div className="absolute right-3 top-3">
          <div
            className="relative inline-flex h-3 w-3 items-center justify-center"
            title={tooltip}
          >
            {commitHot ? (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400/60 opacity-75" />
            ) : worktreeHot ? (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400/60 opacity-75" />
            ) : null}
            <span
              className={cn(
                "relative inline-flex h-2.5 w-2.5 rounded-full",
                statusColor(task, agent),
              )}
              aria-label={tooltip}
            />
          </div>
        </div>

        <div className="pr-4">
          <div className="flex items-baseline gap-2">
            {task ? (
              <div className="font-mono text-xs text-muted-foreground">
                t{task.id}
              </div>
            ) : null}
            <div className="text-sm font-medium leading-tight">
              {task?.title ?? "—"}
            </div>
          </div>
          <div
            className="font-mono text-xs text-muted-foreground"
            title={node.branchName}
          >
            {branchLabel}
          </div>
        </div>

        {agent ? (
          <div className="mt-auto flex items-end justify-end">
            <span className="rounded-md bg-muted/60 px-2 py-0.5 font-mono text-xs text-muted-foreground">
              {agent.displayName}
            </span>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
