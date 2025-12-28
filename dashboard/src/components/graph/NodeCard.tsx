import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { Agent, AgentSession, GraphNode, Task } from "@/lib/graph-utils"
import { isRecentActivity, type NodeActivity } from "@/lib/presence"

interface NodeCardProps {
  node: GraphNode
  task?: Task
  agent?: Agent
  session?: AgentSession
  activity?: NodeActivity
  branchLabel: string
  isSelected: boolean
  isHighlighted?: boolean
  onSelect: () => void
}

function agentStatusColor(agent: Agent | undefined) {
  const status = agent?.status ?? null
  if (status === "running") {
    return "bg-emerald-400"
  }
  if (status === "blocked") {
    return "bg-amber-400"
  }
  if (status === "error") {
    return "bg-rose-400"
  }
  return "bg-muted-foreground/50"
}

export function NodeCard({
  node,
  task,
  agent,
  session,
  activity,
  branchLabel,
  isSelected,
  isHighlighted = false,
  onSelect,
}: NodeCardProps) {
  const now = Date.now()
  const commitHot = isRecentActivity(activity?.lastCommitAt, now)
  const worktreeHot = isRecentActivity(activity?.lastWorktreeAt, now)

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
      <CardContent className="grid gap-1 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="font-mono text-sm" title={node.branchName}>
            {branchLabel}
          </div>
          <div className="text-xs text-muted-foreground">
            <div className="flex items-center gap-1.5">
              <span>node {node.id}</span>
              {agent ? (
                <>
                  <span className="text-muted-foreground/50">·</span>
                  <span className="truncate">{agent.displayName}</span>
                  <span
                    className={cn(
                      "h-2 w-2 rounded-full",
                      agentStatusColor(agent),
                    )}
                    aria-label={`agent status: ${agent.status}`}
                    title={`agent status: ${agent.status}`}
                  />
                </>
              ) : (
                <>
                  <span className="text-muted-foreground/50">·</span>
                  <span className="italic text-muted-foreground/80">
                    unassigned
                  </span>
                </>
              )}
              {session ? (
                <>
                  <span className="text-muted-foreground/50">·</span>
                  <span title={`session status: ${session.status}`}>
                    {session.status}
                  </span>
                </>
              ) : null}
              {commitHot ? (
                <span
                  className="relative ml-1 inline-flex h-2 w-2"
                  aria-label="recent commit activity"
                  title="recent commit activity"
                >
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400/60 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-sky-400/80" />
                </span>
              ) : worktreeHot ? (
                <span
                  className="relative ml-1 inline-flex h-2 w-2"
                  aria-label="recent worktree activity"
                  title="recent worktree activity"
                >
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400/60 opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-amber-400/80" />
                </span>
              ) : null}
            </div>
          </div>
        </div>
        <div className="text-sm text-muted-foreground">
          {task?.title ?? "—"}
        </div>
      </CardContent>
    </Card>
  )
}
