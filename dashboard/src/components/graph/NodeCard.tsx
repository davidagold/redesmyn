import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"

interface NodeCardProps {
  node: GraphNode
  task?: Task
  agent?: Agent
  branchLabel: string
  isSelected: boolean
  onSelect: () => void
}

export function NodeCard({
  node,
  task,
  agent,
  branchLabel,
  isSelected,
  onSelect,
}: NodeCardProps) {
  return (
    <Card
      data-node-card
      className={cn(
        "cursor-pointer transition-colors hover:bg-accent/40",
        isSelected && "ring-2 ring-ring",
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
            node {node.id}
            {agent ? ` · agent: ${agent.displayName}` : ""}
          </div>
        </div>
        <div className="text-sm text-muted-foreground">
          {task?.title ?? "—"}
        </div>
      </CardContent>
    </Card>
  )
}
