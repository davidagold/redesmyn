import type { AgentSession } from "@/lib/graph-utils"
import { ResourceBadge } from "@/components/ui/resource-badge"
import { cn } from "@/lib/utils"

type TurnState = AgentSession["agentSemanticStatus"]["turnState"] | null

function labelForTurnState(turnState: TurnState) {
  switch (turnState) {
    case "busy":
      return "working"
    case "completed":
      return "done"
    case "ready":
      return "ready"
    case "blocked":
      return "blocked"
    case "unknown":
    case null:
      return "unknown"
  }
}

function textTone(turnState: TurnState) {
  switch (turnState) {
    case "busy":
      return "text-foreground/90"
    case "completed":
      return "text-foreground/90"
    case "ready":
      return "text-foreground/80"
    case "blocked":
      return "text-destructive"
    case "unknown":
    case null:
      return "text-muted-foreground"
  }
}

export type AgentTurnStateBadgeProps = {
  turnState: TurnState
  className?: string
}

export function AgentTurnStateBadge({
  turnState,
  className,
}: AgentTurnStateBadgeProps) {
  return (
    <ResourceBadge
      label="turn"
      value={labelForTurnState(turnState)}
      variant="muted"
      size="xs"
      className={cn(textTone(turnState), className)}
    />
  )
}
