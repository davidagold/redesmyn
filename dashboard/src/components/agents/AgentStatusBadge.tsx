import type { AgentSession } from "@/lib/graph-utils"
import { cn } from "@/lib/utils"
import { AgentStatusIcon } from "./AgentStatusIcon"

type Status = AgentSession["status"] | null

function labelForStatus(status: Status) {
  return status ?? "not started"
}

function textTone(status: Status) {
  switch (status) {
    case "error":
      return "text-destructive"
    case null:
      return "text-muted-foreground"
    default:
      return "text-foreground/80"
  }
}

export type AgentStatusBadgeProps = {
  status: Status
  className?: string
}

export function AgentStatusBadge({ status, className }: AgentStatusBadgeProps) {
  const label = labelForStatus(status)
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] uppercase tracking-wide",
        textTone(status),
        className,
      )}
    >
      <AgentStatusIcon status={status} className="size-2" />
      <span>{label}</span>
    </div>
  )
}
