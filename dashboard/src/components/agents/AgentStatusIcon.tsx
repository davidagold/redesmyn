import { cn } from "@/lib/utils"
import type { AgentSession, Task } from "@/lib/graph-utils"

type Status = AgentSession["status"] | null

function statusClasses(taskState: Task["state"] | undefined, status: Status) {
  if (taskState === "blocked") {
    return "border-amber-400 bg-amber-400"
  }
  if (taskState === "done") {
    return "border-muted-foreground/40 bg-muted-foreground/40"
  }
  if (status === null) {
    return "border-muted-foreground/60 bg-transparent border-dashed"
  }
  if (status === "running") {
    return "border-emerald-400 bg-emerald-400"
  }
  if (status === "blocked") {
    return "border-amber-400 bg-amber-400"
  }
  if (status === "error") {
    return "border-rose-400 bg-rose-400"
  }
  return "border-muted-foreground/60 bg-transparent"
}

export type AgentStatusIconProps = {
  status: Status
  taskState?: Task["state"]
  className?: string
  label?: string
}

export function AgentStatusIcon({
  status,
  taskState,
  className,
  label,
}: AgentStatusIconProps) {
  return (
    <span
      className={cn(
        "inline-flex size-3 rounded-full border",
        statusClasses(taskState, status),
        className,
      )}
      aria-label={label}
    />
  )
}
