import { BaseEdge, getSmoothStepPath, type EdgeProps } from "@xyflow/react"
import { GRAPH_EDGE_BORDER_RADIUS } from "./graphConfig"
import { cn } from "@/lib/utils"
import type { EdgePulseData } from "./edgePulse"

export function RoundedSmoothStepEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  style,
  data,
}: EdgeProps) {
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: GRAPH_EDGE_BORDER_RADIUS,
  })

  const pulse =
    (data as { pulse?: EdgePulseData | null } | undefined)?.pulse ?? null

  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      {pulse ? (
        <path
          key={pulse.token}
          d={edgePath}
          fill="none"
          stroke="currentColor"
          strokeWidth={2.25}
          strokeLinecap="round"
          strokeDasharray="14 1000"
          className={cn(
            "pointer-events-none",
            pulse.kind === "merge"
              ? "rn-edge-pulse-merge text-emerald-300/70"
              : "rn-edge-pulse-rebase text-sky-300/70",
          )}
        />
      ) : null}
    </>
  )
}
