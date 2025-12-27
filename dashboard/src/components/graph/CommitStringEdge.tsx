import {
  BaseEdge,
  EdgeLabelRenderer,
  type Edge,
  getSmoothStepPath,
  useStore,
  type EdgeProps,
} from "@xyflow/react"
import type { CSSProperties } from "react"

export type CommitStringEdgeData = {
  commitCount?: number | null
  baseSha?: string | null
  headSha?: string | null
}

export type CommitStringFlowEdge = Edge<CommitStringEdgeData, "commitString">

const selectZoom = (state: { transform: [number, number, number] }) =>
  state.transform[2]

function dashPattern(commitCount: number, zoom: number): string | undefined {
  if (!Number.isFinite(commitCount) || commitCount <= 0) {
    return undefined
  }

  const clamped = Math.min(commitCount, 50)
  const density = 1 + clamped / 12
  const dash = Math.max(2, Math.round(2.5 / zoom))
  const gap = Math.max(2, Math.round(10 / density))
  return `${dash} ${gap}`
}

export function CommitStringEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
  style,
}: EdgeProps<CommitStringFlowEdge>) {
  const zoom = useStore(selectZoom)
  const commitCount = data?.commitCount ?? null
  const showLabel = commitCount !== null && zoom >= 0.8
  const showTicks = commitCount !== null && zoom >= 1.05

  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 18,
  })

  const edgeStyle: CSSProperties = {
    ...style,
    strokeDasharray: showTicks
      ? dashPattern(commitCount ?? 0, zoom)
      : undefined,
  }

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={edgeStyle}
        interactionWidth={24}
      />

      {showLabel ? (
        <EdgeLabelRenderer>
          <div
            className="pointer-events-none absolute rounded-full border border-border/30 bg-background/80 px-2 py-0.5 text-[11px] font-medium text-foreground/80 shadow-sm"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            }}
          >
            {commitCount}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  )
}
