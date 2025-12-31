import {
  BaseEdge,
  EdgeLabelRenderer,
  type Edge,
  getSmoothStepPath,
  useStore,
  type EdgeProps,
} from "@xyflow/react"
import type { CSSProperties } from "react"
import {
  edgeLodBand,
  GRAPH_EDGE_BORDER_RADIUS,
  GRAPH_EDGE_STYLE_ANIMATION_MS,
} from "./graphConfig"

export type CommitStringEdgeData = {
  commitCount?: number | null
  baseSha?: string | null
  headSha?: string | null
}

export type CommitStringFlowEdge = Edge<CommitStringEdgeData, "commitString">

const selectEdgeLod = (state: { transform: [number, number, number] }) =>
  edgeLodBand(state.transform[2])

function dashPattern(commitCount: number): string | undefined {
  if (!Number.isFinite(commitCount) || commitCount <= 0) {
    return undefined
  }

  const clamped = Math.min(commitCount, 50)
  const density = 1 + clamped / 12
  const dash = 2
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
  const lod = useStore(selectEdgeLod)
  const commitCount = data?.commitCount ?? null
  const showLabel = commitCount !== null && lod >= 1
  const showTicks = commitCount !== null && lod >= 2

  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: GRAPH_EDGE_BORDER_RADIUS,
  })

  const edgeStyle: CSSProperties = {
    ...style,
    transition: `stroke ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-width ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-opacity ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease`,
    strokeDasharray: showTicks ? dashPattern(commitCount ?? 0) : undefined,
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
