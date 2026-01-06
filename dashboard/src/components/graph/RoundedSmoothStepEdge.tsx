import { BaseEdge, getSmoothStepPath, type EdgeProps } from "@xyflow/react"
import { GRAPH_EDGE_BORDER_RADIUS } from "./graphConfig"

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

  return (
    <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
  )
}
