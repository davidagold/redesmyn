import ELK from "elkjs/lib/elk.bundled.js"
import type { GraphNode } from "@/lib/graph-utils"
import type { FlowPosition } from "./flowLayout"

export interface ElkLayoutOptions {
  nodeWidth: number
  nodeHeight: number
  xOffset?: number
  yOffset?: number
}

type ElkNode = {
  id: string
  width: number
  height: number
}

type ElkEdge = {
  id: string
  sources: [string]
  targets: [string]
}

type ElkGraph = {
  id: string
  layoutOptions: Record<string, string>
  children: ElkNode[]
  edges: ElkEdge[]
}

type LaidOutElkNode = ElkNode & {
  x?: number
  y?: number
}

type LaidOutElkGraph = ElkGraph & {
  children?: LaidOutElkNode[]
}

const elk = new ELK()

export async function layoutWithElk(
  nodes: GraphNode[],
  childrenByParent: Map<number | null, GraphNode[]>,
  { nodeWidth, nodeHeight, xOffset = 0, yOffset = 0 }: ElkLayoutOptions,
): Promise<Map<number, FlowPosition>> {
  const sortedNodes = [...nodes].sort((a, b) => a.id - b.id)

  const edges: { sourceId: number targetId: number }[] = []
  for (const node of sortedNodes) {
    if (node.parentNodeId === null) {
      continue
    }
    edges.push({ sourceId: node.parentNodeId, targetId: node.id })
  }
  edges.sort((a, b) =>
    a.sourceId !== b.sourceId
      ? a.sourceId - b.sourceId
      : a.targetId - b.targetId,
  )

  // Note: ELK is deterministic as long as our inputs are deterministic. Keep node/edge
  // ordering stable and avoid relying on JS object key ordering.
  const graph: ElkGraph = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "mrtree",
      "elk.direction": "RIGHT",
      "elk.spacing.nodeNode": "80",
      "elk.layered.spacing.nodeNodeBetweenLayers": "140",
      "elk.edgeRouting": "ORTHOGONAL",
    },
    children: sortedNodes.map((node) => ({
      id: String(node.id),
      width: nodeWidth,
      height: nodeHeight,
    })),
    edges: edges.map((edge) => ({
      id: `edge:${edge.sourceId}:${edge.targetId}`,
      sources: [String(edge.sourceId)],
      targets: [String(edge.targetId)],
    })),
  }

  const layout = (await elk.layout(graph)) as LaidOutElkGraph
  const positions = new Map<number, FlowPosition>()
  for (const child of layout.children ?? []) {
    const nodeId = Number(child.id)
    if (Number.isNaN(nodeId)) {
      continue
    }
    positions.set(nodeId, {
      x: xOffset + (child.x ?? 0),
      y: yOffset + (child.y ?? 0),
    })
  }

  // When ELK can't place something (e.g. disconnected nodes), fall back to a
  // deterministic first-seen traversal order.
  let fallbackRow = 0
  for (const root of childrenByParent.get(null) ?? []) {
    if (positions.has(root.id)) {
      continue
    }
    positions.set(root.id, {
      x: xOffset,
      y: yOffset + fallbackRow * nodeHeight,
    })
    fallbackRow += 1
  }

  return positions
}
