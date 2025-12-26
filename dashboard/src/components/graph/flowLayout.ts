import type { GraphNode } from "@/lib/graph-utils"

export interface FlowPosition {
  x: number
  y: number
}

export interface FlowLayoutOptions {
  xSpacing: number
  ySpacing: number
  xOffset?: number
  yOffset?: number
}

export function layoutTree(
  rootNodes: GraphNode[],
  childrenByParent: Map<number | null, GraphNode[]>,
  { xSpacing, ySpacing, xOffset = 0, yOffset = 0 }: FlowLayoutOptions,
): Map<number, FlowPosition> {
  const positions = new Map<number, FlowPosition>()

  function walk(node: GraphNode, depth: number, row: number): number {
    positions.set(node.id, {
      x: xOffset + depth * xSpacing,
      y: yOffset + row * ySpacing,
    })

    let nextRow = row + 1
    for (const child of childrenByParent.get(node.id) ?? []) {
      nextRow = walk(child, depth + 1, nextRow)
    }
    return nextRow
  }

  let row = 0
  for (const root of rootNodes) {
    row = walk(root, 0, row)
  }

  return positions
}
