import {
  DIAGONAL_BIAS_SLOPE,
  GRAPH_NODE_HEIGHT,
  SELECTION_LENS_CORRIDOR_GAP_PX,
  SELECTION_LENS_CORRIDOR_PADDING_PX,
} from "./graphConfig.ts"

export type NodeLike = {
  id: number
  parentNodeId: number | null
}

export type FlowPosition = {
  x: number
  y: number
}

export type NodeSpan = {
  focusPath: number[]
  focusSet: Set<number>
  focusRootId: number
  focusEndId: number
}

type SideGroup = {
  rootId: number
  nodeIds: number[]
  direction: "above" | "below"
  minY: number
  maxY: number
  offset: number
}

export function computeNodeSpan(
  selectedNodeId: number,
  nodesById: Map<number, NodeLike>,
  childrenByParent: Map<number | null, NodeLike[]>,
): NodeSpan | null {
  if (!nodesById.has(selectedNodeId)) {
    return null
  }

  const visitedUpstream = new Set<number>()
  let rootId: number = selectedNodeId
  while (!visitedUpstream.has(rootId)) {
    visitedUpstream.add(rootId)
    const parentId = nodesById.get(rootId)?.parentNodeId ?? null
    if (parentId === null) {
      break
    }
    rootId = parentId
  }

  const focusPathUp: number[] = []
  const visitedPath = new Set<number>()
  let cursor: number | null = selectedNodeId
  while (cursor !== null && !visitedPath.has(cursor)) {
    visitedPath.add(cursor)
    focusPathUp.push(cursor)
    if (cursor === rootId) {
      break
    }
    cursor = nodesById.get(cursor)?.parentNodeId ?? null
  }

  if (!focusPathUp.length || focusPathUp[focusPathUp.length - 1] !== rootId) {
    return null
  }

  focusPathUp.reverse()

  const visitedDownstream = new Set<number>(focusPathUp)
  const focusPathDown: number[] = []
  let endId: number = selectedNodeId
  let downId: number = selectedNodeId

  while (true) {
    const children = childrenByParent.get(downId) ?? []
    if (children.length !== 1) {
      endId = downId
      break
    }
    const nextId = children[0].id
    if (visitedDownstream.has(nextId)) {
      endId = downId
      break
    }
    focusPathDown.push(nextId)
    visitedDownstream.add(nextId)
    downId = nextId
  }

  const focusPath = [...focusPathUp, ...focusPathDown]
  return {
    focusPath,
    focusSet: new Set(focusPath),
    focusRootId: rootId,
    focusEndId: endId,
  }
}

export function computeSelectionLens(
  selectedNodeId: number | null,
  nodesById: Map<number, NodeLike>,
  childrenByParent: Map<number | null, NodeLike[]>,
): NodeSpan | null {
  if (selectedNodeId === null) {
    return null
  }
  const span = computeNodeSpan(selectedNodeId, nodesById, childrenByParent)
  if (!span) {
    return null
  }
  return span.focusPath.length >= 3 ? span : null
}

function collectSubtreeNodes<T extends NodeLike>(
  rootId: number,
  childrenByParent: Map<number | null, T[]>,
): number[] {
  const collected: number[] = []
  const stack = [rootId]
  const seen = new Set<number>()
  while (stack.length) {
    const next = stack.pop()
    if (next === undefined || seen.has(next)) {
      continue
    }
    seen.add(next)
    collected.push(next)
    const children = childrenByParent.get(next) ?? []
    for (const child of children) {
      stack.push(child.id)
    }
  }
  return collected
}

function intersectsBand(
  minY: number,
  maxY: number,
  bandTop: number,
  bandBottom: number,
) {
  return maxY >= bandTop && minY <= bandBottom
}

export function applySelectionLens(
  basePositions: Map<number, FlowPosition>,
  lens: NodeSpan,
  childrenByParent: Map<number | null, NodeLike[]>,
): Map<number, FlowPosition> {
  const rootPos = basePositions.get(lens.focusRootId)
  if (!rootPos) {
    return basePositions
  }

  const rootX = rootPos.x
  const focusShiftById = new Map<number, number>()
  for (const nodeId of lens.focusPath) {
    const pos = basePositions.get(nodeId)
    if (!pos) {
      continue
    }
    focusShiftById.set(nodeId, (pos.x - rootX) * DIAGONAL_BIAS_SLOPE)
  }

  let corridorTop = Number.POSITIVE_INFINITY
  let corridorBottom = Number.NEGATIVE_INFINITY
  for (const nodeId of lens.focusPath) {
    const pos = basePositions.get(nodeId)
    if (!pos) {
      continue
    }
    const shift = focusShiftById.get(nodeId) ?? 0
    corridorTop = Math.min(corridorTop, pos.y + shift)
    corridorBottom = Math.max(corridorBottom, pos.y + shift + GRAPH_NODE_HEIGHT)
  }

  if (!Number.isFinite(corridorTop) || !Number.isFinite(corridorBottom)) {
    return basePositions
  }

  corridorBottom += SELECTION_LENS_CORRIDOR_PADDING_PX

  const groups: SideGroup[] = []
  for (const focusNodeId of lens.focusPath) {
    const focusPos = basePositions.get(focusNodeId)
    if (!focusPos) {
      continue
    }
    const focusY = focusPos.y

    const children = childrenByParent.get(focusNodeId) ?? []
    for (const child of children) {
      if (lens.focusSet.has(child.id)) {
        continue
      }

      const childPos = basePositions.get(child.id)
      const childY = childPos?.y ?? focusY
      if (childY < focusY) {
        continue
      }

      const nodeIds = collectSubtreeNodes(child.id, childrenByParent)
      let minY = Number.POSITIVE_INFINITY
      let maxY = Number.NEGATIVE_INFINITY
      for (const nodeId of nodeIds) {
        const pos = basePositions.get(nodeId)
        if (!pos) {
          continue
        }
        minY = Math.min(minY, pos.y)
        maxY = Math.max(maxY, pos.y + GRAPH_NODE_HEIGHT)
      }

      if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
        continue
      }

      groups.push({
        rootId: child.id,
        nodeIds,
        direction: "below",
        minY,
        maxY,
        offset: 0,
      })
    }
  }

  for (const group of groups) {
    if (!intersectsBand(group.minY, group.maxY, corridorTop, corridorBottom)) {
      continue
    }
    group.offset = corridorBottom - group.minY + SELECTION_LENS_CORRIDOR_GAP_PX
  }

  const belowGroups = groups.sort((a, b) => {
    const aMin = a.minY + a.offset
    const bMin = b.minY + b.offset
    if (aMin !== bMin) {
      return aMin - bMin
    }
    return a.rootId - b.rootId
  })

  let cursorBelow = corridorBottom + SELECTION_LENS_CORRIDOR_GAP_PX
  for (const group of belowGroups) {
    const minY = group.minY + group.offset
    const maxY = group.maxY + group.offset
    if (minY < cursorBelow) {
      group.offset += cursorBelow - minY
    }
    cursorBelow = maxY + SELECTION_LENS_CORRIDOR_GAP_PX
  }

  const offsetByNodeId = new Map<number, number>()
  for (const group of groups) {
    for (const nodeId of group.nodeIds) {
      offsetByNodeId.set(nodeId, group.offset)
    }
  }

  const lensPositions = new Map<number, FlowPosition>()
  for (const [nodeId, pos] of basePositions) {
    if (lens.focusSet.has(nodeId)) {
      const shift = focusShiftById.get(nodeId) ?? 0
      lensPositions.set(nodeId, { x: pos.x, y: pos.y + shift })
      continue
    }

    const offset = offsetByNodeId.get(nodeId) ?? 0
    lensPositions.set(nodeId, { x: pos.x, y: pos.y + offset })
  }

  return lensPositions
}
