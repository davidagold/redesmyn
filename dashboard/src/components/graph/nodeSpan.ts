export type NodeLike = {
  id: number
  parentTaskId: number | null
}

export function computeNodeSpan(
  selectedNodeId: number,
  nodesById: Map<number, NodeLike>,
  childrenByParent: Map<number | null, NodeLike[]>,
): number[] | null {
  if (!nodesById.has(selectedNodeId)) {
    return null
  }

  const visitedUpstream = new Set<number>()
  let rootId: number = selectedNodeId
  while (!visitedUpstream.has(rootId)) {
    visitedUpstream.add(rootId)
    const parentId = nodesById.get(rootId)?.parentTaskId ?? null
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
    cursor = nodesById.get(cursor)?.parentTaskId ?? null
  }

  if (!focusPathUp.length || focusPathUp[focusPathUp.length - 1] !== rootId) {
    return null
  }

  focusPathUp.reverse()

  const visitedDownstream = new Set<number>(focusPathUp)
  const focusPathDown: number[] = []
  let downId: number = selectedNodeId

  while (true) {
    const children = childrenByParent.get(downId) ?? []
    if (children.length !== 1) {
      break
    }
    const nextId = children[0].id
    if (visitedDownstream.has(nextId)) {
      break
    }
    focusPathDown.push(nextId)
    visitedDownstream.add(nextId)
    downId = nextId
  }

  return [...focusPathUp, ...focusPathDown]
}
