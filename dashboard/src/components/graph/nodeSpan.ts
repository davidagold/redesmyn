export type NodeLike = {
  id: number
  parentTaskId: number | null
}

export type NodeSpan = {
  focusPath: number[]
  focusSet: Set<number>
  focusRootId: number
  focusEndId: number
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

