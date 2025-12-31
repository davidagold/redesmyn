import test from "node:test"
import assert from "node:assert/strict"
import {
  applySelectionLens,
  computeNodeSpan,
  computeSelectionLens,
  type FlowPosition,
  type NodeLike,
} from "../src/components/graph/selectionLens.ts"
import {
  GRAPH_NODE_HEIGHT,
  SELECTION_LENS_CORRIDOR_GAP_PX,
} from "../src/components/graph/graphConfig.ts"

type Node = NodeLike

function buildGraph() {
  const nodes: Node[] = [
    { id: 4, parentTaskId: null },
    { id: 5, parentTaskId: 4 },
    { id: 6, parentTaskId: 4 },
    { id: 7, parentTaskId: 4 },
    { id: 8, parentTaskId: 6 },
    { id: 9, parentTaskId: 8 },
    { id: 10, parentTaskId: 5 },
  ]

  const nodesById = new Map<number, Node>()
  for (const node of nodes) {
    nodesById.set(node.id, node)
  }

  const childrenByParent = new Map<number | null, Node[]>()
  for (const node of nodes) {
    const key = node.parentTaskId ?? null
    childrenByParent.set(key, [...(childrenByParent.get(key) ?? []), node])
  }
  for (const [key, value] of childrenByParent.entries()) {
    value.sort((a, b) => a.id - b.id)
    childrenByParent.set(key, value)
  }

  return { nodesById, childrenByParent }
}

function buildBasePositions(): Map<number, FlowPosition> {
  return new Map<number, FlowPosition>([
    [4, { x: 0, y: 200 }],
    [5, { x: 400, y: 80 }],
    [6, { x: 400, y: 200 }],
    [7, { x: 400, y: 320 }],
    [10, { x: 800, y: 80 }],
    [8, { x: 800, y: 200 }],
    [9, { x: 1200, y: 200 }],
  ])
}

function assertNoOverlapY(aY: number, bY: number, height: number, gap = 0) {
  const [topA, bottomA] = [aY, aY + height]
  const [topB, bottomB] = [bY, bY + height]
  const separated = bottomA + gap <= topB || bottomB + gap <= topA
  assert.ok(
    separated,
    `expected no overlap: [${topA},${bottomA}] vs [${topB},${bottomB}]`,
  )
}

test("computeNodeSpan: graph-viz topology", () => {
  const { nodesById, childrenByParent } = buildGraph()

  assert.deepEqual(
    computeNodeSpan(4, nodesById, childrenByParent)?.focusPath,
    [4],
  )
  assert.deepEqual(
    computeNodeSpan(5, nodesById, childrenByParent)?.focusPath,
    [4, 5, 10],
  )
  assert.deepEqual(
    computeNodeSpan(6, nodesById, childrenByParent)?.focusPath,
    [4, 6, 8, 9],
  )
  assert.deepEqual(
    computeNodeSpan(7, nodesById, childrenByParent)?.focusPath,
    [4, 7],
  )
  assert.deepEqual(
    computeNodeSpan(10, nodesById, childrenByParent)?.focusPath,
    [4, 5, 10],
  )
})

test("computeSelectionLens: ignores single-node spans", () => {
  const { nodesById, childrenByParent } = buildGraph()

  assert.equal(computeSelectionLens(4, nodesById, childrenByParent), null)
  assert.equal(computeSelectionLens(7, nodesById, childrenByParent), null)
  assert.ok(computeSelectionLens(5, nodesById, childrenByParent))
})

test("applySelectionLens: keeps non-focused branches horizontal + non-overlapping", () => {
  const { nodesById, childrenByParent } = buildGraph()
  const base = buildBasePositions()
  const lens = computeSelectionLens(5, nodesById, childrenByParent)
  assert.ok(lens)

  const positions = applySelectionLens(base, lens, childrenByParent)

  const y6 = positions.get(6)?.y
  const y7 = positions.get(7)?.y
  const y8 = positions.get(8)?.y
  const y9 = positions.get(9)?.y
  assert.equal(typeof y6, "number")
  assert.equal(typeof y7, "number")
  assert.equal(typeof y8, "number")
  assert.equal(typeof y9, "number")

  // Non-focused branch stays horizontally oriented (no diagonal skew applied to subtree).
  assert.equal(y6, y8)
  assert.equal(y6, y9)

  // Parallel siblings below the focused corridor must be packed to avoid collisions.
  assert.ok(y6 < y7, "expected vertical order to be preserved (6 above 7)")
  assertNoOverlapY(y6, y7, GRAPH_NODE_HEIGHT, SELECTION_LENS_CORRIDOR_GAP_PX)
})
