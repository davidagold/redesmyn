import test from "node:test"
import assert from "node:assert/strict"
import {
  computeNodeSpan,
  type NodeLike,
} from "../src/components/graph/nodeSpan.ts"

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

