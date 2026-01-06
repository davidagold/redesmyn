import test from "node:test"
import assert from "node:assert/strict"
import {
  computeRunToolbarModel,
  type RunToolbarAgentSession,
  type RunToolbarTask,
} from "../src/routes/epicView/runToolbarModel.ts"

function task(partial: Partial<RunToolbarTask> & Pick<RunToolbarTask, "id">) {
  return {
    id: partial.id,
    branchName:
      "branchName" in partial ? partial.branchName : `rn/ui-v0/T-${partial.id}`,
    state: partial.state ?? "todo",
    stackInSync: "stackInSync" in partial ? partial.stackInSync : true,
  } satisfies RunToolbarTask
}

function session(
  partial: Partial<RunToolbarAgentSession> & Pick<RunToolbarAgentSession, "status">,
) {
  return {
    status: partial.status,
  } satisfies RunToolbarAgentSession
}

test("computeRunToolbarModel: null tasks yields empty model", () => {
  const model = computeRunToolbarModel({
    tasks: null,
    tasksById: new Map(),
    agentSessionsByNodeId: new Map(),
    selectedNodeIds: new Set(),
    stackProjectionsFresh: true,
  })

  assert.equal(model.summary, null)
  assert.equal(model.buckets, null)
  assert.deepEqual(model.actionTargets.all, {
    start: [],
    restart: [],
    stop: [],
  })
  assert.deepEqual(model.actionTargets.selected, {
    start: [],
    restart: [],
    stop: [],
  })
  assert.deepEqual(model.harnessRequired, { all: false, selected: false })
})

test("computeRunToolbarModel: computes counts, buckets, and action targets", () => {
  const tasks: RunToolbarTask[] = [
    task({ id: 1 }),
    task({ id: 2, stackInSync: false }),
    task({ id: 3, state: "blocked", stackInSync: false }),
    task({ id: 4, state: "done" }),
    task({ id: 5, branchName: null }),
    task({ id: 6, state: "todo" }),
    task({ id: 7, state: "todo" }),
    task({ id: 8, state: "todo" }),
  ]

  const tasksById = new Map<number, RunToolbarTask>(tasks.map((t) => [t.id, t]))

  const agentSessionsByNodeId = new Map<number, RunToolbarAgentSession>([
    [1, session({ status: "running" })],
    [2, session({ status: "blocked" })],
    [6, session({ status: "error" })],
    [7, session({ status: "stopped" })],
  ])

  const model = computeRunToolbarModel({
    tasks,
    tasksById,
    agentSessionsByNodeId,
    selectedNodeIds: new Set([1, 5, 6, 7, 8]),
    stackProjectionsFresh: true,
  })

  assert.deepEqual(model.summary, {
    eligible: 5,
    running: 1,
    blocked: 2,
    failed: 1,
    outOfSync: 1,
  })

  assert.deepEqual(model.buckets, {
    eligible: [1, 2, 6, 7, 8],
    running: [1],
    blocked: [2, 3],
    failed: [6],
    outOfSync: [2, 3],
  })

  assert.deepEqual(model.actionTargets.all, {
    start: [7, 8],
    restart: [6],
    stop: [1, 2],
  })

  assert.deepEqual(model.actionTargets.selected, {
    start: [7, 8],
    restart: [6],
    stop: [1],
  })

  assert.deepEqual(model.harnessRequired, { all: true, selected: true })
})
