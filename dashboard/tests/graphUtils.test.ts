import test from "node:test"
import assert from "node:assert/strict"
import type { EpicGraph } from "../src/api.ts"
import { displayBranchLabel, formatBranchName } from "../src/lib/graph-utils.ts"

type GraphNode = EpicGraph["tasks"][number]

function makeNode(overrides: Partial<GraphNode>): GraphNode {
  return {
    id: 1,
    epicId: 1,
    title: "T-1 Example task",
    branchName: null,
    parentTaskId: null,
    worktreePath: null,
    githubPrId: null,
    githubIssueId: null,
    readme: null,
    source: "local",
    authority: "local",
    state: "todo",
    linearIssueId: null,
    linearIdentifier: null,
    localPath: null,
    createdAt: "2026-01-04T00:00:00Z",
    updatedAt: "2026-01-04T00:00:00Z",
    agentId: null,
    ...overrides,
  }
}

test("formatBranchName: strips epic prefix when present", () => {
  assert.equal(
    formatBranchName(
      "rn/harness-interface-v0/T-2-harness-identification",
      "harness-interface-v0",
    ),
    "T-2-harness-identification",
  )
  assert.equal(formatBranchName("main", "harness-interface-v0"), "main")
})

test("displayBranchLabel: uses formatted branch name when present", () => {
  const node = makeNode({
    id: 2,
    branchName: "rn/harness-interface-v0/T-2-harness-identification",
  })
  assert.deepEqual(displayBranchLabel(node, "harness-interface-v0"), {
    label: "T-2-harness-identification",
    provisional: false,
  })
})

test("displayBranchLabel: derives a concise provisional label from the title", () => {
  const node = makeNode({
    id: 75,
    title: "T-2 Harness identification + user override (Codex/Claude/Generic)",
    branchName: null,
  })
  assert.deepEqual(displayBranchLabel(node, "harness-interface-v0"), {
    label: "T-2-harness-identification",
    provisional: true,
  })
})

test("displayBranchLabel: drops boilerplate like 'implementation' and parentheticals", () => {
  const node = makeNode({
    id: 76,
    title:
      "T-3 Codex harness interface implementation (turn detection + capabilities)",
    branchName: null,
  })
  assert.deepEqual(displayBranchLabel(node, "harness-interface-v0"), {
    label: "T-3-codex-harness-interface",
    provisional: true,
  })
})
