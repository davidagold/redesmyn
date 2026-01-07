import type { EpicGraph } from "@/api"

export type GraphNode = EpicGraph["tasks"][number]
export type Task = EpicGraph["tasks"][number]
export type MergeRun = NonNullable<EpicGraph["mergeRuns"]>[number]
export type AgentSession = EpicGraph["agentSessions"][number]
export type TrunkTimeline = EpicGraph["trunk"]
export type RepoExecutorStatus = EpicGraph["repoExecutor"]
export type BranchLabel = {
  label: string
  provisional: boolean
}

export type SpineResolution = {
  spineTaskIds: number[]
  warnings: string[]
}

export type SpineSplit = {
  merged: number[]
  active: number[]
}

export type ActiveMergeSpineResult = {
  activeSpineTaskIds: number[]
  warnings: string[]
}

export function formatBranchName(
  branchName: string,
  epicSlug?: string | null,
): string {
  if (!epicSlug) {
    return branchName
  }
  const prefix = `rn/${epicSlug}/`
  return branchName.startsWith(prefix)
    ? branchName.slice(prefix.length)
    : branchName
}

function _slugifyBranchSegment(raw: string): string {
  const collapsed = raw
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "")
  return collapsed
}

export function displayBranchLabel(
  node: GraphNode,
  epicSlug?: string | null,
): BranchLabel {
  if (node.branchName) {
    return {
      label: formatBranchName(node.branchName, epicSlug),
      provisional: false,
    }
  }

  const title = node.title ?? ""
  const prefixMatch = title.match(/^(T-\d+)\s+(.*)$/)
  const identifier = prefixMatch?.[1] ?? `task-${node.id}`
  const titleRemainderRaw = (prefixMatch?.[2] ?? title).trim()
  const titleRemainderSansParens = titleRemainderRaw
    .replace(/\([^)]*\)/g, " ")
    .replace(/\s+/g, " ")
    .trim()
  const firstSegment = titleRemainderSansParens.split("+")[0]?.trim() ?? ""
  const titleRemainder = firstSegment
    .replace(/\bimplementation\b/gi, "")
    .replace(/\s+/g, " ")
    .trim()
  const slug = _slugifyBranchSegment(titleRemainder)
  const short = (slug.slice(0, 60).replace(/-+$/g, "") || "task").trim()

  return { label: `${identifier}-${short}`, provisional: true }
}

export function makeEdgeId(fromNodeId: number, toNodeId: number): string {
  return `edge:${fromNodeId}:${toNodeId}`
}

export function buildTasksMap(tasks: Task[]): Map<number, Task> {
  const map = new Map<number, Task>()
  for (const task of tasks) {
    map.set(task.id, task)
  }
  return map
}

export function resolveSpineTaskIds(
  tasksById: Map<number, Task>,
  leafTaskId: number,
): SpineResolution {
  const warnings: string[] = []
  const visited = new Set<number>()

  const spineLeafToRoot: number[] = [leafTaskId]
  visited.add(leafTaskId)

  let cursor = leafTaskId
  while (true) {
    const task = tasksById.get(cursor)
    if (!task) {
      warnings.push(`Could not resolve merge spine: task ${cursor} missing.`)
      return { spineTaskIds: [leafTaskId], warnings }
    }

    const parentId = task.parentTaskId ?? null
    if (parentId === null) {
      break
    }

    if (visited.has(parentId)) {
      warnings.push("Could not resolve merge spine: detected a cycle.")
      return { spineTaskIds: [leafTaskId], warnings }
    }

    if (!tasksById.has(parentId)) {
      warnings.push(
        `Could not resolve merge spine: missing parent task ${parentId}.`,
      )
      return { spineTaskIds: [leafTaskId], warnings }
    }

    spineLeafToRoot.push(parentId)
    visited.add(parentId)
    cursor = parentId
  }

  spineLeafToRoot.reverse()
  return { spineTaskIds: spineLeafToRoot, warnings }
}

export function splitMergedSpinePrefix(
  tasksById: Map<number, Task>,
  spineTaskIds: number[],
): SpineSplit {
  const merged: number[] = []
  for (const taskId of spineTaskIds) {
    const task = tasksById.get(taskId)
    if (!task || task.state !== "done") {
      break
    }
    merged.push(taskId)
  }
  const active = spineTaskIds.slice(merged.length)
  return { merged, active }
}

export function computeActiveMergeSpineTaskIds(
  tasksById: Map<number, Task>,
  leafTaskId: number,
): ActiveMergeSpineResult {
  // Best-effort client-side spine walk used for UX (counts/confirmations).
  // The server remains the source of truth for spine resolution.
  const { spineTaskIds, warnings } = resolveSpineTaskIds(tasksById, leafTaskId)
  if (warnings.length > 0) {
    return { activeSpineTaskIds: [leafTaskId], warnings }
  }
  const { active } = splitMergedSpinePrefix(tasksById, spineTaskIds)
  return { activeSpineTaskIds: active, warnings }
}

export type MergeReadySpinePreview = {
  count: number
  warnings: string[]
}

export function previewMergeReadySpineMarkCount(
  tasksById: Map<number, Task>,
  leafTaskId: number,
): MergeReadySpinePreview {
  const { activeSpineTaskIds, warnings } = computeActiveMergeSpineTaskIds(
    tasksById,
    leafTaskId,
  )
  if (warnings.length > 0) {
    return { count: 1, warnings }
  }

  let count = 0
  for (const taskId of activeSpineTaskIds) {
    const task = tasksById.get(taskId) ?? null
    if (!task || task.state === "done") {
      continue
    }
    if (!task.branchName) {
      continue
    }
    if (task.mergeReadyAt) {
      continue
    }
    count += 1
  }

  return { count: Math.max(count, 1), warnings }
}

export function buildAgentSessionsByNodeId(
  agentSessions: AgentSession[],
): Map<number, AgentSession> {
  const map = new Map<number, AgentSession>()
  for (const session of agentSessions) {
    if (session.taskId === null) {
      continue
    }
    map.set(session.taskId, session)
  }
  return map
}

export function buildNodesMap(nodes: GraphNode[]): Map<number, GraphNode> {
  const map = new Map<number, GraphNode>()
  for (const node of nodes) {
    map.set(node.id, node)
  }
  return map
}

export function buildMergeRunsByTaskId(
  runs: MergeRun[],
): Map<number, MergeRun> {
  const map = new Map<number, MergeRun>()
  for (const run of runs) {
    if (!map.has(run.requestedTaskId)) {
      map.set(run.requestedTaskId, run)
    }
  }
  return map
}

export function buildChildrenMap(
  nodes: GraphNode[],
): Map<number | null, GraphNode[]> {
  const map = new Map<number | null, GraphNode[]>()
  for (const node of nodes) {
    const key = node.parentTaskId ?? null
    map.set(key, [...(map.get(key) ?? []), node])
  }
  for (const [key, value] of map.entries()) {
    value.sort((a, b) => a.id - b.id)
    map.set(key, value)
  }
  return map
}
