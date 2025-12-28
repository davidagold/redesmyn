import type { EpicGraph } from "@/api"

export type GraphNode = EpicGraph["nodes"][number]
export type Task = EpicGraph["tasks"][number]
export type Agent = EpicGraph["agents"][number]
export type AgentSession = NonNullable<EpicGraph["sessions"]>[number]
export type TrunkTimeline = EpicGraph["trunk"]

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

export function buildAgentsMap(agents: Agent[]): Map<number, Agent> {
  const map = new Map<number, Agent>()
  for (const agent of agents) {
    map.set(agent.id, agent)
  }
  return map
}

export function buildSessionsByNodeId(
  sessions: AgentSession[],
): Map<number, AgentSession> {
  const map = new Map<number, AgentSession>()
  for (const session of sessions) {
    if (session.nodeId === null) {
      continue
    }
    map.set(session.nodeId, session)
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

export function buildChildrenMap(
  nodes: GraphNode[],
): Map<number | null, GraphNode[]> {
  const map = new Map<number | null, GraphNode[]>()
  for (const node of nodes) {
    const key = node.parentNodeId ?? null
    map.set(key, [...(map.get(key) ?? []), node])
  }
  for (const [key, value] of map.entries()) {
    value.sort((a, b) => a.id - b.id)
    map.set(key, value)
  }
  return map
}
