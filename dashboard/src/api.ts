import type { components } from "@/api/v1"

export type ApiStatus = components["schemas"]["ApiStatusResponse"]
export type Epic = components["schemas"]["EpicResponse"]
export type EpicGraph = components["schemas"]["EpicGraphResponse"]
export type Agent = components["schemas"]["AgentResponse"]
export type AgentSession = components["schemas"]["AgentSessionResponse"]
export type Node = components["schemas"]["NodeResponse"]
export type NodeRestartSessionRequest = components["schemas"]["NodeRestartSessionRequest"]
export type NodeStartSessionRequest = components["schemas"]["NodeStartSessionRequest"]

export async function fetchStatus(): Promise<ApiStatus> {
  const response = await fetch("/v1/status", {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    throw new Error(`GET /v1/status failed (${response.status})`)
  }
  return response.json() as Promise<ApiStatus>
}

export async function fetchEpics(): Promise<Epic[]> {
  const response = await fetch("/v1/epics", {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    throw new Error(`GET /v1/epics failed (${response.status})`)
  }
  return response.json() as Promise<Epic[]>
}

export async function fetchEpicGraph(
  epic: string | number,
): Promise<EpicGraph> {
  const response = await fetch(`/v1/epics/${epic}/graph`, {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    throw new Error(`GET /v1/epics/${epic}/graph failed (${response.status})`)
  }
  return response.json() as Promise<EpicGraph>
}

async function readErrorDetail(response: Response): Promise<string | null> {
  try {
    const payload = (await response.json()) as unknown
    if (
      payload &&
      typeof payload === "object" &&
      "detail" in payload &&
      typeof payload.detail === "string"
    ) {
      return payload.detail
    }
  } catch {
    // Ignore.
  }
  return null
}

export async function setNodeAgent(
  nodeId: number,
  agentId: number | null,
): Promise<Node> {
  const response = await fetch(`/v1/nodes/${nodeId}/agent`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ agent_id: agentId }),
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `POST /v1/nodes/${nodeId}/agent failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<Node>
}

export async function startNodeSession(
  nodeId: number,
  request: NodeStartSessionRequest,
): Promise<AgentSession> {
  const response = await fetch(`/v1/nodes/${nodeId}/session/start`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `POST /v1/nodes/${nodeId}/session/start failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<AgentSession>
}

export async function stopNodeSession(nodeId: number): Promise<AgentSession> {
  const response = await fetch(`/v1/nodes/${nodeId}/session/stop`, {
    method: "POST",
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `POST /v1/nodes/${nodeId}/session/stop failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<AgentSession>
}

export async function restartNodeSession(
  nodeId: number,
  request: NodeRestartSessionRequest = { detach: true },
): Promise<AgentSession> {
  const response = await fetch(`/v1/nodes/${nodeId}/session/restart`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `POST /v1/nodes/${nodeId}/session/restart failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<AgentSession>
}
