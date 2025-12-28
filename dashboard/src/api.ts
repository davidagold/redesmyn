import type { components } from "@/api/v1"

export type ApiStatus = components["schemas"]["ApiStatusResponse"]
export type Epic = components["schemas"]["EpicResponse"]
export type EpicGraph = components["schemas"]["EpicGraphResponse"]
export type Agent = components["schemas"]["AgentResponse"]
export type Node = components["schemas"]["NodeResponse"]
export type TaskAgentRestartRequest = components["schemas"]["TaskAgentRestartRequest"]
export type TaskAgentStartRequest = components["schemas"]["TaskAgentStartRequest"]
export type TaskAgentStartResponse = components["schemas"]["TaskAgentStartResponse"]
export type TaskAgentStopResponse = components["schemas"]["TaskAgentStopResponse"]

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

export async function startTaskAgent(
  taskId: number,
  request: TaskAgentStartRequest,
): Promise<TaskAgentStartResponse> {
  const response = await fetch(`/v1/tasks/${taskId}/agent/start`, {
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
      `POST /v1/tasks/${taskId}/agent/start failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<TaskAgentStartResponse>
}

export async function stopTaskAgent(
  taskId: number,
): Promise<TaskAgentStopResponse> {
  const response = await fetch(`/v1/tasks/${taskId}/agent/stop`, {
    method: "POST",
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `POST /v1/tasks/${taskId}/agent/stop failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<TaskAgentStopResponse>
}

export async function restartTaskAgent(
  taskId: number,
  request: TaskAgentRestartRequest = { detach: true },
): Promise<TaskAgentStartResponse> {
  const response = await fetch(`/v1/tasks/${taskId}/agent/restart`, {
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
      `POST /v1/tasks/${taskId}/agent/restart failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<TaskAgentStartResponse>
}
