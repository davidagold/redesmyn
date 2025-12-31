import type { components } from "@/api/v1"

export type ApiStatus = components["schemas"]["ApiStatusResponse"]
export type Epic = components["schemas"]["EpicResponse"]
export type EpicGraph = components["schemas"]["EpicGraphResponse"]
export type OrchestrationDefaults = components["schemas"]["OrchestrationDefaultsResponse"]
export type OrchestrationDefaultsUpdateRequest = components["schemas"]["OrchestrationDefaultsUpdateRequest"]
export type Agent = components["schemas"]["AgentResponse"]
export type Node = components["schemas"]["NodeResponse"]
export type Task = components["schemas"]["TaskResponse"]
export type TaskAgentRestartRequest = components["schemas"]["TaskAgentRestartRequest"] & {
  prelude?: string | null
}
export type TaskAgentStartRequest = components["schemas"]["TaskAgentStartRequest"] & {
  prelude?: string | null
}
export type TaskAgentStartResponse = components["schemas"]["TaskAgentStartResponse"]
export type TaskAgentStopResponse = components["schemas"]["TaskAgentStopResponse"]
export type TaskAgentBulkRunRequest = components["schemas"]["TaskAgentBulkRunRequest"] & {
  prelude?: string | null
}
export type TaskAgentBulkRunResponse = components["schemas"]["TaskAgentBulkRunResponse"]
export type TaskAgentBulkActionRequest = components["schemas"]["TaskAgentBulkActionRequest"] & {
  prelude?: string | null
}
export type TaskAgentBulkActionResponse = components["schemas"]["TaskAgentBulkActionResponse"]

export type TaskAgentLogsResponse = {
  path: string
  text: string
  truncated: boolean
}

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

export async function setTaskMergeReady(
  taskId: number,
  ready: boolean,
): Promise<Task> {
  const response = await fetch(`/v1/tasks/${taskId}/merge-ready`, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ready }),
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `POST /v1/tasks/${taskId}/merge-ready failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<Task>
}

export async function fetchOrchestrationDefaults(): Promise<OrchestrationDefaults> {
  const response = await fetch("/v1/config", {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    throw new Error(`GET /v1/config failed (${response.status})`)
  }
  return response.json() as Promise<OrchestrationDefaults>
}

export async function updateOrchestrationDefaults(
  request: OrchestrationDefaultsUpdateRequest,
): Promise<OrchestrationDefaults> {
  const response = await fetch("/v1/config", {
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
      `POST /v1/config failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<OrchestrationDefaults>
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

export async function runTaskAgentsBulk(
  request: TaskAgentBulkRunRequest,
): Promise<TaskAgentBulkRunResponse> {
  const response = await fetch("/v1/tasks/agent/run", {
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
      `POST /v1/tasks/agent/run failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<TaskAgentBulkRunResponse>
}

export async function bulkTaskAgentActions(
  request: TaskAgentBulkActionRequest,
): Promise<TaskAgentBulkActionResponse> {
  const response = await fetch("/v1/tasks/agent/actions", {
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
      `POST /v1/tasks/agent/actions failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<TaskAgentBulkActionResponse>
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

export async function fetchTaskAgentLogs(
  taskId: number,
  params: {
    lines?: number
    maxBytes?: number
  } = {},
): Promise<TaskAgentLogsResponse> {
  const url = new URL(`/v1/tasks/${taskId}/agent/logs`, window.location.origin)
  if (typeof params.lines === "number") {
    url.searchParams.set("lines", String(params.lines))
  }
  if (typeof params.maxBytes === "number") {
    url.searchParams.set("max_bytes", String(params.maxBytes))
  }
  const response = await fetch(url.pathname + url.search, {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `GET /v1/tasks/${taskId}/agent/logs failed (${response.status})${
        detail ? `: ${detail}` : ""
      }`,
    )
  }
  return response.json() as Promise<TaskAgentLogsResponse>
}
