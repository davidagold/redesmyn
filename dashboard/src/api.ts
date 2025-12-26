import type { components } from "@/api/v1"

export type ApiStatus = components["schemas"]["ApiStatusResponse"]
export type Epic = components["schemas"]["EpicResponse"]
export type EpicGraph = components["schemas"]["EpicGraphResponse"]

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
