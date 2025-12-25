import type { components } from "@/api/v1"

export type ApiStatus = components["schemas"]["ApiStatusResponse"]

export async function fetchStatus(): Promise<ApiStatus> {
  const response = await fetch("/v1/status", {
    headers: { Accept: "application/json" },
  })
  if (!response.ok) {
    throw new Error(`GET /v1/status failed (${response.status})`)
  }
  return response.json() as Promise<ApiStatus>
}
