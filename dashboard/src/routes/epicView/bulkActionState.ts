import { useEffect, useState } from "react"

const BULK_ACTION_TTL_MS = 10 * 60 * 1000

export type BulkActionKind = "run" | "stop"

export type BulkActionState = {
  runId: string
  kind: BulkActionKind
  total: number
  completed: number
  failed: number
  startedAt: number
}

export type BulkActionRequestItem = {
  taskId: number
  action: "start" | "restart" | "stop"
}

function bulkActionStorageKey(epicSlug: string) {
  return `redesmyn:bulk-action:${epicSlug}`
}

function readStoredBulkAction(epicSlug: string): BulkActionState | null {
  try {
    const raw = window.sessionStorage.getItem(bulkActionStorageKey(epicSlug))
    if (!raw) {
      return null
    }
    const parsed = JSON.parse(raw) as unknown
    if (!parsed || typeof parsed !== "object") {
      return null
    }
    const state = parsed as Partial<BulkActionState>
    if (
      typeof state.runId !== "string" ||
      (state.kind !== "run" && state.kind !== "stop") ||
      typeof state.total !== "number" ||
      typeof state.completed !== "number" ||
      typeof state.failed !== "number" ||
      typeof state.startedAt !== "number"
    ) {
      return null
    }
    if (
      !Number.isFinite(state.total) ||
      !Number.isFinite(state.completed) ||
      !Number.isFinite(state.failed) ||
      !Number.isFinite(state.startedAt)
    ) {
      return null
    }
    if (state.total <= 0 || state.completed < 0 || state.failed < 0) {
      return null
    }
    return state as BulkActionState
  } catch {
    return null
  }
}

export function usePersistedBulkAction(epicSlug: string | null) {
  const [bulkAction, setBulkAction] = useState<BulkActionState | null>(null)

  useEffect(() => {
    if (!epicSlug) {
      setBulkAction(null)
      return
    }

    const stored = readStoredBulkAction(epicSlug)
    if (!stored) {
      return
    }

    const ageMs = Date.now() - stored.startedAt
    if (
      ageMs < 0 ||
      ageMs > BULK_ACTION_TTL_MS ||
      stored.completed >= stored.total
    ) {
      try {
        window.sessionStorage.removeItem(bulkActionStorageKey(epicSlug))
      } catch {
        // ignore
      }
      return
    }

    setBulkAction(stored)
  }, [epicSlug])

  useEffect(() => {
    if (!epicSlug) {
      return
    }

    const key = bulkActionStorageKey(epicSlug)
    try {
      if (bulkAction === null) {
        window.sessionStorage.removeItem(key)
      } else {
        window.sessionStorage.setItem(key, JSON.stringify(bulkAction))
      }
    } catch {
      // ignore
    }
  }, [bulkAction, epicSlug])

  return { bulkAction, setBulkAction }
}
