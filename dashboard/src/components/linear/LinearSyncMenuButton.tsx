import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { fetchLinearStatus, type LinearPushStats, type SyncStats } from "@/api"
import {
  useLinearLogoutMutation,
  useSyncFromLinearMutation,
  useSyncToLinearMutation,
} from "@/api/mutations"
import { useLinearStatus } from "@/hooks/useLinearStatus"
import type { Task } from "@/lib/graph-utils"
import {
  ArrowDownToLine,
  ArrowUpToLine,
  ChevronDown,
  ExternalLink,
  LogIn,
  LogOut,
} from "lucide-react"
import { LinearIcon } from "@/components/linear/LinearIcon"

type Notice = {
  kind: "info" | "success" | "error"
  message: string
}

type LinearStatusChange = {
  connected: boolean
  loading: boolean
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function formatSyncStats(stats: SyncStats) {
  return `tasks +${stats.tasksCreated}/~${stats.tasksUpdated}, branches +${stats.branchesCreated}/~${stats.branchesUpdated}`
}

function formatPushStats(stats: LinearPushStats) {
  return `issues +${stats.issuesCreated}/~${stats.issuesUpdated}, docs ~${stats.docsUpdated}, blockers ~${stats.blockersUpdated} (skipped ${stats.blockersSkipped})`
}

interface LinearSyncMenuButtonProps {
  epicId: number | null
  epicSlug: string
  variant: "epic" | "task"
  task?: Task | null
  onSynced?: () => Promise<void> | void
  showStatusDot?: boolean
  buttonClassName?: string
  onStatusChange?: (status: LinearStatusChange) => void
}

export function LinearSyncMenuButton({
  epicId,
  epicSlug,
  variant,
  task,
  onSynced,
  showStatusDot = true,
  buttonClassName,
  onStatusChange,
}: LinearSyncMenuButtonProps) {
  const logoutMutation = useLinearLogoutMutation()
  const syncFromMutation = useSyncFromLinearMutation()
  const syncToMutation = useSyncToLinearMutation()

  const [menuOpen, setMenuOpen] = useState(false)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const noticeTimeout = useRef<number | null>(null)

  const {
    status,
    loading: statusLoading,
    refresh: refreshStatus,
  } = useLinearStatus()

  const connected = status?.connected ?? false
  const issueId = task?.linearIssueId ?? null

  const statusLabel = useMemo(() => {
    if (statusLoading) {
      return "Checking…"
    }
    return connected ? "Connected" : "Not connected"
  }, [connected, statusLoading])

  const statusDotClass = useMemo(() => {
    if (statusLoading) {
      return "bg-muted-foreground/40"
    }
    return connected ? "bg-emerald-500" : "bg-muted-foreground/40"
  }, [connected, statusLoading])

  useEffect(() => {
    if (!notice) {
      if (noticeTimeout.current) {
        window.clearTimeout(noticeTimeout.current)
        noticeTimeout.current = null
      }
      return
    }

    noticeTimeout.current = window.setTimeout(() => {
      setNotice(null)
      noticeTimeout.current = null
    }, 6000)

    return () => {
      if (noticeTimeout.current) {
        window.clearTimeout(noticeTimeout.current)
        noticeTimeout.current = null
      }
    }
  }, [notice])

  useEffect(() => {
    onStatusChange?.({ connected, loading: statusLoading })
  }, [connected, onStatusChange, statusLoading])

  async function handleConnect() {
    setMenuOpen(false)
    setBusyAction("connect")
    setNotice({
      kind: "info",
      message: "Complete authorization in the new tab…",
    })

    const popup = window.open("/v1/linear/oauth/start", "_blank", "noreferrer")
    if (!popup) {
      setBusyAction(null)
      setNotice({
        kind: "error",
        message: "Popup blocked. Allow popups and try again.",
      })
      return
    }

    const deadline = Date.now() + 2 * 60 * 1000
    while (Date.now() < deadline) {
      try {
        const next = await fetchLinearStatus()
        if (next.connected) {
          await refreshStatus()
          setNotice({ kind: "success", message: "Linear connected." })
          setBusyAction(null)
          return
        }
      } catch {
        // ignore polling errors; user may be mid-auth
      }
      await sleep(1000)
    }

    setBusyAction(null)
    setNotice({
      kind: "error",
      message: "Timed out waiting for Linear authorization.",
    })
  }

  async function handleDisconnect() {
    setMenuOpen(false)
    setBusyAction("disconnect")
    try {
      await logoutMutation.mutateAsync()
      setNotice({ kind: "success", message: "Disconnected from Linear." })
    } catch (e) {
      setNotice({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setBusyAction(null)
    }
  }

  async function handleSyncFromLinear() {
    setMenuOpen(false)
    setBusyAction("sync-from")
    try {
      const stats = await syncFromMutation.mutateAsync({ epicId, epicSlug })
      try {
        await onSynced?.()
      } catch {
        // ignore refresh failures; sync may have still succeeded
      }
      setNotice({
        kind: "success",
        message: `Synced from Linear: ${formatSyncStats(stats)}`,
      })
    } catch (e) {
      setNotice({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setBusyAction(null)
    }
  }

  async function handleSyncToLinear() {
    setMenuOpen(false)
    setBusyAction("sync-to")
    try {
      const stats = await syncToMutation.mutateAsync({ epicId, epicSlug })
      try {
        await onSynced?.()
      } catch {
        // ignore refresh failures; sync may have still succeeded
      }
      setNotice({
        kind: "success",
        message: `Synced to Linear: ${formatPushStats(stats)}`,
      })
    } catch (e) {
      setNotice({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setBusyAction(null)
    }
  }

  function handleOpenInLinear() {
    setMenuOpen(false)

    if (variant === "epic") {
      window.open(`/v1/epics/${epicSlug}/linear/open`, "_blank", "noreferrer")
      return
    }

    if (!issueId) {
      setNotice({
        kind: "error",
        message: "This task is not linked to a Linear issue yet.",
      })
      return
    }
    window.open(`/v1/linear/issues/${issueId}/open`, "_blank", "noreferrer")
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        className={cn("h-7 gap-2 px-2", buttonClassName)}
        onClick={() => setMenuOpen((open) => !open)}
        aria-expanded={menuOpen}
        disabledReason={busyAction ? "Working…" : null}
        title={`Linear (${statusLabel})`}
      >
        <span className="inline-flex items-center gap-2">
          <LinearIcon className="size-3.5 text-muted-foreground" />
          <span>Linear</span>
          {showStatusDot ? (
            <span
              className={cn("size-2 rounded-full", statusDotClass)}
              aria-hidden="true"
            />
          ) : null}
        </span>
        <ChevronDown className="h-3 w-3 text-muted-foreground" />
      </Button>

      {notice ? (
        <div className="absolute right-0 top-full z-30 mt-2 w-72 rounded-lg border bg-popover px-3 py-2 text-xs shadow">
          <div
            className={cn(
              notice.kind === "error" && "text-destructive",
              notice.kind !== "error" && "text-muted-foreground",
            )}
          >
            {notice.message}
          </div>
        </div>
      ) : null}

      {menuOpen ? (
        <>
          <button
            type="button"
            aria-label="Close Linear menu"
            className="fixed inset-0 z-10 cursor-default bg-transparent"
            onClick={() => setMenuOpen(false)}
          />
          <div className="absolute right-0 top-full z-20 mt-2 w-72 rounded-lg border bg-popover p-2 shadow">
            <div className="grid gap-1">
              {connected ? (
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => void handleDisconnect()}
                  disabledReason={busyAction ? "Working…" : null}
                >
                  <LogOut className="h-3.5 w-3.5" />
                  Disconnect
                </Button>
              ) : (
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => void handleConnect()}
                  disabledReason={busyAction ? "Working…" : null}
                >
                  <LogIn className="h-3.5 w-3.5" />
                  Connect
                </Button>
              )}

              <div className="my-1 h-px bg-border/50" />

              <Button
                variant="ghost"
                className="w-full justify-start"
                onClick={() => void handleSyncFromLinear()}
                disabledReason={
                  busyAction
                    ? "Working…"
                    : !connected
                      ? "Connect Linear to sync"
                      : null
                }
              >
                <ArrowDownToLine className="h-3.5 w-3.5" />
                Sync from Linear
              </Button>

              <Button
                variant="ghost"
                className="w-full justify-start"
                onClick={() => void handleSyncToLinear()}
                disabledReason={
                  busyAction
                    ? "Working…"
                    : !connected
                      ? "Connect Linear to sync"
                      : null
                }
              >
                <ArrowUpToLine className="h-3.5 w-3.5" />
                Sync to Linear
              </Button>

              <div className="my-1 h-px bg-border/50" />

              <Button
                variant="ghost"
                className="w-full justify-start"
                onClick={handleOpenInLinear}
                disabledReason={
                  busyAction
                    ? "Working…"
                    : !connected
                      ? "Connect Linear to open"
                      : null
                }
              >
                <ExternalLink className="h-3.5 w-3.5" />
                {variant === "epic"
                  ? "Open project in Linear"
                  : "Open issue in Linear"}
              </Button>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
