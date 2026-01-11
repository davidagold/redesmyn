import { useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import {
  useGitHubLogoutMutation,
  useUpdateGitHubIntegrationConfigMutation,
} from "@/api/mutations"
import { useGitHubStatus } from "@/hooks/useGitHubStatus"
import { cn } from "@/lib/utils"
import { Copy, ExternalLink, Github, LogIn, LogOut } from "lucide-react"

type Notice = {
  kind: "info" | "success" | "error"
  message: string
}

export type GitHubStatusChange = {
  connected: boolean
  loading: boolean
  warning: boolean
}

export type GitHubMenuButtonProps = {
  showStatusDot?: boolean
  buttonClassName?: string
  onStatusChange?: (status: GitHubStatusChange) => void
}

function repoUrl(repoFullName: string) {
  return `https://github.com/${repoFullName}`
}

function connectCommand() {
  return "rn github auth"
}

export function GitHubMenuButton({
  showStatusDot = true,
  buttonClassName,
  onStatusChange,
}: GitHubMenuButtonProps) {
  const logoutMutation = useGitHubLogoutMutation()
  const configMutation = useUpdateGitHubIntegrationConfigMutation()

  const [menuOpen, setMenuOpen] = useState(false)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [notice, setNotice] = useState<Notice | null>(null)
  const noticeTimeout = useRef<number | null>(null)

  const {
    status,
    loading: statusLoading,
    refresh: refreshStatus,
  } = useGitHubStatus()

  const connected = status?.connected ?? false
  const warning = status?.warning ?? false
  const repoFullName = status?.repoFullName ?? null
  const missingScopes = status?.missingPrScopes ?? null
  const autoForcePush = status?.autoForcePush ?? false

  const statusLabel = useMemo(() => {
    if (statusLoading) {
      return "Checking…"
    }
    if (!connected) {
      return "Not connected"
    }
    return warning ? "Connected (missing scopes)" : "Connected"
  }, [connected, statusLoading, warning])

  const statusDotClass = useMemo(() => {
    if (statusLoading) {
      return "bg-muted-foreground/40"
    }
    if (!connected) {
      return "bg-muted-foreground/40"
    }
    return warning ? "bg-amber-300" : "bg-emerald-500"
  }, [connected, statusLoading, warning])

  useEffect(() => {
    onStatusChange?.({ connected, loading: statusLoading, warning })
  }, [connected, onStatusChange, statusLoading, warning])

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

  async function handleCopyCommand(text: string, successMessage: string) {
    try {
      await navigator.clipboard.writeText(text)
      setNotice({ kind: "success", message: successMessage })
    } catch {
      setNotice({ kind: "error", message: "Clipboard unavailable." })
    }
  }

  async function handleDisconnect() {
    setMenuOpen(false)
    setBusyAction("disconnect")
    try {
      await logoutMutation.mutateAsync()
      await refreshStatus()
      setNotice({ kind: "success", message: "Disconnected from GitHub." })
    } catch (e) {
      setNotice({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      })
    } finally {
      setBusyAction(null)
    }
  }

  async function handleToggleAutoForcePush(next: boolean) {
    setBusyAction("toggle-force-push")
    try {
      await configMutation.mutateAsync({ autoForcePush: next })
      await refreshStatus()
      setNotice({
        kind: "success",
        message: `Auto force-push ${next ? "enabled" : "disabled"}.`,
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

  function handleOpenRepo() {
    setMenuOpen(false)
    if (!repoFullName) {
      return
    }
    window.open(repoUrl(repoFullName), "_blank", "noreferrer")
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="icon"
        className={cn("h-7 w-7", buttonClassName)}
        onClick={() => setMenuOpen((open) => !open)}
        aria-expanded={menuOpen}
        aria-label={`GitHub (${statusLabel})`}
        disabledReason={busyAction ? "Working…" : null}
        title={`GitHub (${statusLabel})`}
      >
        <span className="relative inline-flex items-center justify-center">
          <Github className="size-3.5 text-muted-foreground" />
          {showStatusDot ? (
            <span
              className={cn(
                "absolute -right-0.5 -top-0.5 size-2 rounded-full border border-background",
                statusDotClass,
              )}
              aria-hidden="true"
            />
          ) : null}
        </span>
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
            aria-label="Close GitHub menu"
            className="fixed inset-0 z-10 cursor-default bg-transparent"
            onClick={() => setMenuOpen(false)}
          />
          <div className="absolute right-0 top-full z-20 mt-2 w-80 rounded-lg border bg-popover p-2 shadow">
            <div className="space-y-2 px-2 py-1 text-xs">
              <div className="flex items-start justify-between gap-2">
                <div className="text-muted-foreground">Status</div>
                <div className="text-foreground">{statusLabel}</div>
              </div>
              {repoFullName ? (
                <div className="flex items-start justify-between gap-2">
                  <div className="text-muted-foreground">Repo</div>
                  <div className="font-mono text-foreground">
                    {repoFullName}
                  </div>
                </div>
              ) : null}
              {warning ? (
                <div className="text-muted-foreground">
                  Missing GitHub scopes for PRs
                  {missingScopes && missingScopes.length ? (
                    <span className="text-muted-foreground">
                      {": "}
                      <span className="font-mono">
                        {missingScopes.join(", ")}
                      </span>
                    </span>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="my-2 h-px bg-border/50" />

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
                  onClick={() => {
                    setMenuOpen(false)
                    setNotice({
                      kind: "info",
                      message: `Run \`${connectCommand()}\` in your terminal to connect.`,
                    })
                  }}
                  disabledReason={busyAction ? "Working…" : null}
                >
                  <LogIn className="h-3.5 w-3.5" />
                  Connect
                </Button>
              )}

              <Button
                variant="ghost"
                className="w-full justify-start"
                onClick={() =>
                  void handleCopyCommand(connectCommand(), "Command copied")
                }
              >
                <Copy className="h-3.5 w-3.5" />
                Copy connect command
              </Button>

              <div className="my-1 h-px bg-border/50" />

              <Button
                variant="ghost"
                className="w-full justify-start"
                onClick={handleOpenRepo}
                disabledReason={
                  busyAction
                    ? "Working…"
                    : !repoFullName
                      ? "Repo not detected"
                      : null
                }
              >
                <ExternalLink className="h-3.5 w-3.5" />
                Open repo on GitHub
              </Button>

              <div className="my-1 h-px bg-border/50" />

              <div className="flex items-center justify-between gap-3 rounded-md px-2 py-1.5">
                <div className="text-xs text-muted-foreground">
                  Auto force-push
                </div>
                <Switch
                  checked={autoForcePush}
                  onCheckedChange={(checked) =>
                    void handleToggleAutoForcePush(checked)
                  }
                  disabledReason={
                    busyAction
                      ? "Working…"
                      : configMutation.isPending
                        ? "Saving…"
                        : null
                  }
                />
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  )
}
