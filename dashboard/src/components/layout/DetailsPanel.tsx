import { useEffect, useMemo, useState } from "react"
import { createPortal } from "react-dom"
import { Markdown } from "@/components/markdown"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { SlidePanel } from "@/components/ui/slide-panel"
import {
  fetchTaskAgentLogs,
  restartTaskAgent,
  setTaskMergeReady,
  startTaskAgent,
  stopTaskAgent,
} from "@/api"
import type { Agent, GraphNode, MergeRun, Task } from "@/lib/graph-utils"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { copyToClipboard } from "@/lib/clipboard"
import { getRebaseRemediation } from "@/lib/merge-remediation"
import { cn } from "@/lib/utils"
import { AgentStatusBadge } from "@/components/agents/AgentStatusBadge"
import { FloatingActions } from "@/components/ui/floating-actions"
import {
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  MessageSquareText,
  Play,
  RotateCcw,
  Square,
  Terminal,
} from "lucide-react"
import { Switch } from "@/components/ui/switch"

type EdgeSelection = {
  id: string
  fromNodeId: number
  toNodeId: number
  fromLabel: string
  toLabel: string
  commitCount?: number | null
  baseSha?: string | null
  headSha?: string | null
}

function stripAnsi(text: string) {
  // CSI sequences (colors, cursor moves, etc.)
  // eslint-disable-next-line no-control-regex
  const withoutCsi = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
  // OSC sequences (titles, hyperlinks, etc.)
  // eslint-disable-next-line no-control-regex
  return withoutCsi.replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
}

interface DetailsPanelProps {
  open: boolean
  task: Task | null
  node?: GraphNode | null
  agent?: Agent | null
  mergeRun?: MergeRun | null
  onRequestRefresh?: () => void
  edge?: EdgeSelection | null
}

function MergeRunDetails({
  mergeRun,
  node,
}: {
  mergeRun: MergeRun
  node: GraphNode | null | undefined
}) {
  const [calloutExpanded, setCalloutExpanded] = useState(false)
  const [detailsExpanded, setDetailsExpanded] = useState(false)
  const remediation = getRebaseRemediation(mergeRun, node?.branchName ?? null)

  const statusLabel = mergeRun.status.replace(/^\w/, (c) => c.toUpperCase())
  const blockedError = mergeRun.blockedError?.trim() || null
  const blockedBranch = mergeRun.blockedBranchName ?? null
  const blockedWorktree = mergeRun.blockedWorktreePath ?? null
  const blockedStepKind = mergeRun.blockedStepKind ?? null

  const showCallout =
    mergeRun.status === "blocked" ||
    mergeRun.status === "resumable" ||
    mergeRun.status === "failed"

  const calloutTone =
    mergeRun.status === "blocked" || mergeRun.status === "failed"
      ? "destructive"
      : mergeRun.status === "resumable"
        ? "emerald"
        : "default"

  const calloutClasses =
    calloutTone === "destructive"
      ? "border border-destructive/25 bg-destructive/10 ring-destructive/10"
      : calloutTone === "emerald"
        ? "border border-emerald-400/25 bg-emerald-400/10 ring-emerald-400/10"
        : "border border-border/60 bg-background/30 ring-border/10"

  const calloutTitle =
    // When a descendant rebase conflicts, the merge run is blocked in the restack portion.
    // (The spine merge may or may not have completed depending on the merge mode.)
    mergeRun.status === "blocked"
      ? (mergeRun as { blockedOnSpine?: boolean | null }).blockedOnSpine ===
        false
        ? "Restack blocked (conflicts)"
        : "Merge blocked (conflicts)"
      : mergeRun.status === "resumable"
        ? "Merge ready to resume"
        : mergeRun.status === "failed"
          ? "Merge failed"
          : null

  const calloutIcon =
    mergeRun.status === "resumable" ? (
      <Play className="size-3.5 text-emerald-400" />
    ) : (
      <AlertTriangle className="size-3.5 text-destructive" />
    )

  const attachCommand = remediation?.attachCommand ?? null
  const agentNote = remediation?.message ?? null

  const statusBadgeClasses =
    mergeRun.status === "blocked" || mergeRun.status === "failed"
      ? "border-destructive/25 bg-destructive/10 text-destructive"
      : mergeRun.status === "resumable"
        ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-100"
        : "border-border/60 bg-background/30 text-muted-foreground"

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={cn(
            "inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium",
            statusBadgeClasses,
          )}
        >
          {statusLabel}
        </span>
        <span className="bg-background/30 text-muted-foreground inline-flex items-center rounded-full border border-border/60 px-2 py-0.5 text-xs capitalize">
          {mergeRun.scope}
        </span>
      </div>

      {showCallout && calloutTitle ? (
        <Collapsible
          open={calloutExpanded}
          onOpenChange={setCalloutExpanded}
          className="w-full"
        >
          <Card
            size="sm"
            className={cn(
              "w-full gap-2 py-2 shadow-sm backdrop-blur",
              calloutClasses,
            )}
          >
            <CardContent className="px-3">
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  {calloutIcon}
                  <div className="truncate text-xs font-medium text-foreground">
                    {calloutTitle}
                  </div>
                </div>
                <CollapsibleTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={
                        calloutExpanded ? "Collapse callout" : "Expand callout"
                      }
                      title={
                        calloutExpanded ? "Collapse callout" : "Expand callout"
                      }
                    >
                      {calloutExpanded ? (
                        <ChevronUp className="size-3" />
                      ) : (
                        <ChevronDown className="size-3" />
                      )}
                    </Button>
                  }
                />
              </div>

              {mergeRun.status === "blocked" && remediation ? (
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Button
                    variant="ghost"
                    size="xs"
                    disabledReason={
                      attachCommand
                        ? null
                        : "No task id recorded for this blocked step."
                    }
                    onClick={() => {
                      if (!attachCommand) {
                        return
                      }
                      void copyToClipboard(attachCommand)
                    }}
                  >
                    <Terminal className="size-3" />
                    Copy attach
                  </Button>
                  <Button
                    variant="ghost"
                    size="xs"
                    disabledReason={
                      agentNote ? null : "No remediation message available."
                    }
                    onClick={() => {
                      if (!agentNote) {
                        return
                      }
                      void copyToClipboard(agentNote)
                    }}
                  >
                    <MessageSquareText className="size-3" />
                    Copy agent note
                  </Button>
                </div>
              ) : null}

              <CollapsibleContent className="mt-2 w-full min-w-0">
                {blockedStepKind ? (
                  <div className="text-xs text-muted-foreground">
                    Step: <span className="font-mono">{blockedStepKind}</span>
                  </div>
                ) : null}

                {blockedBranch ? (
                  <div className="mt-1 text-xs text-muted-foreground">
                    Branch:{" "}
                    <span className="break-words font-mono text-foreground/80">
                      {blockedBranch}
                    </span>
                  </div>
                ) : null}

                {blockedWorktree ? (
                  <div className="mt-1 text-xs text-muted-foreground">
                    Worktree:{" "}
                    <span className="break-words font-mono text-foreground/80">
                      {blockedWorktree}
                    </span>
                  </div>
                ) : null}

                {blockedError ? (
                  <div className="mt-2">
                    <div className="flex items-center justify-between gap-2">
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => setDetailsExpanded((open) => !open)}
                      >
                        {detailsExpanded ? "Hide" : "Show"} details
                      </Button>
                      <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => void copyToClipboard(blockedError)}
                      >
                        Copy
                      </Button>
                    </div>
                    {detailsExpanded ? (
                      <div className="mt-2 max-h-48 overflow-auto rounded-md bg-background/40 px-2 py-1.5 font-mono text-[0.625rem] text-foreground/80">
                        <div className="whitespace-pre-wrap break-words">
                          {blockedError}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </CollapsibleContent>
            </CardContent>
          </Card>
        </Collapsible>
      ) : null}
    </div>
  )
}

function AgentActions({
  task,
  agent,
  onRequestRefresh,
  floatingActionsPortalId,
}: {
  task: Task
  agent: Agent | null
  onRequestRefresh: () => void
  floatingActionsPortalId?: string
}) {
  const { defaults: orchestrationDefaults } = useOrchestrationDefaults()
  const [pending, setPending] = useState<"start" | "stop" | "restart" | null>(
    null,
  )
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [logsOpen, setLogsOpen] = useState(false)
  const [logsPending, setLogsPending] = useState(false)
  const [logsError, setLogsError] = useState<string | null>(null)
  const [logsText, setLogsText] = useState<string | null>(null)
  const [logsPath, setLogsPath] = useState<string | null>(null)
  const [logsTruncated, setLogsTruncated] = useState<boolean>(false)
  const [oneTimePreludeOpen, setOneTimePreludeOpen] = useState(false)
  const [oneTimePrelude, setOneTimePrelude] = useState("")
  const [mergeReadyPending, setMergeReadyPending] = useState(false)
  const [mergeReady, setMergeReady] = useState<boolean>(
    Boolean(task.mergeReadyAt),
  )

  useEffect(() => {
    setPending(null)
    setError(null)
    setNotice(null)
    setLogsOpen(false)
    setLogsPending(false)
    setLogsError(null)
    setLogsText(null)
    setLogsPath(null)
    setLogsTruncated(false)
    setOneTimePreludeOpen(false)
    setOneTimePrelude("")
    setMergeReadyPending(false)
  }, [task.id])

  useEffect(() => {
    setMergeReady(Boolean(task.mergeReadyAt))
  }, [task.id, task.mergeReadyAt])

  useEffect(() => {
    if (!notice) {
      return
    }
    const id = window.setTimeout(() => setNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [notice])

  const taskId = task.id
  const hasAgent = agent !== null
  const agentName = useMemo(
    () => agent?.displayName ?? "No agent",
    [agent?.displayName],
  )
  const statusLabel = agent?.status ?? null
  const harnessKind = agent?.harnessProfileId?.split("/")[0] ?? null
  const isRunning = statusLabel === "running" || statusLabel === "blocked"
  const canRestart =
    statusLabel === "running" ||
    statusLabel === "blocked" ||
    statusLabel === "error"

  const configuredHarnessCommand =
    orchestrationDefaults?.harness.command?.trim() ?? ""
  const agentArgv = agent?.resolvedProfile?.argv ?? null
  const harnessCommand =
    agentArgv && agentArgv.length > 0
      ? agentArgv.join(" ")
      : configuredHarnessCommand

  const oneTimePreludeValue = oneTimePrelude.trim()

  async function handleStart() {
    setPending("start")
    setError(null)
    try {
      const response = await startTaskAgent(taskId, {
        harness: configuredHarnessCommand,
        detach: orchestrationDefaults?.harness.detach ?? true,
        prelude: oneTimePreludeValue ? oneTimePreludeValue : null,
      })
      const warnings = response.warnings ?? []
      if (warnings.length > 0) {
        if (!response.started || response.agentStatus === "error") {
          setError(warnings.join("\n"))
        } else {
          setNotice(warnings[0])
        }
      }
      setOneTimePrelude("")
      setOneTimePreludeOpen(false)
      onRequestRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
    }
  }

  async function handleStop() {
    setPending("stop")
    setError(null)
    try {
      await stopTaskAgent(taskId)
      onRequestRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
    }
  }

  async function handleRestart() {
    setPending("restart")
    setError(null)
    try {
      const response = await restartTaskAgent(taskId, {
        harness: null,
        detach: orchestrationDefaults?.harness.detach ?? true,
        prelude: oneTimePreludeValue ? oneTimePreludeValue : null,
      })
      const warnings = response.warnings ?? []
      if (warnings.length > 0) {
        if (!response.started || response.agentStatus === "error") {
          setError(warnings.join("\n"))
        } else {
          setNotice(warnings[0])
        }
      }
      setOneTimePrelude("")
      setOneTimePreludeOpen(false)
      onRequestRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
    }
  }

  async function handleCopy(text: string, label = "Copied") {
    try {
      await copyToClipboard(text)
      setNotice(label)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleSetMergeReady(next: boolean) {
    setMergeReadyPending(true)
    setError(null)
    try {
      await setTaskMergeReady(taskId, next)
      setMergeReady(next)
      onRequestRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setMergeReadyPending(false)
    }
  }

  const attachCommand = useMemo(
    () => `rn agent attach --task ${taskId}`,
    [taskId],
  )

  const logsCommand = useMemo(() => `rn agent logs --task ${taskId}`, [taskId])

  const checkoutCommand = useMemo(
    () => `rn checkout --task ${taskId}`,
    [taskId],
  )

  async function refreshLogs() {
    if (!hasAgent) {
      setLogsText(null)
      setLogsPath(null)
      setLogsTruncated(false)
      setLogsError(null)
      return
    }
    setLogsPending(true)
    setLogsError(null)
    try {
      const response = await fetchTaskAgentLogs(taskId, {
        lines: 200,
        maxBytes: 65_536,
      })
      setLogsText(response.text)
      setLogsPath(response.path)
      setLogsTruncated(response.truncated)
    } catch (e) {
      setLogsError(e instanceof Error ? e.message : String(e))
    } finally {
      setLogsPending(false)
    }
  }

  useEffect(() => {
    if (!logsOpen) {
      return
    }
    if (!hasAgent) {
      setLogsText(null)
      setLogsPath(null)
      setLogsTruncated(false)
      setLogsError(null)
      return
    }
    void refreshLogs()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logsOpen, taskId])

  const displayLogsText = useMemo(() => {
    if (!logsText) {
      return null
    }
    return stripAnsi(logsText)
  }, [logsText])

  const floatingPortal =
    isRunning && floatingActionsPortalId
      ? document.getElementById(floatingActionsPortalId)
      : null

  const floatingActions = isRunning ? (
    <FloatingActions>
      <Button
        variant="ghost"
        size="xs"
        className="h-full rounded-none border-0"
        title="Copy attach command"
        onClick={() => void handleCopy(attachCommand, "Attach command copied")}
        disabledReason={pending !== null ? "Action in progress" : null}
      >
        <Terminal />
        Attach
      </Button>
      <Button
        variant="ghost"
        size="xs"
        className="h-full rounded-none border-0 border-l"
        onClick={() => void handleRestart()}
        disabledReason={pending !== null ? "Action in progress" : null}
      >
        <RotateCcw />
        Restart
      </Button>
      <Button
        variant="ghost"
        size="xs"
        className="h-full rounded-none border-0 border-l text-destructive hover:bg-destructive/10"
        onClick={() => void handleStop()}
        disabledReason={pending !== null ? "Action in progress" : null}
      >
        <Square />
        Stop
      </Button>
    </FloatingActions>
  ) : null

  return (
    <div className="grid gap-3">
      {floatingPortal && floatingActions
        ? createPortal(floatingActions, floatingPortal)
        : null}

      <div className="flex items-start justify-between gap-3">
        <div className={"min-w-0 " + (isRunning ? "pr-28" : "")}>
          <div className="flex flex-wrap items-center gap-2">
            <div className="rounded-md bg-foreground/5 px-2 py-1 font-mono text-xs text-foreground/80">
              {agentName}
            </div>
            <AgentStatusBadge status={statusLabel} />
            {harnessKind ? (
              <div className="rounded-md bg-foreground/5 px-2 py-1 font-mono text-xs text-foreground/70">
                {harnessKind}
              </div>
            ) : null}
          </div>
        </div>

        {!isRunning ? (
          <div className="flex shrink-0 items-center gap-2">
            <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
              {canRestart ? (
                <Button
                  variant="outline"
                  size="xs"
                  className="h-full rounded-none border-0"
                  onClick={() => void handleRestart()}
                  disabledReason={
                    pending !== null ? "Action in progress" : null
                  }
                >
                  <RotateCcw />
                  Restart
                </Button>
              ) : (
                <Button
                  variant="outline"
                  size="xs"
                  className="h-full rounded-none border-0"
                  onClick={() => void handleStart()}
                  disabledReason={
                    pending !== null
                      ? "Action in progress"
                      : !configuredHarnessCommand
                        ? "Set a harness command in Configure"
                        : null
                  }
                >
                  Start
                </Button>
              )}
            </div>
          </div>
        ) : null}
      </div>

      <div className="grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">Harness</div>
          <span className="rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] text-foreground/70">
            {(orchestrationDefaults?.harness.detach ?? true)
              ? "Detached"
              : "Foreground"}
          </span>
        </div>
        <div className="rounded-md border bg-background/40 px-2 py-2 font-mono text-xs text-foreground shadow-sm">
          {harnessCommand || "—"}
        </div>
        <div className="text-xs text-muted-foreground">
          {agentArgv
            ? "Command used for the most recent run."
            : "Command used when starting this agent."}
        </div>
      </div>

      <div className="grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">One-time prelude</div>
          <Button
            variant="ghost"
            size="xs"
            onClick={() => setOneTimePreludeOpen((open) => !open)}
            disabledReason={pending !== null ? "Action in progress" : null}
          >
            {oneTimePreludeOpen ? "Hide" : "Set"}
          </Button>
        </div>
        {oneTimePreludeOpen ? (
          <>
            <textarea
              className="min-h-20 resize-y rounded-md border bg-background/40 px-2 py-2 font-mono text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
              value={oneTimePrelude}
              onChange={(e) => setOneTimePrelude(e.target.value)}
              placeholder="Optional. Sent once on the next Start/Restart. Supports placeholders like {task_id}, {task_title}, {epic_slug}."
              disabled={pending !== null}
            />
            <div className="flex items-center justify-between gap-3">
              <div className="text-xs text-muted-foreground">
                Not saved. Cleared after a successful start/restart.
              </div>
              <Button
                variant="ghost"
                size="xs"
                onClick={() => setOneTimePrelude("")}
                disabledReason={pending !== null ? "Action in progress" : null}
              >
                Clear
              </Button>
            </div>
          </>
        ) : null}
      </div>

      <div className="grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <div className="text-xs text-muted-foreground">Merge</div>
          <Switch
            checked={mergeReady}
            onCheckedChange={(checked) => void handleSetMergeReady(checked)}
            disabledReason={
              pending !== null
                ? "Action in progress"
                : mergeReadyPending
                  ? "Saving…"
                  : null
            }
          />
        </div>
        <div className="text-xs text-muted-foreground">
          Mark this task as ready to merge (required for{" "}
          <span className="font-mono">rn merge</span> unless forced).
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          variant="ghost"
          size="xs"
          onClick={() => void handleCopy(logsCommand, "Logs command copied")}
          disabledReason={
            pending !== null
              ? "Action in progress"
              : !hasAgent
                ? "Start the agent to see logs"
                : null
          }
        >
          Copy logs
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={() => setLogsOpen((open) => !open)}
          disabledReason={
            pending !== null
              ? "Action in progress"
              : !hasAgent
                ? "Start the agent to see output"
                : null
          }
        >
          {logsOpen ? "Hide logs" : "Show logs"}
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={() => void handleCopy(checkoutCommand, "Checkout copied")}
          disabledReason={pending !== null ? "Action in progress" : null}
        >
          Copy rn checkout
        </Button>
      </div>

      {logsOpen ? (
        <div className="grid gap-2">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">
              Agent output (tail)
            </div>
            <div className="flex items-center gap-2">
              {logsPath ? (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => void handleCopy(logsPath, "Log path copied")}
                  disabledReason={logsPending ? "Loading…" : null}
                >
                  Copy path
                </Button>
              ) : null}
              <Button
                variant="ghost"
                size="xs"
                onClick={() => void refreshLogs()}
                disabledReason={logsPending ? "Loading…" : null}
              >
                Refresh
              </Button>
            </div>
          </div>
          <div className="relative max-w-full overflow-hidden rounded-md border bg-background/40 px-2 py-2 font-mono text-xs text-foreground shadow-sm">
            <div className="absolute right-2 top-2 z-10">
              <Button
                variant="outline"
                size="xs"
                className="bg-background/70 backdrop-blur"
                onClick={() =>
                  void handleCopy(displayLogsText ?? "", "Output copied")
                }
                disabledReason={
                  logsPending
                    ? "Loading…"
                    : !displayLogsText
                      ? "No output to copy"
                      : null
                }
              >
                Copy
              </Button>
            </div>
            <pre className="max-h-56 min-w-0 overflow-auto whitespace-pre pr-16">
              {!hasAgent
                ? "Start the agent to see output."
                : displayLogsText || (logsPending ? "Loading…" : "No output")}
            </pre>
          </div>
          {logsTruncated ? (
            <div className="text-xs text-muted-foreground">
              Output truncated.
            </div>
          ) : null}
          {logsError ? (
            <pre className="whitespace-pre-wrap text-xs text-destructive">
              {logsError}
            </pre>
          ) : null}
        </div>
      ) : null}

      {error ? (
        <pre className="whitespace-pre-wrap text-xs text-destructive">
          {error}
        </pre>
      ) : null}
      {notice ? (
        <pre className="whitespace-pre-wrap text-xs text-muted-foreground">
          {notice}
        </pre>
      ) : null}
    </div>
  )
}

export function DetailsPanel({
  open,
  task,
  node,
  agent,
  mergeRun,
  onRequestRefresh,
  edge,
}: DetailsPanelProps) {
  const selectionKey = edge
    ? `edge:${edge.id}`
    : node
      ? `node:${node.id}`
      : task
        ? `task:${task.id}`
        : "none"

  const defaultSections = edge
    ? ["contract"]
    : mergeRun &&
        (mergeRun.status === "blocked" ||
          mergeRun.status === "resumable" ||
          mergeRun.status === "running" ||
          mergeRun.status === "failed")
      ? ["agent", "merge-run", "readme"]
      : ["agent", "readme"]
  const floatingActionsPortalId = "details-panel-floating-actions"

  return (
    <SlidePanel open={open}>
      <div className="p-3">
        <div id={floatingActionsPortalId} />
        <Accordion
          key={selectionKey}
          multiple
          defaultValue={defaultSections}
          className="border-0"
        >
          {edge ? (
            <>
              <AccordionItem value="contract">
                <AccordionTrigger>Contract</AccordionTrigger>
                <AccordionContent className="pt-3">
                  <div className="text-muted-foreground">
                    No contract recorded yet.
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="commits">
                <AccordionTrigger>Commits</AccordionTrigger>
                <AccordionContent className="pt-3">
                  <div className="grid gap-3">
                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">Count</div>
                      {edge.commitCount !== null &&
                      edge.commitCount !== undefined ? (
                        <div className="text-sm">{edge.commitCount}</div>
                      ) : (
                        <div className="text-sm text-muted-foreground">
                          Unavailable
                        </div>
                      )}
                    </div>
                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">Range</div>
                      {edge.baseSha && edge.headSha ? (
                        <div className="font-mono text-xs text-foreground/80">
                          {edge.baseSha.slice(0, 7)}..{edge.headSha.slice(0, 7)}
                        </div>
                      ) : (
                        <div className="text-sm text-muted-foreground">
                          Unavailable
                        </div>
                      )}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="messages">
                <AccordionTrigger>Messages</AccordionTrigger>
                <AccordionContent className="pt-3">
                  <div className="text-muted-foreground">No messages.</div>
                </AccordionContent>
              </AccordionItem>
            </>
          ) : (
            <>
              <AccordionItem value="agent">
                <AccordionTrigger>Agent</AccordionTrigger>
                <AccordionContent className="pt-3">
                  {task && onRequestRefresh ? (
                    <AgentActions
                      task={task}
                      agent={agent ?? null}
                      onRequestRefresh={onRequestRefresh}
                      floatingActionsPortalId={floatingActionsPortalId}
                    />
                  ) : null}
                </AccordionContent>
              </AccordionItem>

              {mergeRun ? (
                <AccordionItem value="merge-run">
                  <AccordionTrigger>Merge Run</AccordionTrigger>
                  <AccordionContent className="pt-3">
                    <MergeRunDetails mergeRun={mergeRun} node={node} />
                  </AccordionContent>
                </AccordionItem>
              ) : null}

              <AccordionItem value="readme">
                <AccordionTrigger>README</AccordionTrigger>
                <AccordionContent className="pt-3">
                  {task?.readme ? (
                    <Markdown
                      content={task.readme}
                      omitFirstHeading
                      omitMetadataSection
                    />
                  ) : (
                    <div className="text-muted-foreground">No README.</div>
                  )}
                </AccordionContent>
              </AccordionItem>
            </>
          )}
        </Accordion>
      </div>
    </SlidePanel>
  )
}
