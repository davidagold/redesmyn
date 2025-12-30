import { useEffect, useMemo, useState } from "react"
import { Markdown } from "@/components/markdown"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import { SlidePanel } from "@/components/ui/slide-panel"
import {
  fetchTaskAgentLogs,
  restartTaskAgent,
  startTaskAgent,
  stopTaskAgent,
} from "@/api"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { copyToClipboard } from "@/lib/clipboard"
import { AgentStatusBadge } from "@/components/agents/AgentStatusBadge"
import { RotateCcw, Square, Terminal } from "lucide-react"

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
  const withoutCsi = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
  // OSC sequences (titles, hyperlinks, etc.)
  return withoutCsi.replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, "")
}

interface DetailsPanelProps {
  open: boolean
  task: Task | null
  node?: GraphNode | null
  agent?: Agent | null
  onRequestRefresh?: () => void
  edge?: EdgeSelection | null
}

function AgentActions({
  task,
  agent,
  onRequestRefresh,
}: {
  task: Task
  agent: Agent | null
  onRequestRefresh: () => void
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
  }, [task.id])

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

  async function handleStart() {
    setPending("start")
    setError(null)
    try {
      const response = await startTaskAgent(taskId, {
        harness: configuredHarnessCommand,
        detach: orchestrationDefaults?.harness.detach ?? true,
      })
      const warnings = response.warnings ?? []
      if (warnings.length > 0) {
        if (!response.started || response.agentStatus === "error") {
          setError(warnings.join("\n"))
        } else {
          setNotice(warnings[0])
        }
      }
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
      })
      const warnings = response.warnings ?? []
      if (warnings.length > 0) {
        if (!response.started || response.agentStatus === "error") {
          setError(warnings.join("\n"))
        } else {
          setNotice(warnings[0])
        }
      }
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

  return (
    <div className="grid gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
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

        <div className="flex shrink-0 items-center gap-2">
          {isRunning ? (
            <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
              <Button
                variant="outline"
                size="xs"
                className="h-full rounded-none border-0"
                title="Copy attach command"
                onClick={() =>
                  void handleCopy(attachCommand, "Attach command copied")
                }
                disabledReason={pending !== null ? "Action in progress" : null}
              >
                <Terminal />
                Attach
              </Button>
              <Button
                variant="outline"
                size="xs"
                className="h-full rounded-none border-0 border-l"
                onClick={() => void handleRestart()}
                disabledReason={pending !== null ? "Action in progress" : null}
              >
                <RotateCcw />
                Restart
              </Button>
              <Button
                variant="outline"
                size="xs"
                className="h-full rounded-none border-0 border-l text-destructive hover:bg-destructive/10"
                onClick={() => void handleStop()}
                disabledReason={pending !== null ? "Action in progress" : null}
              >
                <Square />
                Stop
              </Button>
            </div>
          ) : (
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
          )}
        </div>
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

  const defaultSections = edge ? ["contract"] : ["agent", "readme"]

  return (
    <SlidePanel open={open}>
      <div className="p-3">
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
                    />
                  ) : null}
                </AccordionContent>
              </AccordionItem>

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
