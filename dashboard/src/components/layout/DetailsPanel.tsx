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
import { restartTaskAgent, startTaskAgent, stopTaskAgent } from "@/api"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import {
  getStoredHarnessCommand,
  storeHarnessCommand,
} from "@/lib/agent-settings"
import { copyToClipboard } from "@/lib/clipboard"
import { cn } from "@/lib/utils"
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
  const [command, setCommand] = useState(getStoredHarnessCommand)
  const [commandTouched, setCommandTouched] = useState(false)

  useEffect(() => {
    setPending(null)
    setError(null)
    setNotice(null)
  }, [task.id])

  useEffect(() => {
    if (commandTouched) {
      return
    }
    const next = orchestrationDefaults?.harness.command
    if (next) {
      setCommand(next)
    }
  }, [commandTouched, orchestrationDefaults?.harness.command])

  useEffect(() => {
    if (!notice) {
      return
    }
    const id = window.setTimeout(() => setNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [notice])

  const taskId = task.id
  const agentName = useMemo(() => `a-${taskId}`, [taskId])
  const statusLabel = agent?.status ?? "not started"
  const harnessKind = agent?.harnessProfileId?.split("/")[0] ?? null
  const isRunning = statusLabel === "running" || statusLabel === "blocked"
  const canRestart =
    statusLabel === "running" ||
    statusLabel === "blocked" ||
    statusLabel === "error"

  async function handleStart() {
    setPending("start")
    setError(null)
    try {
      await startTaskAgent(taskId, {
        harness: command,
        detach: orchestrationDefaults?.harness.detach ?? true,
      })
      storeHarnessCommand(command)
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
      await restartTaskAgent(taskId, {
        harness: command.trim() || null,
        detach: orchestrationDefaults?.harness.detach ?? true,
      })
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

  const statusTone = useMemo(() => {
    switch (statusLabel) {
      case "running":
        return {
          label: "running",
          dotClassName: "bg-emerald-400",
          pillClassName: "text-foreground/80",
        }
      case "blocked":
        return {
          label: "blocked",
          dotClassName: "bg-amber-400",
          pillClassName: "text-foreground/80",
        }
      case "error":
        return {
          label: "error",
          dotClassName: "bg-destructive",
          pillClassName: "text-destructive",
        }
      case "stopped":
        return {
          label: "stopped",
          dotClassName: "border border-sky-400",
          pillClassName: "text-foreground/80",
        }
      default:
        return {
          label: statusLabel,
          dotClassName: "border border-border/60",
          pillClassName: "text-muted-foreground",
        }
    }
  }, [statusLabel])

  return (
    <div className="grid gap-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="rounded-md bg-foreground/5 px-2 py-1 font-mono text-xs text-foreground/80">
              {agentName}
            </div>
            <div
              className={cn(
                "inline-flex items-center gap-1 rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] uppercase tracking-wide",
                statusTone.pillClassName,
              )}
            >
              <span
                className={cn("h-2 w-2 rounded-full", statusTone.dotClassName)}
                aria-hidden="true"
              />
              <span>{statusTone.label}</span>
            </div>
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
                      : !command.trim()
                        ? "Enter a harness command"
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

      {!isRunning ? (
        <div className="grid gap-1">
          <div className="flex items-center justify-between gap-3">
            <div className="text-xs text-muted-foreground">Harness</div>
            <span className="rounded-md bg-foreground/5 px-2 py-1 text-[0.625rem] text-foreground/70">
              {(orchestrationDefaults?.harness.detach ?? true)
                ? "Detached"
                : "Foreground"}
            </span>
          </div>
          <input
            className="h-7 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
            value={command}
            onChange={(e) => {
              const next = e.target.value
              setCommand(next)
              setCommandTouched(true)
              storeHarnessCommand(next)
            }}
            placeholder="codex"
            disabled={pending !== null}
          />
        </div>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <Button
          variant="ghost"
          size="xs"
          onClick={() => void handleCopy(logsCommand, "Logs command copied")}
          disabledReason={pending !== null ? "Action in progress" : null}
        >
          Copy logs
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

      {error ? <div className="text-xs text-destructive">{error}</div> : null}
      {notice ? (
        <div className="text-xs text-muted-foreground">{notice}</div>
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
          multiple={!!edge}
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
