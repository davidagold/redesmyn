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

  return (
    <div className="grid gap-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="truncate text-sm font-medium">Agent {agentName}</div>
          <div className="text-xs text-muted-foreground">
            {statusLabel}
            {harnessKind ? ` • ${harnessKind}` : ""}
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          {isRunning ? (
            <>
              <Button
                variant="outline"
                size="xs"
                onClick={() => void handleCopy(attachCommand, "Command copied")}
                disabled={pending !== null}
              >
                Copy attach
              </Button>
              <Button
                variant="outline"
                size="xs"
                onClick={() => void handleRestart()}
                disabled={pending !== null}
              >
                Restart
              </Button>
              <Button
                variant="destructive"
                size="xs"
                onClick={() => void handleStop()}
                disabled={pending !== null}
              >
                Stop
              </Button>
            </>
          ) : (
            <>
              {canRestart ? (
                <Button
                  variant="outline"
                  size="xs"
                  onClick={() => void handleRestart()}
                  disabled={pending !== null}
                >
                  Restart
                </Button>
              ) : (
                <Button
                  size="xs"
                  onClick={() => void handleStart()}
                  disabled={pending !== null || !command.trim()}
                >
                  Start
                </Button>
              )}
            </>
          )}
        </div>
      </div>

      {!isRunning ? (
        <div className="grid gap-2">
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
          <div className="text-xs text-muted-foreground">
            Starts{" "}
            {(orchestrationDefaults?.harness.detach ?? true)
              ? "detached (tmux when available)"
              : "in the foreground"}
            .
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap gap-2">
        <Button
          variant="ghost"
          size="xs"
          onClick={() => void handleCopy(logsCommand, "Command copied")}
          disabled={pending !== null}
        >
          Copy logs
        </Button>
        <Button
          variant="ghost"
          size="xs"
          onClick={() => void handleCopy(checkoutCommand, "Command copied")}
          disabled={pending !== null}
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
