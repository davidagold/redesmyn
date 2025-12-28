import { useEffect, useMemo, useState } from "react"
import { Markdown } from "@/components/markdown"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { SlidePanel } from "@/components/ui/slide-panel"
import {
  restartNodeSession,
  setNodeAgent,
  startNodeSession,
  stopNodeSession,
} from "@/api"
import type { Agent, AgentSession, GraphNode, Task } from "@/lib/graph-utils"

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
  node: GraphNode | null
  task: Task | null
  agent: Agent | null
  agents: Agent[]
  session: AgentSession | null
  onRequestRefresh: () => void
  edge?: EdgeSelection | null
}

async function copyToClipboard(text: string) {
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    await navigator.clipboard.writeText(text)
    return
  }

  if (typeof window !== "undefined") {
    window.prompt("Copy to clipboard:", text)
  }
}

function getStoredAgentCommand() {
  if (typeof window === "undefined") {
    return "codex"
  }
  return window.localStorage.getItem("rn.agentCommand") ?? "codex"
}

function storeAgentCommand(value: string) {
  if (typeof window === "undefined") {
    return
  }
  window.localStorage.setItem("rn.agentCommand", value)
}

function NodeActions({
  node,
  agent,
  agents,
  session,
  onRequestRefresh,
}: {
  node: GraphNode
  agent: Agent | null
  agents: Agent[]
  session: AgentSession | null
  onRequestRefresh: () => void
}) {
  const [agentMenuOpen, setAgentMenuOpen] = useState(false)
  const [pending, setPending] =
    useState<"assign" | "start" | "stop" | "restart" | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [command, setCommand] = useState(getStoredAgentCommand)

  useEffect(() => {
    setAgentMenuOpen(false)
    setPending(null)
    setError(null)
    setNotice(null)
  }, [node.id])

  useEffect(() => {
    if (!notice) {
      return
    }
    const id = window.setTimeout(() => setNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [notice])

  const attachCommand = useMemo(() => {
    if (!session) {
      return null
    }
    if (session.attach.type === "tmux") {
      return `rn agent attach --session ${session.id}`
    }
    return `rn agent logs --session ${session.id}`
  }, [session])

  const checkoutCommand = useMemo(
    () => `rn checkout --node ${node.id}`,
    [node.id],
  )
  const worktreePath = node.worktreePath

  async function handleAssign(nextAgentId: number | null) {
    setPending("assign")
    setError(null)
    try {
      await setNodeAgent(node.id, nextAgentId)
      onRequestRefresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setPending(null)
      setAgentMenuOpen(false)
    }
  }

  async function handleStart() {
    setPending("start")
    setError(null)
    try {
      await startNodeSession(node.id, { command, detach: true })
      storeAgentCommand(command)
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
      await stopNodeSession(node.id)
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
      await restartNodeSession(node.id, { detach: true })
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

  return (
    <Card size="sm" className="bg-card/40">
      <CardContent className="grid gap-4">
        <div className="grid gap-1">
          <div className="text-xs text-muted-foreground">Assigned agent</div>
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0 truncate text-sm">
              {agent ? agent.displayName : "Unassigned"}
            </div>
            <div className="relative flex shrink-0 items-center gap-2">
              <Button
                variant="outline"
                size="xs"
                onClick={() => setAgentMenuOpen((open) => !open)}
                aria-expanded={agentMenuOpen}
                disabled={pending !== null}
              >
                {agent ? "Reassign" : "Assign"}
              </Button>
              {agent ? (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => void handleAssign(null)}
                  disabled={pending !== null}
                >
                  Unassign
                </Button>
              ) : null}

              {agentMenuOpen ? (
                <>
                  <button
                    type="button"
                    aria-label="Close agent menu"
                    className="fixed inset-0 z-10 cursor-default bg-transparent"
                    onClick={() => setAgentMenuOpen(false)}
                  />
                  <div className="absolute right-0 top-full z-20 mt-2 w-56 rounded-lg border bg-popover p-2 shadow">
                    <div className="grid gap-1">
                      {agents.length ? (
                        agents.map((a) => (
                          <Button
                            key={a.id}
                            variant={
                              a.id === node.agentId ? "secondary" : "ghost"
                            }
                            className="w-full justify-start"
                            onClick={() => void handleAssign(a.id)}
                            disabled={pending !== null}
                          >
                            {a.displayName}
                          </Button>
                        ))
                      ) : (
                        <div className="px-2 py-1 text-xs text-muted-foreground">
                          No agents registered. Use{" "}
                          <span className="font-mono">rn agent register</span>.
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>

        <div className="grid gap-2">
          <div className="text-xs text-muted-foreground">Session</div>
          {session ? (
            <div className="grid gap-2">
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0 truncate text-sm">
                  {session.status} ({session.attach.type})
                </div>
                <div className="flex shrink-0 flex-wrap gap-2">
                  {attachCommand ? (
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={() =>
                        void handleCopy(attachCommand, "Command copied")
                      }
                      disabled={pending !== null}
                    >
                      {session.attach.type === "tmux"
                        ? "Copy attach"
                        : "Copy logs"}
                    </Button>
                  ) : null}
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
                </div>
              </div>
              {session.cwdPath ? (
                <div className="font-mono text-xs text-foreground/70">
                  cwd: {session.cwdPath}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="grid gap-2">
              <div className="flex flex-col gap-2">
                <input
                  className="h-7 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                  placeholder="codex"
                  disabled={pending !== null}
                />
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs text-muted-foreground">
                    Starts detached (tmux when available).
                  </div>
                  <Button
                    size="xs"
                    onClick={() => void handleStart()}
                    disabled={pending !== null || !agent}
                  >
                    Start
                  </Button>
                </div>
              </div>
              {!agent ? (
                <div className="text-xs text-muted-foreground">
                  Assign an agent before starting a session.
                </div>
              ) : null}
            </div>
          )}
        </div>

        <div className="grid gap-2">
          <div className="text-xs text-muted-foreground">Worktree</div>
          {node.worktreePath ? (
            <div className="font-mono text-xs text-foreground/70">
              {node.worktreePath}
            </div>
          ) : (
            <div className="text-xs text-muted-foreground">
              Not created yet (will be created on session start).
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="ghost"
              size="xs"
              onClick={() => void handleCopy(checkoutCommand, "Command copied")}
              disabled={pending !== null}
            >
              Copy rn checkout
            </Button>
            {worktreePath ? (
              <Button
                variant="ghost"
                size="xs"
                onClick={() => void handleCopy(worktreePath, "Path copied")}
                disabled={pending !== null}
              >
                Copy path
              </Button>
            ) : null}
          </div>
        </div>

        {error ? <div className="text-xs text-destructive">{error}</div> : null}
        {notice ? (
          <div className="text-xs text-muted-foreground">{notice}</div>
        ) : null}
      </CardContent>
    </Card>
  )
}

export function DetailsPanel({
  open,
  node,
  task,
  agent,
  agents,
  session,
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
                  {node ? (
                    <NodeActions
                      node={node}
                      agent={agent}
                      agents={agents}
                      session={session}
                      onRequestRefresh={onRequestRefresh}
                    />
                  ) : (
                    <div className="text-muted-foreground">
                      No node selected.
                    </div>
                  )}
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
