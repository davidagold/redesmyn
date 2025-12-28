import { Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { Button } from "@/components/ui/button"
import { useEpics } from "@/hooks/useEpics"
import { type StreamEvent, useEventStream } from "@/hooks/useEventStream"
import { useGraph } from "@/hooks/useGraph"
import {
  getStoredHarnessCommand,
  storeHarnessCommand,
} from "@/lib/agent-settings"
import { copyToClipboard } from "@/lib/clipboard"
import { formatBranchName, makeEdgeId } from "@/lib/graph-utils"
import type { NodeActivity } from "@/lib/presence"
import { ChevronRight } from "lucide-react"

type FleetMode = "auto" | "fixed"

const RUN_FLEET_MODE_KEY = "rn.run.fleetMode"
const RUN_FLEET_SIZE_KEY = "rn.run.fleetSize"
const RUN_DETACH_KEY = "rn.run.detach"

function getStoredFleetMode(): FleetMode {
  if (typeof window === "undefined") {
    return "auto"
  }
  const raw = window.localStorage.getItem(RUN_FLEET_MODE_KEY)
  return raw === "fixed" ? "fixed" : "auto"
}

function storeFleetMode(mode: FleetMode) {
  if (typeof window === "undefined") {
    return
  }
  window.localStorage.setItem(RUN_FLEET_MODE_KEY, mode)
}

function getStoredFleetSize(): number {
  if (typeof window === "undefined") {
    return 3
  }
  const raw = window.localStorage.getItem(RUN_FLEET_SIZE_KEY)
  if (!raw) {
    return 3
  }
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 3
}

function storeFleetSize(size: number) {
  if (typeof window === "undefined") {
    return
  }
  window.localStorage.setItem(RUN_FLEET_SIZE_KEY, String(size))
}

function getStoredDetach(): boolean {
  if (typeof window === "undefined") {
    return true
  }
  const raw = window.localStorage.getItem(RUN_DETACH_KEY)
  return raw !== "false"
}

function storeDetach(detach: boolean) {
  if (typeof window === "undefined") {
    return
  }
  window.localStorage.setItem(RUN_DETACH_KEY, detach ? "true" : "false")
}

function shellQuote(value: string) {
  if (value === "") {
    return "''"
  }
  if (/^[a-zA-Z0-9_./:@=+-]+$/.test(value)) {
    return value
  }
  return `'${value.split("'").join("'\\\\''")}'`
}

export function EpicView() {
  const navigate = useNavigate()
  const params = useParams({ strict: false })
  const epicSlug = params.epicSlug as string | undefined
  const nodeIdParam = params.nodeId as string | undefined
  const nodeId = nodeIdParam ? parseInt(nodeIdParam, 10) : null
  const fromNodeIdParam = params.fromNodeId as string | undefined
  const toNodeIdParam = params.toNodeId as string | undefined
  const parsedFromNodeId = fromNodeIdParam
    ? parseInt(fromNodeIdParam, 10)
    : null
  const parsedToNodeId = toNodeIdParam ? parseInt(toNodeIdParam, 10) : null
  const fromNodeId =
    parsedFromNodeId !== null && !Number.isNaN(parsedFromNodeId)
      ? parsedFromNodeId
      : null
  const toNodeId =
    parsedToNodeId !== null && !Number.isNaN(parsedToNodeId)
      ? parsedToNodeId
      : null
  const selectedEdgeId =
    fromNodeId !== null && toNodeId !== null
      ? makeEdgeId(fromNodeId, toNodeId)
      : null

  const {
    epics,
    loading: epicsLoading,
    error: epicsError,
    refresh: refreshEpics,
  } = useEpics()
  const [epicMenuOpen, setEpicMenuOpen] = useState(false)
  const [focusMode, setFocusMode] = useState(false)
  const refreshTimerRef = useRef<number | null>(null)
  const [activityByNodeId, setActivityByNodeId] =
    useState<Map<number, NodeActivity>>(new Map())
  const [, setActivityTick] = useState(0)

  const [fleetMode, setFleetMode] = useState<FleetMode>(getStoredFleetMode)
  const [fleetSizeText, setFleetSizeText] = useState(() =>
    String(getStoredFleetSize()),
  )
  const [harnessCommand, setHarnessCommand] = useState<string>(
    getStoredHarnessCommand,
  )
  const [detach, setDetach] = useState(getStoredDetach)
  const [runNotice, setRunNotice] = useState<string | null>(null)

  const selectedEpic = useMemo(
    () => epics.find((e) => e.slug === epicSlug) ?? null,
    [epics, epicSlug],
  )

  const {
    graph,
    error: graphError,
    loading: graphLoading,
    refresh: refreshGraph,
    tasksById,
    agentsById,
    childrenByParent,
    nodesById,
    rootNodes,
  } = useGraph(selectedEpic?.id ?? null)

  const selectedNode = useMemo(
    () => (nodeId !== null ? (nodesById.get(nodeId) ?? null) : null),
    [nodesById, nodeId],
  )

  const selectedTask = useMemo(() => {
    if (!selectedNode || selectedNode.primaryTaskId === null) {
      return null
    }
    return tasksById.get(selectedNode.primaryTaskId) ?? null
  }, [selectedNode, tasksById])

  const selectedEdge = useMemo(() => {
    if (
      fromNodeId === null ||
      toNodeId === null ||
      Number.isNaN(fromNodeId) ||
      Number.isNaN(toNodeId)
    ) {
      return null
    }
    const fromNode = nodesById.get(fromNodeId) ?? null
    const toNode = nodesById.get(toNodeId) ?? null
    if (!fromNode || !toNode) {
      return null
    }

    const fromTask =
      fromNode.primaryTaskId !== null
        ? (tasksById.get(fromNode.primaryTaskId) ?? null)
        : null
    const toTask =
      toNode.primaryTaskId !== null
        ? (tasksById.get(toNode.primaryTaskId) ?? null)
        : null

    const fromLabel =
      fromTask?.title ?? formatBranchName(fromNode.branchName, epicSlug)
    const toLabel =
      toTask?.title ?? formatBranchName(toNode.branchName, epicSlug)

    return {
      id: makeEdgeId(fromNode.id, toNode.id),
      fromNodeId: fromNode.id,
      toNodeId: toNode.id,
      fromLabel,
      toLabel,
    }
  }, [epicSlug, fromNodeId, nodesById, tasksById, toNodeId])

  const selectedLabel = selectedNode
    ? (selectedTask?.title ??
      formatBranchName(selectedNode.branchName, epicSlug))
    : selectedEdge
      ? `${selectedEdge.fromLabel} → ${selectedEdge.toLabel}`
      : null

  const loading = epicsLoading || graphLoading
  const error = epicsError || graphError

  const runSummary = useMemo(() => {
    if (!graph) {
      return null
    }
    const tasks = graph.tasks ?? []

    let eligible = 0
    let running = 0
    let blocked = 0
    let failed = 0

    for (const task of tasks) {
      if (task.nodeId === null) {
        continue
      }
      if (task.state === "blocked") {
        blocked += 1
        continue
      }
      if (task.state === "done") {
        continue
      }

      eligible += 1
      const node = nodesById.get(task.nodeId) ?? null
      const agent =
        node && node.agentId !== null
          ? (agentsById.get(node.agentId) ?? null)
          : null

      if (agent?.status === "running") {
        running += 1
      } else if (agent?.status === "blocked") {
        blocked += 1
      } else if (agent?.status === "error") {
        failed += 1
      }
    }

    return { eligible, running, blocked, failed }
  }, [agentsById, graph, nodesById])

  const parsedFleetSize = useMemo(() => {
    const parsed = Number.parseInt(fleetSizeText, 10)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return null
    }
    return parsed
  }, [fleetSizeText])

  const runCommand = useMemo(() => {
    if (!selectedEpic || !runSummary) {
      return null
    }
    const size =
      fleetMode === "auto" ? runSummary.eligible : (parsedFleetSize ?? null)
    if (size === null) {
      return null
    }
    const trimmed = harnessCommand.trim()
    if (!trimmed) {
      return null
    }
    const parts: string[] = [
      "rn",
      "run",
      "--epic",
      shellQuote(selectedEpic.slug),
      "--fleet-size",
      String(size),
      "--harness",
      shellQuote(trimmed),
    ]
    if (!detach) {
      parts.push("--no-detach")
    }
    return parts.join(" ")
  }, [
    detach,
    fleetMode,
    harnessCommand,
    parsedFleetSize,
    runSummary,
    selectedEpic,
  ])

  useEffect(() => {
    if (!runNotice) {
      return
    }
    const id = window.setTimeout(() => setRunNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [runNotice])

  useEffect(() => {
    const id = window.setInterval(
      () => setActivityTick((tick) => tick + 1),
      1_000,
    )
    return () => window.clearInterval(id)
  }, [])

  const scheduleGraphRefresh = useCallback(() => {
    if (refreshTimerRef.current !== null) {
      return
    }
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null
      void refreshGraph()
    }, 250)
  }, [refreshGraph])

  const handleStreamEvent = useCallback((event: StreamEvent) => {
    const nodeId = event.data.node_id
    if (typeof nodeId === "number") {
      const observedAt = Date.parse(event.createdAt) || Date.now()
      setActivityByNodeId((prev) => {
        const next = new Map(prev)
        const current = next.get(nodeId) ?? {}
        if (event.eventType === "git.commit") {
          next.set(nodeId, { ...current, lastCommitAt: observedAt })
        } else if (event.eventType === "worktree.health") {
          next.set(nodeId, { ...current, lastWorktreeAt: observedAt })
        }
        return next
      })
    }
  }, [])

  useEventStream({
    epic: selectedEpic?.slug ?? null,
    onEvent: handleStreamEvent,
    onResync: scheduleGraphRefresh,
  })

  useEffect(() => {
    return () => {
      if (refreshTimerRef.current !== null) {
        window.clearTimeout(refreshTimerRef.current)
      }
    }
  }, [])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (
        e.target instanceof HTMLElement &&
        (e.target.isContentEditable ||
          e.target.tagName === "INPUT" ||
          e.target.tagName === "TEXTAREA" ||
          e.target.tagName === "SELECT")
      ) {
        return
      }

      if (e.key === "Escape") {
        if (epicMenuOpen) {
          setEpicMenuOpen(false)
          return
        }
        if ((nodeId !== null || selectedEdgeId !== null) && epicSlug) {
          setFocusMode(false)
          void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
        }
        return
      }

      if (e.key === "f" && !e.metaKey && !e.ctrlKey && !e.altKey) {
        if (!selectedNode) {
          return
        }
        setFocusMode((enabled) => !enabled)
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [epicMenuOpen, epicSlug, navigate, nodeId, selectedEdgeId, selectedNode])

  function handleSelectNode(id: number) {
    if (epicSlug) {
      void navigate({
        to: "/graph/$epicSlug/$nodeId",
        params: { epicSlug, nodeId: String(id) },
      })
    }
  }

  function handleSelectEdge(fromId: number, toId: number) {
    setFocusMode(false)
    if (epicSlug) {
      void navigate({
        to: "/graph/$epicSlug/e/$fromNodeId/$toNodeId",
        params: {
          epicSlug,
          fromNodeId: String(fromId),
          toNodeId: String(toId),
        },
      })
    }
  }

  function handleClearSelection() {
    setFocusMode(false)
    if (epicSlug) {
      void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
    }
  }

  function handleSelectEpic(epicId: number) {
    const epic = epics.find((e) => e.id === epicId)
    if (epic) {
      void navigate({ to: "/graph/$epicSlug", params: { epicSlug: epic.slug } })
    }
    setEpicMenuOpen(false)
  }

  async function handleRefresh() {
    await refreshEpics()
    await refreshGraph()
  }

  async function handleCopyRunCommand() {
    if (!runCommand) {
      return
    }
    try {
      await copyToClipboard(runCommand)
      setRunNotice("Command copied")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <ContentPanel>
      <ContentPanelHeader>
        <div className="flex min-w-0 items-center gap-2 text-sm">
          <EpicSelector
            epics={epics}
            selectedEpic={selectedEpic}
            selectedEpicId={selectedEpic?.id ?? null}
            loading={loading}
            menuOpen={epicMenuOpen}
            onMenuToggle={() => setEpicMenuOpen((open) => !open)}
            onMenuClose={() => setEpicMenuOpen(false)}
            onSelectEpic={handleSelectEpic}
          />

          {selectedLabel ? (
            <>
              <ChevronRight
                className="h-4 w-4 shrink-0 text-muted-foreground/60"
                aria-hidden="true"
              />
              <Button
                variant="ghost"
                className="w-fit"
                onClick={handleClearSelection}
              >
                <span className="truncate">{selectedLabel}</span>
              </Button>
            </>
          ) : null}
        </div>

        <Button
          variant="outline"
          onClick={() => void handleRefresh()}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh"}
        </Button>
      </ContentPanelHeader>

      {error ? (
        <div className="border-b px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      {selectedEpic && runSummary ? (
        <div className="border-b px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-baseline gap-3">
              <div className="text-sm font-medium">Run</div>
              <div className="text-xs text-muted-foreground">
                Running {runSummary.running} / Eligible {runSummary.eligible} •
                Blocked {runSummary.blocked} • Failed {runSummary.failed}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => void handleCopyRunCommand()}
                disabled={!runCommand || runSummary.eligible === 0}
                title={
                  runSummary.eligible === 0
                    ? "No eligible tasks"
                    : (runCommand ?? undefined)
                }
              >
                Copy rn run …
              </Button>
              {runNotice ? (
                <div className="text-xs text-muted-foreground">{runNotice}</div>
              ) : null}
            </div>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2">
              <Button
                variant={fleetMode === "auto" ? "secondary" : "ghost"}
                size="xs"
                onClick={() => {
                  setFleetMode("auto")
                  storeFleetMode("auto")
                }}
              >
                Auto-size
              </Button>
              <Button
                variant={fleetMode === "fixed" ? "secondary" : "ghost"}
                size="xs"
                onClick={() => {
                  setFleetMode("fixed")
                  storeFleetMode("fixed")
                }}
              >
                Fixed
              </Button>

              {fleetMode === "fixed" ? (
                <input
                  className="h-7 w-20 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                  type="number"
                  min={1}
                  value={fleetSizeText}
                  onChange={(e) => {
                    setFleetSizeText(e.target.value)
                  }}
                  onBlur={() => {
                    if (parsedFleetSize !== null) {
                      storeFleetSize(parsedFleetSize)
                      setFleetSizeText(String(parsedFleetSize))
                      return
                    }
                    const fallback = getStoredFleetSize()
                    setFleetSizeText(String(fallback))
                  }}
                />
              ) : (
                <div className="text-xs text-muted-foreground">
                  Fleet size = {runSummary.eligible}
                </div>
              )}
            </div>

            <input
              className="h-7 min-w-[180px] flex-1 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
              value={harnessCommand}
              onChange={(e) => {
                const next = e.target.value
                setHarnessCommand(next)
                storeHarnessCommand(next)
              }}
              placeholder="codex"
            />

            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                checked={detach}
                onChange={(e) => {
                  setDetach(e.target.checked)
                  storeDetach(e.target.checked)
                }}
              />
              Detach (tmux)
            </label>
          </div>

          {runCommand ? (
            <div
              className="mt-2 truncate font-mono text-xs text-muted-foreground"
              title={runCommand}
            >
              {runCommand}
            </div>
          ) : (
            <div className="mt-2 text-xs text-muted-foreground">
              Enter a harness command to generate `rn run …`.
            </div>
          )}
        </div>
      ) : null}

      {graph ? (
        <div className="relative flex min-h-0 flex-1">
          <GraphView
            rootNodes={rootNodes}
            childrenByParent={childrenByParent}
            tasksById={tasksById}
            agentsById={agentsById}
            activityByNodeId={activityByNodeId}
            trunk={graph.trunk ?? null}
            selectedNodeId={nodeId}
            selectedEdgeId={selectedEdgeId}
            focusMode={focusMode}
            epicSlug={epicSlug}
            onSelectNode={handleSelectNode}
            onSelectEdge={handleSelectEdge}
            onClearSelection={handleClearSelection}
            onRequestRefresh={scheduleGraphRefresh}
          />

          <DetailsPanel
            open={!!selectedNode || !!selectedEdge}
            node={selectedNode}
            task={selectedTask}
            agent={
              selectedNode?.agentId !== null &&
              selectedNode?.agentId !== undefined
                ? (agentsById.get(selectedNode.agentId) ?? null)
                : null
            }
            onRequestRefresh={scheduleGraphRefresh}
            edge={selectedEdge}
          />
        </div>
      ) : (
        <div className="flex-1 p-6 text-sm text-muted-foreground">
          {loading ? "Loading graph..." : "No graph data available."}
        </div>
      )}

      <Outlet />
    </ContentPanel>
  )
}
