import { Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { Button } from "@/components/ui/button"
import { restartTaskAgent, startTaskAgent, stopTaskAgent } from "@/api"
import { useEpics } from "@/hooks/useEpics"
import { type StreamEvent, useEventStream } from "@/hooks/useEventStream"
import { useGraph } from "@/hooks/useGraph"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { getStoredHarnessCommand } from "@/lib/agent-settings"
import { formatBranchName, makeEdgeId } from "@/lib/graph-utils"
import type { NodeActivity } from "@/lib/presence"
import { ChevronRight, Play, Square } from "lucide-react"

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
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<number>>(new Set())
  const selectedNodeIdsRef = useRef(selectedNodeIds)
  const [runAction, setRunAction] =
    useState<"runAll" | "runSelected" | "stopAll" | "stopSelected" | null>(null)
  const refreshTimerRef = useRef<number | null>(null)
  const [activityByNodeId, setActivityByNodeId] =
    useState<Map<number, NodeActivity>>(new Map())
  const [, setActivityTick] = useState(0)

  const [runNotice, setRunNotice] = useState<string | null>(null)
  const {
    defaults: orchestrationDefaults,
    loading: orchestrationDefaultsLoading,
    refresh: refreshOrchestrationDefaults,
  } = useOrchestrationDefaults()

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

  const actionTargets = useMemo(() => {
    const empty = {
      start: [] as number[],
      restart: [] as number[],
      stop: [] as number[],
    }
    const targets = { all: { ...empty }, selected: { ...empty } }
    if (!graph) {
      return targets
    }

    const allStart = new Set<number>()
    const allRestart = new Set<number>()
    const allStop = new Set<number>()

    for (const task of graph.tasks ?? []) {
      if (task.nodeId === null) {
        continue
      }
      if (task.state === "blocked" || task.state === "done") {
        continue
      }

      const node = nodesById.get(task.nodeId) ?? null
      const agent =
        node && node.agentId !== null
          ? (agentsById.get(node.agentId) ?? null)
          : null
      const status = agent?.status ?? null

      if (!agent || status === "stopped") {
        allStart.add(task.id)
      } else if (status === "error") {
        allRestart.add(task.id)
      }

      if (status === "running" || status === "blocked") {
        allStop.add(task.id)
      }
    }

    const selectedStart = new Set<number>()
    const selectedRestart = new Set<number>()
    const selectedStop = new Set<number>()

    for (const selectedNodeId of selectedNodeIds) {
      const node = nodesById.get(selectedNodeId) ?? null
      if (!node || node.primaryTaskId === null) {
        continue
      }
      const task = tasksById.get(node.primaryTaskId) ?? null
      if (!task || task.state === "blocked" || task.state === "done") {
        continue
      }
      const agent =
        node.agentId !== null ? (agentsById.get(node.agentId) ?? null) : null
      const status = agent?.status ?? null

      if (!agent || status === "stopped") {
        selectedStart.add(task.id)
      } else if (status === "error") {
        selectedRestart.add(task.id)
      }

      if (status === "running" || status === "blocked") {
        selectedStop.add(task.id)
      }
    }

    targets.all.start = [...allStart].sort((a, b) => a - b)
    targets.all.restart = [...allRestart].sort((a, b) => a - b)
    targets.all.stop = [...allStop].sort((a, b) => a - b)
    targets.selected.start = [...selectedStart].sort((a, b) => a - b)
    targets.selected.restart = [...selectedRestart].sort((a, b) => a - b)
    targets.selected.stop = [...selectedStop].sort((a, b) => a - b)

    return targets
  }, [agentsById, graph, nodesById, selectedNodeIds, tasksById])

  const runCommand = useMemo(() => {
    if (!selectedEpic || !orchestrationDefaults) {
      return null
    }
    if (!orchestrationDefaults.harness.command) {
      return null
    }
    if (
      orchestrationDefaults.fleet.mode === "fixed" &&
      orchestrationDefaults.fleet.size === null
    ) {
      return null
    }
    return `rn run --epic ${shellQuote(selectedEpic.slug)}`
  }, [orchestrationDefaults, selectedEpic])

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
    selectedNodeIdsRef.current = selectedNodeIds
  }, [selectedNodeIds])

  useEffect(() => {
    if (selectedEdgeId !== null) {
      setSelectedNodeIds(new Set())
      return
    }
    if (nodeId === null) {
      setSelectedNodeIds(new Set())
      return
    }
    setSelectedNodeIds((current) => {
      if (current.size === 0) {
        return new Set([nodeId])
      }
      if (current.has(nodeId)) {
        return current
      }
      const next = new Set(current)
      next.add(nodeId)
      return next
    })
  }, [nodeId, selectedEdgeId])

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

  function handleSelectNode(id: number, options: { additive: boolean }) {
    if (!epicSlug) {
      return
    }

    if (!options.additive) {
      setSelectedNodeIds(new Set([id]))
      void navigate({
        to: "/graph/$epicSlug/$nodeId",
        params: { epicSlug, nodeId: String(id) },
      })
      return
    }

    const current = selectedNodeIdsRef.current
    const next = new Set(current)
    const wasSelected = next.has(id)

    if (wasSelected) {
      next.delete(id)
    } else {
      next.add(id)
    }

    setSelectedNodeIds(next)

    if (!wasSelected) {
      void navigate({
        to: "/graph/$epicSlug/$nodeId",
        params: { epicSlug, nodeId: String(id) },
      })
      return
    }

    if (nodeId !== id) {
      return
    }

    const nextPrimary = next.values().next().value ?? null
    if (nextPrimary === null) {
      void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
      return
    }

    void navigate({
      to: "/graph/$epicSlug/$nodeId",
      params: { epicSlug, nodeId: String(nextPrimary) },
    })
  }

  function handleSelectEdge(fromId: number, toId: number) {
    setFocusMode(false)
    setSelectedNodeIds(new Set())
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
    setSelectedNodeIds(new Set())
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
    await refreshOrchestrationDefaults()
    await refreshGraph()
  }

  const canRunAll =
    runAction === null &&
    (actionTargets.all.start.length > 0 || actionTargets.all.restart.length > 0)
  const canStopAll = runAction === null && actionTargets.all.stop.length > 0
  const showSelectedActions = selectedNodeIds.size > 1
  const canRunSelected =
    runAction === null &&
    showSelectedActions &&
    (actionTargets.selected.start.length > 0 ||
      actionTargets.selected.restart.length > 0)
  const canStopSelected =
    runAction === null &&
    showSelectedActions &&
    actionTargets.selected.stop.length > 0

  async function runTargets(targets: {
    start: number[]
    restart: number[]
    stop: number[]
  }) {
    const harness = getStoredHarnessCommand()
    for (const taskId of targets.start) {
      await startTaskAgent(taskId, { harness, detach: true })
    }
    for (const taskId of targets.restart) {
      await restartTaskAgent(taskId, { detach: true })
    }
  }

  async function stopTargets(targets: { stop: number[] }) {
    for (const taskId of targets.stop) {
      await stopTaskAgent(taskId)
    }
  }

  async function handleRunAll() {
    if (!canRunAll) {
      return
    }
    setRunAction("runAll")
    try {
      await runTargets(actionTargets.all)
      scheduleGraphRefresh()
      setRunNotice("Started")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
    } finally {
      setRunAction(null)
    }
  }

  async function handleStopAll() {
    if (!canStopAll) {
      return
    }
    setRunAction("stopAll")
    try {
      await stopTargets(actionTargets.all)
      scheduleGraphRefresh()
      setRunNotice("Stopped")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
    } finally {
      setRunAction(null)
    }
  }

  async function handleRunSelected() {
    if (!canRunSelected) {
      return
    }
    setRunAction("runSelected")
    try {
      await runTargets(actionTargets.selected)
      scheduleGraphRefresh()
      setRunNotice("Started selection")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
    } finally {
      setRunAction(null)
    }
  }

  async function handleStopSelected() {
    if (!canStopSelected) {
      return
    }
    setRunAction("stopSelected")
    try {
      await stopTargets(actionTargets.selected)
      scheduleGraphRefresh()
      setRunNotice("Stopped selection")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
    } finally {
      setRunAction(null)
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
        <div className="border-b px-4 py-2">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm font-medium">Run</div>
            <div className="text-xs text-muted-foreground">
              Running {runSummary.running} / Eligible {runSummary.eligible} •
              Blocked {runSummary.blocked} • Failed {runSummary.failed}
            </div>
            <div className="flex items-center gap-2">
              <div className="flex overflow-hidden rounded-md border border-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-none border-0"
                  disabled={!canRunAll}
                  title={
                    canRunAll
                      ? "Start or restart all eligible tasks"
                      : runAction !== null
                        ? "Action in progress"
                        : "Nothing to start"
                  }
                  onClick={() => void handleRunAll()}
                >
                  <Play />
                  Run all
                </Button>
                <div
                  className={
                    "overflow-hidden transition-all duration-200 will-change-transform " +
                    (showSelectedActions
                      ? "max-w-[10rem] translate-x-0 opacity-100"
                      : "max-w-0 translate-x-2 opacity-0")
                  }
                >
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-none border-0 border-l"
                    disabled={!canRunSelected}
                    title={
                      canRunSelected
                        ? "Start or restart selected tasks"
                        : runAction !== null
                          ? "Action in progress"
                          : "Nothing to start in selection"
                    }
                    onClick={() => void handleRunSelected()}
                  >
                    Run selected
                  </Button>
                </div>
              </div>

              <div className="flex overflow-hidden rounded-md border border-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-none border-0"
                  disabled={!canStopAll}
                  title={
                    canStopAll
                      ? "Stop all running tasks"
                      : runAction !== null
                        ? "Action in progress"
                        : "Nothing to stop"
                  }
                  onClick={() => void handleStopAll()}
                >
                  <Square />
                  Stop all
                </Button>
                <div
                  className={
                    "overflow-hidden transition-all duration-200 will-change-transform " +
                    (showSelectedActions
                      ? "max-w-[10rem] translate-x-0 opacity-100"
                      : "max-w-0 translate-x-2 opacity-0")
                  }
                >
                  <Button
                    variant="outline"
                    size="sm"
                    className="rounded-none border-0 border-l"
                    disabled={!canStopSelected}
                    title={
                      canStopSelected
                        ? "Stop selected running tasks"
                        : runAction !== null
                          ? "Action in progress"
                          : "Nothing to stop in selection"
                    }
                    onClick={() => void handleStopSelected()}
                  >
                    Stop selected
                  </Button>
                </div>
              </div>
            </div>
            {/* TODO: Reintroduce after refining Run UX. (See CopyRunCommandButton.) */}
            {runNotice ? (
              <div className="text-xs text-muted-foreground">{runNotice}</div>
            ) : null}
          </div>

          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <div>
              Fleet:{" "}
              {orchestrationDefaults
                ? orchestrationDefaults.fleet.mode === "auto"
                  ? "auto"
                  : orchestrationDefaults.fleet.size !== null
                    ? `fixed (${orchestrationDefaults.fleet.size})`
                    : "fixed (size not set)"
                : orchestrationDefaultsLoading
                  ? "loading…"
                  : "—"}
            </div>
            <div className="flex items-center gap-1">
              <span>Harness:</span>
              {orchestrationDefaults?.harness.command ? (
                <span className="max-w-[32rem] truncate font-mono">
                  {orchestrationDefaults.harness.command}
                </span>
              ) : orchestrationDefaultsLoading ? (
                "loading…"
              ) : (
                "—"
              )}
            </div>
            <div>
              Detach:{" "}
              {orchestrationDefaults
                ? orchestrationDefaults.harness.detach
                  ? "on"
                  : "off"
                : orchestrationDefaultsLoading
                  ? "loading…"
                  : "—"}
            </div>
            <div>Default epic: {orchestrationDefaults?.defaultEpic ?? "—"}</div>
          </div>

          {runCommand ? (
            <div
              className="mt-1.5 truncate font-mono text-xs text-muted-foreground"
              title={runCommand}
            >
              {runCommand}
            </div>
          ) : null}
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
            selectedNodeIds={selectedNodeIds}
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
