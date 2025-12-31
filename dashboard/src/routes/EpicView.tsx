import { Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { Button } from "@/components/ui/button"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import {
  runTaskAgentsBulk,
  stopTaskAgent,
  updateOrchestrationDefaults,
} from "@/api"
import { SlidePanel } from "@/components/ui/slide-panel"
import { useEpics } from "@/hooks/useEpics"
import { type StreamEvent, useEventStream } from "@/hooks/useEventStream"
import { useGraph } from "@/hooks/useGraph"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { formatBranchName, makeEdgeId } from "@/lib/graph-utils"
import type { NodeActivity } from "@/lib/presence"
import { ChevronRight, Loader2, Play, Settings2, Square } from "lucide-react"

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
  const [bulkRun, setBulkRun] = useState<{
    runId: string
    total: number
    completed: number
    failed: number
  } | null>(null)
  const refreshTimerRef = useRef<number | null>(null)
  const [activityByNodeId, setActivityByNodeId] =
    useState<Map<number, NodeActivity>>(new Map())
  const [, setActivityTick] = useState(0)

  const [runNotice, setRunNotice] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [configPending, setConfigPending] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)
  const [configNotice, setConfigNotice] = useState<string | null>(null)
  const [configHarness, setConfigHarness] = useState("")
  const [configDetach, setConfigDetach] = useState(true)
  const [configSandboxType, setConfigSandboxType] =
    useState<"none" | "worktree">("none")
  const [configSandboxNetwork, setConfigSandboxNetwork] =
    useState<"allow" | "deny">("allow")
  const [configPrelude, setConfigPrelude] = useState("")
  const [configSendPrelude, setConfigSendPrelude] = useState(true)
  const [configSubmitPrelude, setConfigSubmitPrelude] = useState(true)
  const [defaultPreludeOpen, setDefaultPreludeOpen] = useState(false)
  const {
    defaults: orchestrationDefaults,
    loading: orchestrationDefaultsLoading,
    refresh: refreshOrchestrationDefaults,
  } = useOrchestrationDefaults()

  const closeConfig = useCallback(() => {
    setConfigOpen(false)
    setDefaultPreludeOpen(false)
    setConfigError(null)
  }, [])

  useEffect(() => {
    if (!configOpen) {
      return
    }
    setConfigError(null)
    setConfigNotice(null)
    setDefaultPreludeOpen(false)
    setConfigHarness(orchestrationDefaults?.harness.command ?? "")
    setConfigDetach(orchestrationDefaults?.harness.detach ?? true)
    setConfigSandboxType(orchestrationDefaults?.sandbox.type ?? "none")
    setConfigSandboxNetwork(orchestrationDefaults?.sandbox.network ?? "allow")
    setConfigPrelude(orchestrationDefaults?.harness.prelude ?? "")
    setConfigSendPrelude(orchestrationDefaults?.harness.sendPrelude ?? true)
    setConfigSubmitPrelude(orchestrationDefaults?.harness.submitPrelude ?? true)
  }, [configOpen, orchestrationDefaults])

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

  const runBuckets = useMemo(() => {
    if (!graph) {
      return null
    }

    const eligible = new Set<number>()
    const running = new Set<number>()
    const blocked = new Set<number>()
    const failed = new Set<number>()

    for (const task of graph.tasks ?? []) {
      if (task.nodeId === null) {
        continue
      }
      if (task.state === "done") {
        continue
      }
      if (task.state === "blocked") {
        blocked.add(task.nodeId)
        continue
      }

      eligible.add(task.nodeId)

      const node = nodesById.get(task.nodeId) ?? null
      const agent =
        node && node.agentId !== null
          ? (agentsById.get(node.agentId) ?? null)
          : null

      if (agent?.status === "running") {
        running.add(task.nodeId)
      } else if (agent?.status === "blocked") {
        blocked.add(task.nodeId)
      } else if (agent?.status === "error") {
        failed.add(task.nodeId)
      }
    }

    return {
      eligible: Array.from(eligible),
      running: Array.from(running),
      blocked: Array.from(blocked),
      failed: Array.from(failed),
    }
  }, [agentsById, graph, nodesById])

  const selectionEquals = useCallback(
    (nodeIds: number[]) => {
      if (selectedNodeIds.size !== nodeIds.length) {
        return false
      }
      for (const nodeId of nodeIds) {
        if (!selectedNodeIds.has(nodeId)) {
          return false
        }
      }
      return true
    },
    [selectedNodeIds],
  )

  const selectBucketNodes = useCallback(
    (nodeIds: number[]) => {
      if (!epicSlug) {
        return
      }
      if (selectionEquals(nodeIds)) {
        setFocusMode(false)
        setSelectedNodeIds(new Set())
        void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
        return
      }
      setFocusMode(nodeIds.length > 1)
      setSelectedNodeIds(new Set(nodeIds))
      void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
    },
    [epicSlug, navigate, selectionEquals],
  )

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

  const handleStreamEvent = useCallback(
    (event: StreamEvent) => {
      if (!("nodeId" in event.data) || typeof event.data.nodeId !== "number") {
        return
      }
      const nodeId = event.data.nodeId

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

      if (event.data.type !== "task.agent_run") {
        return
      }

      const data = event.data

      if (data.phase !== "started" && data.phase !== "failed") {
        return
      }

      setBulkRun((current) => {
        if (!current || current.runId !== data.runId) {
          return current
        }

        const completed = current.completed + 1
        const failed = current.failed + (data.phase === "failed" ? 1 : 0)

        if (completed >= current.total) {
          setRunNotice(
            failed > 0
              ? `Started (with ${failed} failure${failed === 1 ? "" : "s"})`
              : "Started",
          )
          return null
        }

        return { ...current, completed, failed }
      })
      scheduleGraphRefresh()
    },
    [scheduleGraphRefresh],
  )

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
      if (e.key === "Escape") {
        if (configOpen) {
          closeConfig()
          return
        }
        if (epicMenuOpen) {
          setEpicMenuOpen(false)
          return
        }
        if (selectedNodeIdsRef.current.size > 0 && epicSlug) {
          setFocusMode(false)
          setSelectedNodeIds(new Set())
          void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
          return
        }
        if ((nodeId !== null || selectedEdgeId !== null) && epicSlug) {
          setFocusMode(false)
          void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
        }
        return
      }

      if (
        e.target instanceof HTMLElement &&
        (e.target.isContentEditable ||
          e.target.tagName === "INPUT" ||
          e.target.tagName === "TEXTAREA" ||
          e.target.tagName === "SELECT")
      ) {
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
  }, [
    closeConfig,
    configOpen,
    epicMenuOpen,
    epicSlug,
    navigate,
    nodeId,
    selectedEdgeId,
    selectedNode,
  ])

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

  const resetConfigFields = useCallback(() => {
    setConfigError(null)
    setConfigNotice(null)
    setDefaultPreludeOpen(false)
    setConfigHarness(orchestrationDefaults?.harness.command ?? "")
    setConfigDetach(orchestrationDefaults?.harness.detach ?? true)
    setConfigSandboxType(orchestrationDefaults?.sandbox.type ?? "none")
    setConfigSandboxNetwork(orchestrationDefaults?.sandbox.network ?? "allow")
    setConfigPrelude(orchestrationDefaults?.harness.prelude ?? "")
    setConfigSendPrelude(orchestrationDefaults?.harness.sendPrelude ?? true)
    setConfigSubmitPrelude(orchestrationDefaults?.harness.submitPrelude ?? true)
  }, [orchestrationDefaults])

  const configDirty = useMemo(() => {
    const currentCommand = orchestrationDefaults?.harness.command ?? ""
    const currentDetach = orchestrationDefaults?.harness.detach ?? true
    const currentSandboxType = orchestrationDefaults?.sandbox.type ?? "none"
    const currentSandboxNetwork =
      orchestrationDefaults?.sandbox.network ?? "allow"
    const currentPrelude = orchestrationDefaults?.harness.prelude ?? ""
    const currentSendPrelude =
      orchestrationDefaults?.harness.sendPrelude ?? true
    const currentSubmitPrelude =
      orchestrationDefaults?.harness.submitPrelude ?? true
    return (
      configHarness !== currentCommand ||
      configDetach !== currentDetach ||
      configSandboxType !== currentSandboxType ||
      configSandboxNetwork !== currentSandboxNetwork ||
      configPrelude !== currentPrelude ||
      configSendPrelude !== currentSendPrelude ||
      configSubmitPrelude !== currentSubmitPrelude
    )
  }, [
    configDetach,
    configHarness,
    configSandboxNetwork,
    configSandboxType,
    configPrelude,
    configSendPrelude,
    configSubmitPrelude,
    orchestrationDefaults,
  ])

  const configuredHarnessCommand =
    orchestrationDefaults?.harness.command?.trim() ?? ""
  const configuredDetach = orchestrationDefaults?.harness.detach ?? true

  async function handleSaveConfig() {
    if (configPending || !configDirty) {
      return
    }
    setConfigPending(true)
    setConfigError(null)
    setConfigNotice(null)
    try {
      await updateOrchestrationDefaults({
        harness: {
          command: configHarness.trim() ? configHarness.trim() : null,
          detach: configDetach,
          prelude: configPrelude.trim() ? configPrelude : null,
          sendPrelude: configSendPrelude,
          submitPrelude: configSubmitPrelude,
        },
        sandbox: {
          type: configSandboxType,
          network: configSandboxNetwork,
        },
      })
      await refreshOrchestrationDefaults()
      setConfigNotice("Saved")
    } catch (e) {
      setConfigError(e instanceof Error ? e.message : String(e))
    } finally {
      setConfigPending(false)
    }
  }

  const needsHarnessForAll = actionTargets.all.start.length > 0
  const needsHarnessForSelected =
    selectedNodeIds.size > 1 && actionTargets.selected.start.length > 0
  const canRunAll =
    runAction === null &&
    bulkRun === null &&
    (actionTargets.all.start.length > 0 ||
      actionTargets.all.restart.length > 0) &&
    (!needsHarnessForAll || !!configuredHarnessCommand)
  const canStopAll =
    runAction === null && bulkRun === null && actionTargets.all.stop.length > 0
  const showSelectedActions = selectedNodeIds.size > 1
  const canRunSelected =
    runAction === null &&
    bulkRun === null &&
    showSelectedActions &&
    (actionTargets.selected.start.length > 0 ||
      actionTargets.selected.restart.length > 0) &&
    (!needsHarnessForSelected || !!configuredHarnessCommand)
  const canStopSelected =
    runAction === null &&
    bulkRun === null &&
    showSelectedActions &&
    actionTargets.selected.stop.length > 0

  async function runTargets(targets: {
    start: number[]
    restart: number[]
    stop: number[]
  }) {
    const total = targets.start.length + targets.restart.length
    if (total === 0) {
      return
    }
    if (targets.start.length > 0 && !configuredHarnessCommand) {
      throw new Error("Set a harness command in Configure")
    }

    const runId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setBulkRun({ runId, total, completed: 0, failed: 0 })

    await runTaskAgentsBulk({
      runId,
      startTaskIds: targets.start,
      restartTaskIds: targets.restart,
      harness: targets.start.length > 0 ? configuredHarnessCommand : null,
      detach: configuredDetach,
      prelude: null,
    })
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
      setRunNotice("Starting…")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
      setBulkRun(null)
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
      setRunNotice("Starting selection…")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
      setBulkRun(null)
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
          disabledReason={loading ? "Refreshing…" : null}
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
            {runBuckets ? (
              <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
                <Button
                  variant={
                    selectionEquals(runBuckets.running) ? "secondary" : "ghost"
                  }
                  size="sm"
                  className="h-full rounded-none border-0 px-1.5 leading-none"
                  onClick={() => selectBucketNodes(runBuckets.running)}
                  disabledReason={
                    runSummary.running > 0 ? null : "No tasks to select"
                  }
                >
                  <span className="truncate">Running</span>
                  <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/80">
                    {runSummary.running}
                  </span>
                </Button>
                <Button
                  variant={
                    selectionEquals(runBuckets.eligible) ? "secondary" : "ghost"
                  }
                  size="sm"
                  className="h-full rounded-none border-0 border-l px-1.5 leading-none"
                  onClick={() => selectBucketNodes(runBuckets.eligible)}
                  disabledReason={
                    runSummary.eligible > 0 ? null : "No tasks to select"
                  }
                >
                  <span className="truncate">Eligible</span>
                  <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/80">
                    {runSummary.eligible}
                  </span>
                </Button>
                <Button
                  variant={
                    selectionEquals(runBuckets.blocked) ? "secondary" : "ghost"
                  }
                  size="sm"
                  className="h-full rounded-none border-0 border-l px-1.5 leading-none"
                  onClick={() => selectBucketNodes(runBuckets.blocked)}
                  disabledReason={
                    runSummary.blocked > 0 ? null : "No tasks to select"
                  }
                >
                  <span className="truncate">Blocked</span>
                  <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/80">
                    {runSummary.blocked}
                  </span>
                </Button>
                <Button
                  variant={
                    selectionEquals(runBuckets.failed) ? "secondary" : "ghost"
                  }
                  size="sm"
                  className="h-full rounded-none border-0 border-l px-1.5 leading-none"
                  onClick={() => selectBucketNodes(runBuckets.failed)}
                  disabledReason={
                    runSummary.failed > 0 ? null : "No tasks to select"
                  }
                >
                  <span className="truncate">Failed</span>
                  <span className="rounded-full bg-foreground/10 px-1.5 py-0.5 text-[0.625rem] text-foreground/80">
                    {runSummary.failed}
                  </span>
                </Button>
              </div>
            ) : null}
            <div className="flex items-center gap-2">
              <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-full rounded-none border-0 leading-none"
                  title="Start or restart all eligible tasks"
                  disabledReason={
                    canRunAll
                      ? null
                      : runAction !== null || bulkRun !== null
                        ? "Action in progress"
                        : needsHarnessForAll && !configuredHarnessCommand
                          ? "Set a harness command in Configure"
                          : "Nothing to start"
                  }
                  onClick={() => void handleRunAll()}
                >
                  {bulkRun ? <Loader2 className="animate-spin" /> : <Play />}
                  Run all
                </Button>
                <div
                  className={
                    "flex items-stretch overflow-hidden transition-[max-width,opacity] duration-200 " +
                    (showSelectedActions
                      ? "max-w-[10rem] opacity-100"
                      : "pointer-events-none max-w-0 opacity-0")
                  }
                >
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-full rounded-none border-0 border-l leading-none"
                    title="Start or restart selected tasks"
                    disabledReason={
                      canRunSelected
                        ? null
                        : runAction !== null || bulkRun !== null
                          ? "Action in progress"
                          : needsHarnessForSelected && !configuredHarnessCommand
                            ? "Set a harness command in Configure"
                            : !showSelectedActions
                              ? "Select 2+ tasks"
                              : "Nothing to start in selection"
                    }
                    onClick={() => void handleRunSelected()}
                  >
                    {bulkRun ? <Loader2 className="animate-spin" /> : null}
                    Selected
                  </Button>
                </div>
              </div>

              <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-full rounded-none border-0 leading-none"
                  title="Stop all running tasks"
                  disabledReason={
                    canStopAll
                      ? null
                      : runAction !== null || bulkRun !== null
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
                    "flex items-stretch overflow-hidden transition-[max-width,opacity] duration-200 " +
                    (showSelectedActions
                      ? "max-w-[10rem] opacity-100"
                      : "pointer-events-none max-w-0 opacity-0")
                  }
                >
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-full rounded-none border-0 border-l leading-none"
                    title="Stop selected running tasks"
                    disabledReason={
                      canStopSelected
                        ? null
                        : runAction !== null || bulkRun !== null
                          ? "Action in progress"
                          : !showSelectedActions
                            ? "Select 2+ tasks"
                            : "Nothing to stop in selection"
                    }
                    onClick={() => void handleStopSelected()}
                  >
                    Selected
                  </Button>
                </div>
              </div>

              <Button
                variant="ghost"
                size="sm"
                className="h-6"
                onClick={() => setConfigOpen((open) => !open)}
                title="Configure harness and agent prelude"
              >
                <Settings2 />
                Configure
              </Button>
            </div>
            {/* TODO: Reintroduce after refining Run UX. (See CopyRunCommandButton.) */}
            {bulkRun ? (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                Starting {bulkRun.completed}/{bulkRun.total}
                {bulkRun.failed > 0 ? ` (${bulkRun.failed} failed)` : null}
              </div>
            ) : runNotice ? (
              <div className="text-xs text-muted-foreground">{runNotice}</div>
            ) : null}
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
          {configOpen ? (
            <button
              type="button"
              aria-label="Close configuration panel"
              className="absolute inset-0 z-20 cursor-default bg-transparent"
              onClick={closeConfig}
            />
          ) : null}

          <SlidePanel
            open={configOpen}
            side="left"
            className="z-30 w-[28rem] border-border/60 bg-background/80 backdrop-blur"
          >
            <div className="relative grid gap-4 p-4">
              <div className="sticky top-4 z-20 h-0 pointer-events-none">
                <div className="flex justify-end">
                  <div className="inline-flex h-6 w-fit overflow-hidden rounded-md border border-border/60 bg-background/40 shadow-sm backdrop-blur pointer-events-auto">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-full rounded-none border-0 leading-none"
                      onClick={() => void handleSaveConfig()}
                      disabledReason={
                        configPending
                          ? "Saving…"
                          : !configDirty
                            ? "No changes"
                            : null
                      }
                    >
                      Save
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-full rounded-none border-0 border-l leading-none"
                      onClick={resetConfigFields}
                      disabledReason={
                        configPending
                          ? "Saving…"
                          : !orchestrationDefaults
                            ? "Defaults not loaded"
                            : !configDirty
                              ? "No changes"
                              : null
                      }
                    >
                      Reset
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-full rounded-none border-0 border-l leading-none"
                      onClick={closeConfig}
                    >
                      Close
                    </Button>
                  </div>
                </div>
              </div>

              <div className="pr-24">
                <div className="text-sm font-medium">Configure</div>
                <div className="mt-1.5 text-xs text-muted-foreground">
                  Updates `config.toml` (repo scope).
                </div>
                {configNotice ? (
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    {configNotice}
                  </div>
                ) : null}
              </div>

              {configError ? (
                <div className="text-xs text-destructive">{configError}</div>
              ) : null}

              <Accordion
                multiple
                defaultValue={["harness", "prelude"]}
                className="border-0"
              >
                <AccordionItem value="harness">
                  <AccordionTrigger>Harness</AccordionTrigger>
                  <AccordionContent className="grid gap-4 pt-3">
                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">
                        Harness command
                      </div>
                      <input
                        className="h-7 rounded-md border bg-background/40 px-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                        value={configHarness}
                        onChange={(e) => setConfigHarness(e.target.value)}
                        placeholder={
                          orchestrationDefaults?.harness.command ?? "codex"
                        }
                        disabled={configPending}
                      />
                      <div className="mt-1.5 text-xs text-muted-foreground">
                        Shell command used to start the harness inside each
                        task’s worktree (e.g.{" "}
                        <span className="font-mono">codex</span>).
                      </div>
                    </div>

                    <div className="grid gap-1">
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-xs text-muted-foreground">
                          Run mode
                        </div>
                        {orchestrationDefaultsLoading ? (
                          <span className="text-xs text-muted-foreground">
                            loading…
                          </span>
                        ) : null}
                      </div>
                      <div className="inline-flex h-6 w-fit overflow-hidden rounded-md border border-border/60">
                        <Button
                          variant={configDetach ? "secondary" : "ghost"}
                          size="sm"
                          className="h-full rounded-none border-0 leading-none"
                          onClick={() => setConfigDetach(true)}
                          disabledReason={configPending ? "Saving…" : null}
                        >
                          Detached
                        </Button>
                        <Button
                          variant={!configDetach ? "secondary" : "ghost"}
                          size="sm"
                          className="h-full rounded-none border-0 border-l leading-none"
                          onClick={() => setConfigDetach(false)}
                          disabledReason={configPending ? "Saving…" : null}
                        >
                          Foreground
                        </Button>
                      </div>
                      <div className="mt-1.5 text-xs text-muted-foreground">
                        Detached runs in a tmux session; foreground runs in your
                        current terminal.
                      </div>
                    </div>

                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">
                        Sandbox
                      </div>
                      <div className="grid gap-2">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs text-foreground">
                            Worktree sandbox
                          </div>
                          <Switch
                            checked={configSandboxType === "worktree"}
                            onCheckedChange={(checked) => {
                              setConfigSandboxType(
                                checked ? "worktree" : "none",
                              )
                              if (!checked) {
                                setConfigSandboxNetwork("allow")
                              }
                            }}
                            disabledReason={configPending ? "Saving…" : null}
                          />
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs text-foreground">
                            Deny network
                          </div>
                          <Switch
                            checked={configSandboxNetwork === "deny"}
                            onCheckedChange={(checked) =>
                              setConfigSandboxNetwork(
                                checked ? "deny" : "allow",
                              )
                            }
                            disabledReason={
                              configPending
                                ? "Saving…"
                                : configSandboxType !== "worktree"
                                  ? "Enable sandbox first"
                                  : null
                            }
                          />
                        </div>
                      </div>
                      <div className="mt-1.5 text-xs text-muted-foreground">
                        Restricts agent writes to the task worktree and Redesmyn
                        state. Enable “Deny network” to force offline operation.
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>

                <AccordionItem value="prelude">
                  <AccordionTrigger>Prelude</AccordionTrigger>
                  <AccordionContent className="grid gap-4 pt-3">
                    <div className="grid gap-1">
                      <div className="flex items-center justify-between gap-3">
                        <div className="text-xs text-muted-foreground">
                          Agent prelude
                        </div>
                        <Popover
                          open={defaultPreludeOpen}
                          onOpenChange={setDefaultPreludeOpen}
                        >
                          <PopoverTrigger
                            render={(triggerProps) => {
                              return (
                                <Button
                                  variant="ghost"
                                  size="xs"
                                  className="h-5 px-2"
                                  disabledReason={
                                    orchestrationDefaultsLoading
                                      ? "Loading…"
                                      : orchestrationDefaults?.harness
                                            .builtInPreludeTemplate
                                        ? null
                                        : "Default prelude unavailable"
                                  }
                                  {...triggerProps}
                                >
                                  Show default
                                </Button>
                              )
                            }}
                          />
                          <PopoverContent className="w-[24rem]">
                            <div className="flex items-center justify-between gap-3">
                              <div className="text-xs font-medium text-foreground">
                                Default prelude
                              </div>
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6"
                                disabledReason={
                                  orchestrationDefaults?.harness
                                    .builtInPreludeTemplate
                                    ? null
                                    : "Default prelude unavailable"
                                }
                                onClick={() => {
                                  const template =
                                    orchestrationDefaults?.harness
                                      .builtInPreludeTemplate ?? ""
                                  setConfigPrelude(template)
                                  setDefaultPreludeOpen(false)
                                }}
                              >
                                Fill as starting point
                              </Button>
                            </div>
                            <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded-md border border-border/60 bg-background/30 p-2 font-mono text-[0.625rem] text-foreground/80">
                              {orchestrationDefaults?.harness
                                .builtInPreludeTemplate ?? ""}
                            </pre>
                          </PopoverContent>
                        </Popover>
                      </div>
                      <textarea
                        className="min-h-[10rem] resize-y rounded-md border bg-background/40 px-2 py-2 text-xs text-foreground shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/30"
                        value={configPrelude}
                        onChange={(e) => setConfigPrelude(e.target.value)}
                        placeholder="Optional. Leave blank to use the built-in prelude."
                        disabled={configPending}
                      />
                      <div className="mt-1.5 text-xs text-muted-foreground">
                        Sent to the agent right after the harness starts. Use it
                        to point the agent at relevant docs and guidance.
                      </div>
                    </div>

                    <div className="grid gap-1">
                      <div className="text-xs text-muted-foreground">
                        Prelude delivery
                      </div>
                      <div className="grid gap-2">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs text-foreground">
                            Auto-send prelude
                          </div>
                          <Switch
                            checked={configSendPrelude}
                            onCheckedChange={(checked) => {
                              setConfigSendPrelude(checked)
                              if (!checked) {
                                setConfigSubmitPrelude(false)
                              }
                            }}
                            disabledReason={configPending ? "Saving…" : null}
                          />
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-xs text-foreground">
                            Press Enter to submit
                          </div>
                          <Switch
                            checked={configSubmitPrelude}
                            onCheckedChange={setConfigSubmitPrelude}
                            disabledReason={
                              configPending
                                ? "Saving…"
                                : !configSendPrelude
                                  ? "Enable Auto-send first"
                                  : null
                            }
                          />
                        </div>
                      </div>
                      <div className="mt-1.5 text-xs text-muted-foreground">
                        Auto-send types the prelude into the harness. Press
                        Enter submits it so the agent starts working
                        immediately.
                      </div>
                    </div>

                    <div className="rounded-md border border-border/60 bg-background/30 p-2 text-xs">
                      <div className="text-xs text-muted-foreground">
                        Available placeholders
                      </div>
                      <div className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                        <div className="font-mono text-foreground/80">{`{task_id}`}</div>
                        <div className="text-muted-foreground">
                          Task numeric id.
                        </div>
                        <div className="font-mono text-foreground/80">{`{task_title}`}</div>
                        <div className="text-muted-foreground">Task title.</div>
                        <div className="font-mono text-foreground/80">{`{task_doc}`}</div>
                        <div className="text-muted-foreground">
                          Task README path (if available).
                        </div>
                        <div className="font-mono text-foreground/80">{`{epic_slug}`}</div>
                        <div className="text-muted-foreground">Epic slug.</div>
                        <div className="font-mono text-foreground/80">{`{epic_readme}`}</div>
                        <div className="text-muted-foreground">
                          Epic README path.
                        </div>
                        <div className="font-mono text-foreground/80">{`{branch}`}</div>
                        <div className="text-muted-foreground">
                          Branch name for the task node.
                        </div>
                        <div className="font-mono text-foreground/80">{`{worktree}`}</div>
                        <div className="text-muted-foreground">
                          Absolute path to the task worktree.
                        </div>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </SlidePanel>
          <GraphView
            rootNodes={rootNodes}
            childrenByParent={childrenByParent}
            tasksById={tasksById}
            agentsById={agentsById}
            activityByNodeId={activityByNodeId}
            harnessCommand={configuredHarnessCommand}
            detach={configuredDetach}
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
