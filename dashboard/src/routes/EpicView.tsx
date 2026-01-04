import { Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { RepoDaemonStatusChip } from "@/components/daemon/RepoDaemonStatusChip"
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
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { bulkTaskAgentActions, updateOrchestrationDefaults } from "@/api"
import { SlidePanel } from "@/components/ui/slide-panel"
import { useEpics } from "@/hooks/useEpics"
import { useDaemons } from "@/hooks/useDaemons"
import {
  type StreamEvent,
  type TaskMergeEventData,
  useEventStream,
} from "@/hooks/useEventStream"
import { useGraph } from "@/hooks/useGraph"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { formatBranchName, makeEdgeId } from "@/lib/graph-utils"
import { computeRepoDaemonStatus } from "@/lib/repo-daemon-status"
import { cn } from "@/lib/utils"
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

const BULK_ACTION_TTL_MS = 10 * 60 * 1000

type BulkActionKind = "run" | "stop"

type BulkActionState = {
  runId: string
  kind: BulkActionKind
  total: number
  completed: number
  failed: number
  startedAt: number
}

type BulkActionRequestItem = {
  taskId: number
  action: "start" | "restart" | "stop"
}

function bulkActionStorageKey(epicSlug: string) {
  return `redesmyn:bulk-action:${epicSlug}`
}

function readStoredBulkAction(epicSlug: string): BulkActionState | null {
  try {
    const raw = window.sessionStorage.getItem(bulkActionStorageKey(epicSlug))
    if (!raw) {
      return null
    }
    const parsed = JSON.parse(raw) as unknown
    if (!parsed || typeof parsed !== "object") {
      return null
    }
    const state = parsed as Partial<BulkActionState>
    if (
      typeof state.runId !== "string" ||
      (state.kind !== "run" && state.kind !== "stop") ||
      typeof state.total !== "number" ||
      typeof state.completed !== "number" ||
      typeof state.failed !== "number" ||
      typeof state.startedAt !== "number"
    ) {
      return null
    }
    if (
      !Number.isFinite(state.total) ||
      !Number.isFinite(state.completed) ||
      !Number.isFinite(state.failed) ||
      !Number.isFinite(state.startedAt)
    ) {
      return null
    }
    if (state.total <= 0 || state.completed < 0 || state.failed < 0) {
      return null
    }
    return state as BulkActionState
  } catch {
    return null
  }
}

export function EpicView() {
  const navigate = useNavigate()
  const params = useParams({ strict: false })
  const epicSlug = params.epicSlug as string | undefined
  const taskIdParam = params.taskId as string | undefined
  const taskId = taskIdParam ? parseInt(taskIdParam, 10) : null
  const fromTaskIdParam = params.fromTaskId as string | undefined
  const toTaskIdParam = params.toTaskId as string | undefined
  const parsedFromTaskId = fromTaskIdParam
    ? parseInt(fromTaskIdParam, 10)
    : null
  const parsedToTaskId = toTaskIdParam ? parseInt(toTaskIdParam, 10) : null
  const fromTaskId =
    parsedFromTaskId !== null && !Number.isNaN(parsedFromTaskId)
      ? parsedFromTaskId
      : null
  const toTaskId =
    parsedToTaskId !== null && !Number.isNaN(parsedToTaskId)
      ? parsedToTaskId
      : null
  const selectedEdgeId =
    fromTaskId !== null && toTaskId !== null
      ? makeEdgeId(fromTaskId, toTaskId)
      : null

  const {
    epics,
    loading: epicsLoading,
    error: epicsError,
    refresh: refreshEpics,
  } = useEpics()
  const { daemons, refresh: refreshDaemons } = useDaemons()
  const [epicMenuOpen, setEpicMenuOpen] = useState(false)
  const [focusMode, setFocusMode] = useState(false)
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<number>>(new Set())
  const selectedNodeIdsRef = useRef(selectedNodeIds)
  const [runAction, setRunAction] =
    useState<"runAll" | "runSelected" | "stopAll" | "stopSelected" | null>(null)
  const [bulkAction, setBulkAction] = useState<BulkActionState | null>(null)
  const refreshTimerRef = useRef<number | null>(null)
  const streamRefreshTimerRef = useRef<number | null>(null)
  const daemonRefreshTimerRef = useRef<number | null>(null)
  const [activityByNodeId, setActivityByNodeId] =
    useState<Map<number, NodeActivity>>(new Map())
  const [, setActivityTick] = useState(0)
  const [mergeStepCue, setMergeStepCue] = useState<{
    id: string
    nodeId: number
    kind: TaskMergeEventData["kind"]
  } | null>(null)

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

  useEffect(() => {
    if (!selectedEpic?.slug) {
      setBulkAction(null)
      return
    }

    const stored = readStoredBulkAction(selectedEpic.slug)
    if (!stored) {
      return
    }

    const ageMs = Date.now() - stored.startedAt
    if (
      ageMs < 0 ||
      ageMs > BULK_ACTION_TTL_MS ||
      stored.completed >= stored.total
    ) {
      try {
        window.sessionStorage.removeItem(
          bulkActionStorageKey(selectedEpic.slug),
        )
      } catch {
        // ignore
      }
      return
    }

    setBulkAction(stored)
  }, [selectedEpic?.slug])

  useEffect(() => {
    if (!selectedEpic?.slug) {
      return
    }

    const key = bulkActionStorageKey(selectedEpic.slug)
    try {
      if (bulkAction === null) {
        window.sessionStorage.removeItem(key)
      } else {
        window.sessionStorage.setItem(key, JSON.stringify(bulkAction))
      }
    } catch {
      // ignore
    }
  }, [bulkAction, selectedEpic?.slug])

  const {
    graph,
    error: graphError,
    loading: graphLoading,
    refresh: refreshGraph,
    tasksById,
    mergeRunsByTaskId,
    agentSessionsByNodeId,
    childrenByParent,
    rootNodes,
  } = useGraph(selectedEpic?.id ?? null)

  const selectedTask = useMemo(
    () => (taskId !== null ? (tasksById.get(taskId) ?? null) : null),
    [taskId, tasksById],
  )

  const selectedMergeRun = useMemo(() => {
    if (!selectedTask) {
      return null
    }
    return mergeRunsByTaskId.get(selectedTask.id) ?? null
  }, [mergeRunsByTaskId, selectedTask])

  const selectedEdge = useMemo(() => {
    if (
      fromTaskId === null ||
      toTaskId === null ||
      Number.isNaN(fromTaskId) ||
      Number.isNaN(toTaskId)
    ) {
      return null
    }
    const fromTask = tasksById.get(fromTaskId) ?? null
    const toTask = tasksById.get(toTaskId) ?? null
    if (!fromTask || !toTask) {
      return null
    }

    const fromLabel =
      fromTask.title ?? formatBranchName(fromTask.branchName ?? "", epicSlug)
    const toLabel =
      toTask.title ?? formatBranchName(toTask.branchName ?? "", epicSlug)

    return {
      id: makeEdgeId(fromTask.id, toTask.id),
      fromNodeId: fromTask.id,
      toNodeId: toTask.id,
      fromLabel,
      toLabel,
    }
  }, [epicSlug, fromTaskId, tasksById, toTaskId])

  const selectedLabel = selectedTask
    ? (selectedTask.title ??
      formatBranchName(selectedTask.branchName ?? "", epicSlug))
    : selectedEdge
      ? `${selectedEdge.fromLabel} → ${selectedEdge.toLabel}`
      : null

  const loading = epicsLoading || graphLoading
  const error = epicsError || graphError

  const repoDaemonStatus = useMemo(
    () =>
      computeRepoDaemonStatus({
        repoExecutor: graph?.repoExecutor ?? null,
        daemons,
      }),
    [daemons, graph?.repoExecutor],
  )

  const stackProjectionsFresh = repoDaemonStatus.telemetryFresh

  const runSummary = useMemo(() => {
    if (!graph) {
      return null
    }
    const tasks = graph.tasks ?? []

    let eligible = 0
    let running = 0
    let blocked = 0
    let failed = 0
    let outOfSync: number | null = stackProjectionsFresh ? 0 : null

    for (const task of tasks) {
      if (task.branchName === null) {
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
      const session = agentSessionsByNodeId.get(task.id) ?? null

      if (stackProjectionsFresh && task.stackInSync === false) {
        outOfSync = (outOfSync ?? 0) + 1
      }

      if (session?.status === "running") {
        running += 1
      } else if (session?.status === "blocked") {
        blocked += 1
      } else if (session?.status === "error") {
        failed += 1
      }
    }

    return { eligible, running, blocked, failed, outOfSync }
  }, [agentSessionsByNodeId, graph, stackProjectionsFresh])

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
      if (task.branchName === null) {
        continue
      }
      if (task.state === "blocked" || task.state === "done") {
        continue
      }

      const session = agentSessionsByNodeId.get(task.id) ?? null
      const status = session?.status ?? null

      if (!session || status === "stopped") {
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
      const task = tasksById.get(selectedNodeId) ?? null
      if (!task || task.state === "blocked" || task.state === "done") {
        continue
      }
      const session = agentSessionsByNodeId.get(task.id) ?? null
      const status = session?.status ?? null

      if (!session || status === "stopped") {
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
  }, [agentSessionsByNodeId, graph, selectedNodeIds, tasksById])

  const runBuckets = useMemo(() => {
    if (!graph) {
      return null
    }

    const eligible = new Set<number>()
    const running = new Set<number>()
    const blocked = new Set<number>()
    const failed = new Set<number>()
    const outOfSync = new Set<number>()

    for (const task of graph.tasks ?? []) {
      if (task.branchName === null) {
        continue
      }
      if (task.state === "done") {
        continue
      }

      if (stackProjectionsFresh && task.stackInSync === false) {
        outOfSync.add(task.id)
      }

      if (task.state === "blocked") {
        blocked.add(task.id)
        continue
      }

      eligible.add(task.id)
      const session = agentSessionsByNodeId.get(task.id) ?? null

      if (session?.status === "running") {
        running.add(task.id)
      } else if (session?.status === "blocked") {
        blocked.add(task.id)
      } else if (session?.status === "error") {
        failed.add(task.id)
      }
    }

    return {
      eligible: Array.from(eligible),
      running: Array.from(running),
      blocked: Array.from(blocked),
      failed: Array.from(failed),
      outOfSync: Array.from(outOfSync),
    }
  }, [agentSessionsByNodeId, graph, stackProjectionsFresh])

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

  const scheduleStreamGraphRefresh = useCallback(() => {
    if (streamRefreshTimerRef.current !== null) {
      return
    }
    streamRefreshTimerRef.current = window.setTimeout(() => {
      streamRefreshTimerRef.current = null
      void refreshGraph()
    }, 750)
  }, [refreshGraph])

  const scheduleDaemonsRefresh = useCallback(() => {
    if (daemonRefreshTimerRef.current !== null) {
      return
    }
    daemonRefreshTimerRef.current = window.setTimeout(() => {
      daemonRefreshTimerRef.current = null
      void refreshDaemons()
    }, 250)
  }, [refreshDaemons])

  const handleStreamEvent = useCallback(
    (event: StreamEvent) => {
      if (event.eventType.startsWith("daemon.")) {
        scheduleDaemonsRefresh()
        scheduleGraphRefresh()
        return
      }
      const taskId =
        "taskId" in event.data && typeof event.data.taskId === "number"
          ? event.data.taskId
          : null

      const observedAt = Date.parse(event.createdAt) || Date.now()
      if (taskId !== null) {
        setActivityByNodeId((prev) => {
          const next = new Map(prev)
          const current = next.get(taskId) ?? {}
          if (event.eventType === "git.commit") {
            next.set(taskId, { ...current, lastCommitAt: observedAt })
          } else if (event.eventType === "worktree.health") {
            next.set(taskId, { ...current, lastWorktreeAt: observedAt })
          }
          return next
        })
      }

      if (event.eventType === "merge.run") {
        scheduleGraphRefresh()
        return
      }

      if (event.eventType === "node.stack_in_sync") {
        scheduleStreamGraphRefresh()
        return
      }

      if (event.eventType === "task.merge") {
        const data = event.data
        if (
          data.type === "task.merge" &&
          data.phase === "started" &&
          typeof data.taskId === "number"
        ) {
          setMergeStepCue({
            id: `${event.id}:${data.runId}:${data.kind}:${data.taskId}`,
            nodeId: data.taskId,
            kind: data.kind,
          })
        }
        return
      }

      if (event.eventType !== "task.agent_run") {
        if (event.eventType !== "task.agent_action") {
          return
        }
      }

      scheduleStreamGraphRefresh()

      const runId =
        event.data.type === "task.agent_action" ||
        event.data.type === "task.agent_run"
          ? event.data.runId
          : null
      const phase =
        event.data.type === "task.agent_action" ||
        event.data.type === "task.agent_run"
          ? event.data.phase
          : null

      if (!runId) {
        return
      }

      if (phase !== "started" && phase !== "stopped" && phase !== "failed") {
        return
      }

      setBulkAction((current) => {
        if (!current || current.runId !== runId) {
          return current
        }

        const completed = current.completed + 1
        const failed = current.failed + (phase === "failed" ? 1 : 0)

        if (completed >= current.total) {
          const label = current.kind === "stop" ? "Stopped" : "Started"
          setRunNotice(
            failed > 0
              ? `${label} (with ${failed} failure${failed === 1 ? "" : "s"})`
              : label,
          )
          scheduleGraphRefresh()
          return null
        }

        return { ...current, completed, failed }
      })
    },
    [scheduleDaemonsRefresh, scheduleGraphRefresh, scheduleStreamGraphRefresh],
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
      if (streamRefreshTimerRef.current !== null) {
        window.clearTimeout(streamRefreshTimerRef.current)
      }
      if (daemonRefreshTimerRef.current !== null) {
        window.clearTimeout(daemonRefreshTimerRef.current)
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
    if (taskId === null) {
      setSelectedNodeIds(new Set())
      return
    }
    setSelectedNodeIds((current) => {
      if (current.size === 0) {
        return new Set([taskId])
      }
      if (current.has(taskId)) {
        return current
      }
      const next = new Set(current)
      next.add(taskId)
      return next
    })
  }, [selectedEdgeId, taskId])

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
        if ((taskId !== null || selectedEdgeId !== null) && epicSlug) {
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
        if (!selectedTask) {
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
    selectedEdgeId,
    selectedTask,
    taskId,
  ])

  function handleSelectNode(id: number, options: { additive: boolean }) {
    if (!epicSlug) {
      return
    }

    if (!options.additive) {
      setSelectedNodeIds(new Set([id]))
      void navigate({
        to: "/graph/$epicSlug/$taskId",
        params: { epicSlug, taskId: String(id) },
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
        to: "/graph/$epicSlug/$taskId",
        params: { epicSlug, taskId: String(id) },
      })
      return
    }

    if (taskId !== id) {
      return
    }

    const nextPrimary = next.values().next().value ?? null
    if (nextPrimary === null) {
      void navigate({ to: "/graph/$epicSlug", params: { epicSlug } })
      return
    }

    void navigate({
      to: "/graph/$epicSlug/$taskId",
      params: { epicSlug, taskId: String(nextPrimary) },
    })
  }

  function handleSelectEdge(fromId: number, toId: number) {
    setFocusMode(false)
    setSelectedNodeIds(new Set())
    if (epicSlug) {
      void navigate({
        to: "/graph/$epicSlug/e/$fromTaskId/$toTaskId",
        params: {
          epicSlug,
          fromTaskId: String(fromId),
          toTaskId: String(toId),
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
    await refreshDaemons()
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
    bulkAction === null &&
    (actionTargets.all.start.length > 0 ||
      actionTargets.all.restart.length > 0) &&
    (!needsHarnessForAll || !!configuredHarnessCommand)
  const canStopAll =
    runAction === null &&
    bulkAction === null &&
    actionTargets.all.stop.length > 0
  const showSelectedActions = selectedNodeIds.size > 1
  const canRunSelected =
    runAction === null &&
    bulkAction === null &&
    showSelectedActions &&
    (actionTargets.selected.start.length > 0 ||
      actionTargets.selected.restart.length > 0) &&
    (!needsHarnessForSelected || !!configuredHarnessCommand)
  const canStopSelected =
    runAction === null &&
    bulkAction === null &&
    showSelectedActions &&
    actionTargets.selected.stop.length > 0

  async function submitBulkActions(options: {
    kind: BulkActionKind
    actions: BulkActionRequestItem[]
    harness: string | null
  }) {
    if (options.actions.length === 0) {
      return
    }
    if (options.actions.some((a) => a.action === "start") && !options.harness) {
      throw new Error("Set a harness command in Configure")
    }

    const runId =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`
    setBulkAction({
      runId,
      kind: options.kind,
      total: options.actions.length,
      completed: 0,
      failed: 0,
      startedAt: Date.now(),
    })

    await bulkTaskAgentActions({
      runId,
      actions: options.actions.map((a) => ({
        taskId: a.taskId,
        action: a.action,
      })),
      harness: options.harness,
      detach: configuredDetach,
      prelude: null,
    })
  }

  async function handleRunAll() {
    if (!canRunAll) {
      return
    }
    setRunAction("runAll")
    try {
      await submitBulkActions({
        kind: "run",
        actions: [
          ...actionTargets.all.start.map((taskId) => ({
            taskId,
            action: "start" as const,
          })),
          ...actionTargets.all.restart.map((taskId) => ({
            taskId,
            action: "restart" as const,
          })),
        ],
        harness:
          actionTargets.all.start.length > 0 ? configuredHarnessCommand : null,
      })
      scheduleGraphRefresh()
      setRunNotice("Starting…")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
      setBulkAction(null)
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
      await submitBulkActions({
        kind: "stop",
        actions: actionTargets.all.stop.map((taskId) => ({
          taskId,
          action: "stop" as const,
        })),
        harness: null,
      })
      scheduleGraphRefresh()
      setRunNotice("Stopping…")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
      setBulkAction(null)
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
      await submitBulkActions({
        kind: "run",
        actions: [
          ...actionTargets.selected.start.map((taskId) => ({
            taskId,
            action: "start" as const,
          })),
          ...actionTargets.selected.restart.map((taskId) => ({
            taskId,
            action: "restart" as const,
          })),
        ],
        harness:
          actionTargets.selected.start.length > 0
            ? configuredHarnessCommand
            : null,
      })
      scheduleGraphRefresh()
      setRunNotice("Starting selection…")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
      setBulkAction(null)
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
      await submitBulkActions({
        kind: "stop",
        actions: actionTargets.selected.stop.map((taskId) => ({
          taskId,
          action: "stop" as const,
        })),
        harness: null,
      })
      scheduleGraphRefresh()
      setRunNotice("Stopping selection…")
    } catch (e) {
      setRunNotice(e instanceof Error ? e.message : String(e))
      setBulkAction(null)
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

        <div className="flex items-center justify-end gap-2">
          <RepoDaemonStatusChip
            status={repoDaemonStatus}
            startCommand="rn daemon run"
          />
          <Button
            variant="outline"
            onClick={() => void handleRefresh()}
            disabledReason={loading ? "Refreshing…" : null}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </Button>
        </div>
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
                {runSummary.outOfSync === null ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant="ghost"
                          size="sm"
                          className={cn(
                            "h-full rounded-none border-0 border-l px-1.5 leading-none",
                            triggerProps.className,
                          )}
                          disabledReason="Telemetry stale; sync status unknown"
                        >
                          <span className="truncate">Out of sync</span>
                          <span className="rounded-full bg-amber-400/10 px-1.5 py-0.5 text-[0.625rem] text-amber-200/90">
                            —
                          </span>
                        </Button>
                      )}
                    />
                    <TooltipContent side="bottom" align="center">
                      Telemetry is stale, so sync projections are unknown.
                    </TooltipContent>
                  </Tooltip>
                ) : runSummary.outOfSync > 0 ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant={
                            selectionEquals(runBuckets.outOfSync)
                              ? "secondary"
                              : "ghost"
                          }
                          size="sm"
                          className={cn(
                            "h-full rounded-none border-0 border-l px-1.5 leading-none",
                            triggerProps.className,
                          )}
                          onClick={() =>
                            selectBucketNodes(runBuckets.outOfSync)
                          }
                        >
                          <span className="truncate">Out of sync</span>
                          <span className="rounded-full bg-amber-400/10 px-1.5 py-0.5 text-[0.625rem] text-amber-200/90">
                            {runSummary.outOfSync}
                          </span>
                        </Button>
                      )}
                    />
                    <TooltipContent side="bottom" align="center">
                      Tasks whose branch is out of sync with its effective
                      upstream (ignores merged ancestors).
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-full rounded-none border-0 border-l px-1.5 leading-none"
                    onClick={() => selectBucketNodes(runBuckets.outOfSync)}
                    disabledReason="No tasks to select"
                  >
                    <span className="truncate">Out of sync</span>
                    <span className="rounded-full bg-amber-400/10 px-1.5 py-0.5 text-[0.625rem] text-amber-200/90">
                      {runSummary.outOfSync}
                    </span>
                  </Button>
                )}
              </div>
            ) : null}
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger
                  render={(triggerProps) => (
                    <Button
                      {...triggerProps}
                      variant="ghost"
                      size="sm"
                      className={cn("h-6", triggerProps.className)}
                      onClick={() => setConfigOpen((open) => !open)}
                    >
                      <Settings2 />
                      Configure
                    </Button>
                  )}
                />
                <TooltipContent side="bottom" align="center" sideOffset={10}>
                  Configure harness and agent prelude
                </TooltipContent>
              </Tooltip>

              <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
                {canRunAll ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant="outline"
                          size="sm"
                          className={cn(
                            "h-full rounded-none border-0 leading-none",
                            triggerProps.className,
                          )}
                          onClick={() => void handleRunAll()}
                        >
                          <Play className="text-emerald-400" />
                          Run all
                        </Button>
                      )}
                    />
                    <TooltipContent
                      side="bottom"
                      align="center"
                      sideOffset={10}
                    >
                      Start or restart all eligible tasks
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-full rounded-none border-0 leading-none"
                    disabledReason={
                      runAction !== null || bulkAction !== null
                        ? "Action in progress"
                        : needsHarnessForAll && !configuredHarnessCommand
                          ? "Set a harness command in Configure"
                          : "Nothing to start"
                    }
                  >
                    {bulkAction?.kind === "run" ? (
                      <Loader2 className="animate-spin text-emerald-400" />
                    ) : (
                      <Play className="text-emerald-400" />
                    )}
                    Run all
                  </Button>
                )}
                <div
                  className={
                    "flex items-stretch overflow-hidden transition-[max-width,opacity] duration-200 " +
                    (showSelectedActions
                      ? "max-w-[10rem] opacity-100"
                      : "pointer-events-none max-w-0 opacity-0")
                  }
                >
                  {canRunSelected ? (
                    <Tooltip>
                      <TooltipTrigger
                        render={(triggerProps) => (
                          <Button
                            {...triggerProps}
                            variant="outline"
                            size="sm"
                            className={cn(
                              "h-full rounded-none border-0 border-l leading-none",
                              triggerProps.className,
                            )}
                            onClick={() => void handleRunSelected()}
                          >
                            Selected
                          </Button>
                        )}
                      />
                      <TooltipContent
                        side="bottom"
                        align="center"
                        sideOffset={10}
                      >
                        Start or restart selected tasks
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-full rounded-none border-0 border-l leading-none"
                      disabledReason={
                        runAction !== null || bulkAction !== null
                          ? "Action in progress"
                          : needsHarnessForSelected && !configuredHarnessCommand
                            ? "Set a harness command in Configure"
                            : !showSelectedActions
                              ? "Select 2+ tasks"
                              : "Nothing to start in selection"
                      }
                    >
                      {bulkAction?.kind === "run" ? (
                        <Loader2 className="animate-spin text-emerald-400" />
                      ) : null}
                      Selected
                    </Button>
                  )}
                </div>
              </div>

              <div className="flex h-6 overflow-hidden rounded-md border border-border/60">
                {canStopAll ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant="outline"
                          size="sm"
                          className={cn(
                            "h-full rounded-none border-0 leading-none",
                            triggerProps.className,
                          )}
                          onClick={() => void handleStopAll()}
                        >
                          <Square className="text-destructive" />
                          Stop all
                        </Button>
                      )}
                    />
                    <TooltipContent
                      side="bottom"
                      align="center"
                      sideOffset={10}
                    >
                      Stop all running tasks
                    </TooltipContent>
                  </Tooltip>
                ) : (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-full rounded-none border-0 leading-none"
                    disabledReason={
                      runAction !== null || bulkAction !== null
                        ? "Action in progress"
                        : "Nothing to stop"
                    }
                  >
                    {bulkAction?.kind === "stop" ? (
                      <Loader2 className="animate-spin text-destructive" />
                    ) : (
                      <Square className="text-destructive" />
                    )}
                    Stop all
                  </Button>
                )}
                <div
                  className={
                    "flex items-stretch overflow-hidden transition-[max-width,opacity] duration-200 " +
                    (showSelectedActions
                      ? "max-w-[10rem] opacity-100"
                      : "pointer-events-none max-w-0 opacity-0")
                  }
                >
                  {canStopSelected ? (
                    <Tooltip>
                      <TooltipTrigger
                        render={(triggerProps) => (
                          <Button
                            {...triggerProps}
                            variant="outline"
                            size="sm"
                            className={cn(
                              "h-full rounded-none border-0 border-l leading-none",
                              triggerProps.className,
                            )}
                            onClick={() => void handleStopSelected()}
                          >
                            Selected
                          </Button>
                        )}
                      />
                      <TooltipContent
                        side="bottom"
                        align="center"
                        sideOffset={10}
                      >
                        Stop selected running tasks
                      </TooltipContent>
                    </Tooltip>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-full rounded-none border-0 border-l leading-none"
                      disabledReason={
                        runAction !== null || bulkAction !== null
                          ? "Action in progress"
                          : !showSelectedActions
                            ? "Select 2+ tasks"
                            : "Nothing to stop in selection"
                      }
                    >
                      {bulkAction?.kind === "stop" ? (
                        <Loader2 className="animate-spin text-destructive" />
                      ) : null}
                      Selected
                    </Button>
                  )}
                </div>
              </div>
            </div>
            {/* TODO: Reintroduce after refining Run UX. (See CopyRunCommandButton.) */}
            {bulkAction ? (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Loader2
                  className={
                    "h-3 w-3 animate-spin " +
                    (bulkAction.kind === "stop"
                      ? "text-destructive"
                      : "text-emerald-400")
                  }
                />
                {bulkAction.kind === "stop" ? "Stopping" : "Starting"}{" "}
                {bulkAction.completed}/{bulkAction.total}
                {bulkAction.failed > 0
                  ? ` (${bulkAction.failed} failed)`
                  : null}
              </div>
            ) : runNotice ? (
              <div className="text-xs text-muted-foreground">{runNotice}</div>
            ) : null}
          </div>

          {runCommand ? (
            <Tooltip>
              <TooltipTrigger
                render={(triggerProps) => (
                  <div
                    {...triggerProps}
                    className={cn(
                      "mt-1.5 truncate font-mono text-xs text-muted-foreground",
                      triggerProps.className,
                    )}
                  >
                    {runCommand}
                  </div>
                )}
              />
              <TooltipContent
                side="bottom"
                align="center"
                sideOffset={10}
                className="max-w-[42rem] font-mono"
              >
                {runCommand}
              </TooltipContent>
            </Tooltip>
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
                          Branch name for the task.
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
            agentSessionsByNodeId={agentSessionsByNodeId}
            mergeRunsByTaskId={mergeRunsByTaskId}
            activityByNodeId={activityByNodeId}
            harnessCommand={configuredHarnessCommand}
            detach={configuredDetach}
            trunk={graph.trunk ?? null}
            repoExecutor={graph.repoExecutor ?? null}
            stackProjectionsFresh={stackProjectionsFresh}
            selectedNodeIds={selectedNodeIds}
            selectedNodeId={taskId}
            selectedEdgeId={selectedEdgeId}
            focusMode={focusMode}
            mergeStepCue={mergeStepCue}
            epicSlug={graph.epic.slug}
            routeEpicSlug={epicSlug}
            onSelectNode={handleSelectNode}
            onSelectEdge={handleSelectEdge}
            onClearSelection={handleClearSelection}
            onRequestRefresh={scheduleGraphRefresh}
          />

          <DetailsPanel
            open={!!selectedTask || !!selectedEdge}
            node={selectedTask}
            task={selectedTask}
            agentSession={
              selectedTask
                ? (agentSessionsByNodeId.get(selectedTask.id) ?? null)
                : null
            }
            mergeRun={selectedMergeRun}
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
