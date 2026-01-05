import { Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { ConnectionsCluster } from "@/components/layout/ConnectionsCluster"
import { GraphView } from "@/components/graph/GraphView"
import { RepoDaemonStatusChip } from "@/components/daemon/RepoDaemonStatusChip"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { useBulkTaskAgentActionsMutation } from "@/api/mutations"
import { useEpics } from "@/hooks/useEpics"
import { useDaemons } from "@/hooks/useDaemons"
import { useHosts } from "@/hooks/useHosts"
import type { StreamEvent } from "@/hooks/useEventStream"
import { useGraph } from "@/hooks/useGraph"
import { useOrchestrationDefaults } from "@/hooks/useOrchestrationDefaults"
import { queryKeys } from "@/api/queryKeys"
import { useEpicCacheSync } from "@/api/useEpicCacheSync"
import { formatBranchName, makeEdgeId } from "@/lib/graph-utils"
import { computeRepoDaemonStatus } from "@/lib/repo-daemon-status"
import { cn } from "@/lib/utils"
import { ChevronRight, Loader2, Play, Settings2, Square } from "lucide-react"
import { computeRunToolbarModel } from "@/routes/epicView/runToolbarModel"
import { OrchestrationConfigPanel } from "@/routes/epicView/OrchestrationConfigPanel"
import {
  type BulkActionKind,
  type BulkActionRequestItem,
  usePersistedBulkAction,
} from "@/routes/epicView/bulkActionState"

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

  const { epics, loading: epicsLoading, error: epicsError } = useEpics()
  const { daemons } = useDaemons()
  const { hosts } = useHosts()
  const [epicMenuOpen, setEpicMenuOpen] = useState(false)
  const [focusMode, setFocusMode] = useState(false)
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<number>>(new Set())
  const selectedNodeIdsRef = useRef(selectedNodeIds)
  const [runAction, setRunAction] =
    useState<"runAll" | "runSelected" | "stopAll" | "stopSelected" | null>(null)
  const { bulkAction, setBulkAction } = usePersistedBulkAction(epicSlug ?? null)
  const queryClient = useQueryClient()
  const bulkAgentActions = useBulkTaskAgentActionsMutation()
  const refreshTimerRef = useRef<number | null>(null)

  const [runNotice, setRunNotice] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const {
    defaults: orchestrationDefaults,
    loading: orchestrationDefaultsLoading,
  } = useOrchestrationDefaults()

  const closeConfig = useCallback(() => setConfigOpen(false), [])

  const selectedEpic = useMemo(
    () => epics.find((e) => e.slug === epicSlug) ?? null,
    [epics, epicSlug],
  )

  const {
    graph,
    error: graphError,
    loading: graphLoading,
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
        hosts,
      }),
    [daemons, graph?.repoExecutor, hosts],
  )

  const stackProjectionsFresh = repoDaemonStatus.telemetryFresh

  const runToolbarModel = useMemo(
    () =>
      computeRunToolbarModel({
        tasks: graph?.tasks ?? null,
        tasksById,
        agentSessionsByNodeId,
        selectedNodeIds,
        stackProjectionsFresh,
      }),
    [
      agentSessionsByNodeId,
      graph?.tasks,
      selectedNodeIds,
      stackProjectionsFresh,
      tasksById,
    ],
  )

  const runSummary = runToolbarModel.summary
  const runBuckets = runToolbarModel.buckets
  const actionTargets = runToolbarModel.actionTargets

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

  const scheduleGraphRefresh = useCallback(() => {
    if (!selectedEpic) {
      return
    }
    if (refreshTimerRef.current !== null) {
      return
    }
    refreshTimerRef.current = window.setTimeout(() => {
      refreshTimerRef.current = null
      void queryClient.invalidateQueries({
        queryKey: queryKeys.epicGraph(selectedEpic.id),
      })
    }, 250)
  }, [queryClient, selectedEpic])

  const handleStreamEvent = useCallback(
    (event: StreamEvent) => {
      if (
        event.eventType !== "task.agent_run" &&
        event.eventType !== "task.agent_action"
      ) {
        return
      }

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
    [scheduleGraphRefresh, setBulkAction],
  )

  useEpicCacheSync({
    epicSlug: selectedEpic?.slug ?? null,
    epicId: selectedEpic?.id ?? null,
    onEvent: handleStreamEvent,
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
    await Promise.allSettled([
      queryClient.invalidateQueries({ queryKey: queryKeys.epics() }),
      queryClient.invalidateQueries({
        queryKey: queryKeys.orchestrationDefaults(),
      }),
      queryClient.invalidateQueries({ queryKey: queryKeys.daemons() }),
      queryClient.invalidateQueries({ queryKey: queryKeys.hosts() }),
      selectedEpic
        ? queryClient.invalidateQueries({
            queryKey: queryKeys.epicGraph(selectedEpic.id),
          })
        : Promise.resolve(),
    ])
  }

  const configuredHarnessCommand =
    orchestrationDefaults?.harness.command?.trim() ?? ""
  const configuredDetach = orchestrationDefaults?.harness.detach ?? true

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
      throw new Error("Set an agent command in Configure")
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

    if (!selectedEpic) {
      throw new Error("Epic must be selected before starting agents.")
    }

    await bulkAgentActions.mutateAsync({
      epicId: selectedEpic.id,
      request: {
        runId,
        actions: options.actions.map((a) => ({
          taskId: a.taskId,
          action: a.action,
        })),
        harness: options.harness,
        detach: configuredDetach,
        prelude: null,
      },
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
          {epicSlug ? (
            <ConnectionsCluster
              repoDaemonStatus={repoDaemonStatus}
              epicId={selectedEpic?.id ?? null}
              epicSlug={epicSlug}
              startCommand="rn daemon run"
            />
          ) : (
            <RepoDaemonStatusChip
              status={repoDaemonStatus}
              startCommand="rn daemon run"
            />
          )}
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
                Configure agent defaults and prelude
              </TooltipContent>
            </Tooltip>
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
                {runSummary.outOfSync > 0 ? (
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
                          <span className="truncate">
                            Out of sync
                            {stackProjectionsFresh ? null : (
                              <span className="ml-1 text-[0.625rem] text-muted-foreground/70">
                                stale
                              </span>
                            )}
                          </span>
                          <span className="rounded-full bg-amber-400/10 px-1.5 py-0.5 text-[0.625rem] text-amber-200/90">
                            {runSummary.outOfSync}
                          </span>
                        </Button>
                      )}
                    />
                    <TooltipContent
                      side="bottom"
                      align="center"
                      showArrow={false}
                    >
                      <div className="max-w-xs space-y-1">
                        <div>
                          Tasks whose branch is out of sync with its effective
                          upstream (ignores merged ancestors).
                        </div>
                        {stackProjectionsFresh ? null : (
                          <div>Telemetry is stale, so sync projections may be stale.</div>
                        )}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                ) : !stackProjectionsFresh ? (
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
                          onClick={() => selectBucketNodes(runBuckets.outOfSync)}
                          disabledReason="No tasks to select"
                        >
                          <span className="truncate">
                            Out of sync
                            <span className="ml-1 text-[0.625rem] text-muted-foreground/70">
                              stale
                            </span>
                          </span>
                          <span className="rounded-full bg-amber-400/10 px-1.5 py-0.5 text-[0.625rem] text-amber-200/90">
                            {runSummary.outOfSync}
                          </span>
                        </Button>
                      )}
                    />
                    <TooltipContent
                      side="bottom"
                      align="center"
                      showArrow={false}
                    >
                      Telemetry is stale, so sync projections may be stale.
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
            <div className="ml-auto flex items-center gap-2">
              <div className="flex h-7 flex-row-reverse overflow-hidden rounded-md border border-border/60">
                {canRunAll ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant="outline"
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
                    className="h-full rounded-none border-0 leading-none"
                    disabledReason={
                      runAction !== null || bulkAction !== null
                        ? "Action in progress"
                        : needsHarnessForAll && !configuredHarnessCommand
                          ? "Set an agent command in Configure"
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
                    "flex items-stretch justify-end overflow-hidden transition-[max-width,opacity] duration-200 " +
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
                            className={cn(
                              "h-full rounded-none border-0 border-r leading-none",
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
                      className="h-full rounded-none border-0 border-r leading-none"
                      disabledReason={
                        runAction !== null || bulkAction !== null
                          ? "Action in progress"
                          : needsHarnessForSelected && !configuredHarnessCommand
                            ? "Set an agent command in Configure"
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

              <div className="flex h-7 flex-row-reverse overflow-hidden rounded-md border border-border/60">
                {canStopAll ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={(triggerProps) => (
                        <Button
                          {...triggerProps}
                          variant="outline"
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
                    "flex items-stretch justify-end overflow-hidden transition-[max-width,opacity] duration-200 " +
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
                            className={cn(
                              "h-full rounded-none border-0 border-r leading-none",
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
                      className="h-full rounded-none border-0 border-r leading-none"
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
          <OrchestrationConfigPanel
            open={configOpen}
            defaults={orchestrationDefaults}
            defaultsLoading={orchestrationDefaultsLoading}
            onClose={closeConfig}
          />
          <GraphView
            rootNodes={rootNodes}
            childrenByParent={childrenByParent}
            tasksById={tasksById}
            agentSessionsByNodeId={agentSessionsByNodeId}
            mergeRunsByTaskId={mergeRunsByTaskId}
            harnessCommand={configuredHarnessCommand}
            detach={configuredDetach}
            trunk={graph.trunk ?? null}
            repoExecutor={graph.repoExecutor ?? null}
            stackProjectionsFresh={stackProjectionsFresh}
            selectedNodeIds={selectedNodeIds}
            selectedNodeId={taskId}
            selectedEdgeId={selectedEdgeId}
            focusMode={focusMode}
            epicSlug={graph.epic.slug}
            onSelectNode={handleSelectNode}
            onSelectEdge={handleSelectEdge}
            onClearSelection={handleClearSelection}
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
