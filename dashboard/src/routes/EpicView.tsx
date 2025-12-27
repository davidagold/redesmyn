import { Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { useEffect, useMemo, useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { Button } from "@/components/ui/button"
import { useEpics } from "@/hooks/useEpics"
import { useGraph } from "@/hooks/useGraph"
import { formatBranchName, makeEdgeId } from "@/lib/graph-utils"
import { ChevronRight } from "lucide-react"

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

      {graph ? (
        <div className="relative flex min-h-0 flex-1">
          <GraphView
            rootNodes={rootNodes}
            childrenByParent={childrenByParent}
            tasksById={tasksById}
            agentsById={agentsById}
            trunk={graph.trunk ?? null}
            selectedNodeId={nodeId}
            selectedEdgeId={selectedEdgeId}
            focusMode={focusMode}
            epicSlug={epicSlug}
            onSelectNode={handleSelectNode}
            onSelectEdge={handleSelectEdge}
            onClearSelection={handleClearSelection}
          />

          <DetailsPanel
            open={!!selectedNode || !!selectedEdge}
            task={selectedTask}
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
