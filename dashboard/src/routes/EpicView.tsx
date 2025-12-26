import { useNavigate, useParams } from "@tanstack/react-router"
import { useEffect, useMemo, useState } from "react"
import { ContentPanel, ContentPanelHeader } from "@/components/ui/content-panel"
import { EpicSelector } from "@/components/layout/EpicSelector"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { Button } from "@/components/ui/button"
import { useEpics } from "@/hooks/useEpics"
import { useGraph } from "@/hooks/useGraph"
import { formatBranchName } from "@/lib/graph-utils"
import { ChevronRight } from "lucide-react"

export function EpicView() {
  const navigate = useNavigate()
  const params = useParams({ strict: false })
  const epicSlug = params.epicSlug as string | undefined
  const nodeIdParam = params.nodeId as string | undefined
  const nodeId = nodeIdParam ? parseInt(nodeIdParam, 10) : null

  const {
    epics,
    loading: epicsLoading,
    error: epicsError,
    refresh: refreshEpics,
  } = useEpics()
  const [epicMenuOpen, setEpicMenuOpen] = useState(false)

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

  const selectedLabel =
    selectedTask?.title ??
    (selectedNode ? formatBranchName(selectedNode.branchName, epicSlug) : null)

  const loading = epicsLoading || graphLoading
  const error = epicsError || graphError

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Escape") {
        return
      }
      if (epicMenuOpen) {
        setEpicMenuOpen(false)
        return
      }
      if (nodeId !== null && epicSlug) {
        void navigate({ to: "/$epicSlug", params: { epicSlug } })
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [epicMenuOpen, nodeId, epicSlug, navigate])

  function handleSelectNode(id: number) {
    if (epicSlug) {
      void navigate({
        to: "/$epicSlug/$nodeId",
        params: { epicSlug, nodeId: String(id) },
      })
    }
  }

  function handleClearSelection() {
    if (epicSlug) {
      void navigate({ to: "/$epicSlug", params: { epicSlug } })
    }
  }

  function handleSelectEpic(epicId: number) {
    const epic = epics.find((e) => e.id === epicId)
    if (epic) {
      void navigate({ to: "/$epicSlug", params: { epicSlug: epic.slug } })
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
            selectedNodeId={nodeId}
            epicSlug={epicSlug}
            onSelectNode={handleSelectNode}
            onClearSelection={handleClearSelection}
          />

          <DetailsPanel open={!!selectedNode} task={selectedTask} />
        </div>
      ) : (
        <div className="flex-1 p-6 text-sm text-muted-foreground">
          {loading ? "Loading graph..." : "No graph data available."}
        </div>
      )}
    </ContentPanel>
  )
}
