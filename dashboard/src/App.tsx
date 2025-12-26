import { useEffect, useState } from "react"
import { ContentPanel } from "@/components/ui/content-panel"
import { Sidebar } from "@/components/layout/Sidebar"
import { Header } from "@/components/layout/Header"
import { DetailsPanel } from "@/components/layout/DetailsPanel"
import { GraphView } from "@/components/graph/GraphView"
import { useTheme } from "@/hooks/useTheme"
import { useEpicGraph } from "@/hooks/useEpicGraph"
import { formatBranchName } from "@/lib/graph-utils"

function App() {
  const { theme, cycleTheme } = useTheme()
  const {
    epics,
    selectedEpicId,
    graph,
    selectedNodeId,
    setSelectedNodeId,
    error,
    loading,
    refresh,
    tasksById,
    agentsById,
    childrenByParent,
    rootNodes,
    selectedEpic,
    selectedNode,
    selectedTask,
  } = useEpicGraph()

  const [projectMenuOpen, setProjectMenuOpen] = useState(false)

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Escape") {
        return
      }
      if (projectMenuOpen) {
        setProjectMenuOpen(false)
        return
      }
      if (selectedNodeId !== null) {
        setSelectedNodeId(null)
      }
    }

    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [projectMenuOpen, selectedNodeId, setSelectedNodeId])

  const selectedLabel =
    selectedTask?.title ??
    (selectedNode
      ? formatBranchName(selectedNode.branchName, selectedEpic?.slug)
      : null)

  return (
    <div className="h-screen w-screen bg-surface text-foreground">
      <div className="flex h-full gap-2 p-2">
        <Sidebar
          epics={epics}
          selectedEpic={selectedEpic}
          selectedEpicId={selectedEpicId}
          loading={loading}
          projectMenuOpen={projectMenuOpen}
          onProjectMenuToggle={() => setProjectMenuOpen((open) => !open)}
          onProjectMenuClose={() => setProjectMenuOpen(false)}
          onSelectEpic={(epicId) => {
            void refresh(epicId)
            setProjectMenuOpen(false)
          }}
          theme={theme}
          onCycleTheme={cycleTheme}
        />

        <ContentPanel>
          <Header
            selectedEpic={selectedEpic}
            selectedLabel={selectedLabel}
            loading={loading}
            error={error}
            onClearSelection={() => setSelectedNodeId(null)}
            onRefresh={() => void refresh(selectedEpicId)}
          />

          {graph ? (
            <div className="relative flex min-h-0 flex-1">
              <GraphView
                rootNodes={rootNodes}
                childrenByParent={childrenByParent}
                tasksById={tasksById}
                agentsById={agentsById}
                selectedNodeId={selectedNodeId}
                epicSlug={selectedEpic?.slug}
                onSelectNode={setSelectedNodeId}
                onClearSelection={() => setSelectedNodeId(null)}
              />

              <DetailsPanel open={!!selectedNode} task={selectedTask} />
            </div>
          ) : (
            <div className="flex-1 p-6 text-sm text-muted-foreground">
              Select a project to view its graph.
            </div>
          )}
        </ContentPanel>
      </div>
    </div>
  )
}

export default App
