import { DotGrid } from "@/components/ui/dot-grid"
import { NodeTree } from "./NodeTree"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"

interface GraphViewProps {
  rootNodes: GraphNode[]
  childrenByParent: Map<number | null, GraphNode[]>
  tasksById: Map<number, Task>
  agentsById: Map<number, Agent>
  selectedNodeId: number | null
  epicSlug?: string | null
  onSelectNode: (nodeId: number) => void
  onClearSelection: () => void
}

export function GraphView({
  rootNodes,
  childrenByParent,
  tasksById,
  agentsById,
  selectedNodeId,
  epicSlug,
  onSelectNode,
  onClearSelection,
}: GraphViewProps) {
  function handleBackgroundClick(e: React.MouseEvent) {
    if (!(e.target instanceof Node)) {
      return
    }
    const el = e.target instanceof Element ? e.target : e.target.parentElement
    if (el?.closest("[data-node-card]")) {
      return
    }
    onClearSelection()
  }

  return (
    <main className="relative min-w-0 flex-1 overflow-hidden">
      <DotGrid />
      <div
        className="relative h-full overflow-auto p-6"
        onClick={handleBackgroundClick}
      >
        {rootNodes.length > 0 ? (
          <div className="grid gap-3">
            <NodeTree
              nodes={rootNodes}
              childrenByParent={childrenByParent}
              tasksById={tasksById}
              agentsById={agentsById}
              selectedNodeId={selectedNodeId}
              epicSlug={epicSlug}
              onSelectNode={onSelectNode}
            />
          </div>
        ) : (
          <div className="text-sm text-muted-foreground">No nodes.</div>
        )}
      </div>
    </main>
  )
}
