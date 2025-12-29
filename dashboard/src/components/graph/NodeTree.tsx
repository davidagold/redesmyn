import { NodeCard } from "./NodeCard"
import {
  formatBranchName,
  type Agent,
  type GraphNode,
  type Task,
} from "@/lib/graph-utils"

interface NodeTreeProps {
  nodes: GraphNode[]
  childrenByParent: Map<number | null, GraphNode[]>
  tasksById: Map<number, Task>
  agentsById: Map<number, Agent>
  selectedNodeId: number | null
  epicSlug?: string | null
  onSelectNode: (nodeId: number) => void
  depth?: number
}

export function NodeTree({
  nodes,
  childrenByParent,
  tasksById,
  agentsById,
  selectedNodeId,
  epicSlug,
  onSelectNode,
  depth = 0,
}: NodeTreeProps) {
  return (
    <>
      {nodes.map((node) => {
        const task =
          node.primaryTaskId !== null
            ? tasksById.get(node.primaryTaskId)
            : undefined
        const agent =
          node.agentId !== null ? agentsById.get(node.agentId) : undefined
        const branchLabel = formatBranchName(node.branchName, epicSlug)
        const children = childrenByParent.get(node.id) ?? []

        return (
          <div key={node.id} style={{ paddingLeft: depth * 16 }}>
            <NodeCard
              node={node}
              task={task}
              agent={agent}
              branchLabel={branchLabel}
              isSelected={selectedNodeId === node.id}
              onSelect={(_options) => onSelectNode(node.id)}
            />

            {children.length > 0 && (
              <div className="mt-2 grid gap-2">
                <NodeTree
                  nodes={children}
                  childrenByParent={childrenByParent}
                  tasksById={tasksById}
                  agentsById={agentsById}
                  selectedNodeId={selectedNodeId}
                  epicSlug={epicSlug}
                  onSelectNode={onSelectNode}
                  depth={depth + 1}
                />
              </div>
            )}
          </div>
        )
      })}
    </>
  )
}
