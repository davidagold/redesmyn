import { TaskCard } from "./TaskCard"
import {
  displayBranchLabel,
  type AgentSession,
  type GraphNode,
  type Task,
} from "@/lib/graph-utils"

interface NodeTreeProps {
  nodes: GraphNode[]
  childrenByParent: Map<number | null, GraphNode[]>
  tasksById: Map<number, Task>
  agentSessionsByNodeId: Map<number, AgentSession>
  harnessCommand: string
  detach: boolean
  stackProjectionsFresh: boolean
  selectedNodeId: number | null
  epicSlug?: string | null
  onSelectNode: (nodeId: number) => void
  depth?: number
}

export function NodeTree({
  nodes,
  childrenByParent,
  tasksById,
  agentSessionsByNodeId,
  harnessCommand,
  detach,
  stackProjectionsFresh,
  selectedNodeId,
  epicSlug,
  onSelectNode,
  depth = 0,
}: NodeTreeProps) {
  return (
    <>
      {nodes.map((node) => {
        const agentSession = agentSessionsByNodeId.get(node.id) ?? undefined
        const task = tasksById.get(node.id) ?? undefined
        const { label: branchLabel, provisional: branchLabelProvisional } =
          displayBranchLabel(node, epicSlug)
        const children = childrenByParent.get(node.id) ?? []

        return (
          <div key={node.id} style={{ paddingLeft: depth * 16 }}>
            <TaskCard
              node={node}
              task={task}
              agentSession={agentSession}
              branchLabel={branchLabel}
              branchLabelProvisional={branchLabelProvisional}
              harnessCommand={harnessCommand}
              detach={detach}
              stackProjectionsFresh={stackProjectionsFresh}
              isSelected={selectedNodeId === node.id}
              onSelect={(_options) => onSelectNode(node.id)}
            />

            {children.length > 0 && (
              <div className="mt-2 grid gap-2">
                <NodeTree
                  nodes={children}
                  childrenByParent={childrenByParent}
                  tasksById={tasksById}
                  agentSessionsByNodeId={agentSessionsByNodeId}
                  harnessCommand={harnessCommand}
                  detach={detach}
                  stackProjectionsFresh={stackProjectionsFresh}
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
