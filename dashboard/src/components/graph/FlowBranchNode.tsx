import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import { NodeCard } from "@/components/graph/NodeCard"
import {
  formatBranchName,
  type Agent,
  type GraphNode,
  type Task,
} from "@/lib/graph-utils"

export type FlowBranchNodeData = Record<string, unknown> & {
  node: GraphNode
  task?: Task
  agent?: Agent
  epicSlug?: string | null
  edgeHighlighted?: boolean
  onSelectNode: (nodeId: number) => void
}

export type FlowBranchNodeType = Node<FlowBranchNodeData, "branch">

export function FlowBranchNode({
  data,
  selected,
}: NodeProps<FlowBranchNodeType>) {
  const { node, task, agent, epicSlug, edgeHighlighted, onSelectNode } = data
  const branchLabel = formatBranchName(node.branchName, epicSlug)

  return (
    <>
      <Handle
        type="target"
        position={Position.Left}
        className="h-2 w-2 border-0 bg-transparent opacity-0"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="h-2 w-2 border-0 bg-transparent opacity-0"
      />
      <NodeCard
        node={node}
        task={task}
        agent={agent}
        branchLabel={branchLabel}
        isSelected={selected}
        isHighlighted={edgeHighlighted}
        onSelect={() => onSelectNode(node.id)}
      />
    </>
  )
}
