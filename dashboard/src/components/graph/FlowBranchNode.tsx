import type { Node, NodeProps } from "@xyflow/react"
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
  onSelectNode: (nodeId: number) => void
}

export type FlowBranchNodeType = Node<FlowBranchNodeData, "branch">

export function FlowBranchNode({
  data,
  selected,
}: NodeProps<FlowBranchNodeType>) {
  const { node, task, agent, epicSlug, onSelectNode } = data
  const branchLabel = formatBranchName(node.branchName, epicSlug)

  return (
    <NodeCard
      node={node}
      task={task}
      agent={agent}
      branchLabel={branchLabel}
      isSelected={selected}
      onSelect={() => onSelectNode(node.id)}
    />
  )
}
