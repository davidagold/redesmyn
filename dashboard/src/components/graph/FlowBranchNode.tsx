import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import { NodeCard } from "@/components/graph/NodeCard"
import {
  formatBranchName,
  type Agent,
  type GraphNode,
  type MergeRun,
  type Task,
} from "@/lib/graph-utils"
import type { NodeActivity } from "@/lib/presence"

export type FlowBranchNodeData = Record<string, unknown> & {
  node: GraphNode
  task?: Task
  agent?: Agent
  mergeRun?: MergeRun
  blockingMergeRun?: MergeRun
  activity?: NodeActivity
  epicSlug?: string | null
  harnessCommand: string
  detach: boolean
  edgeHighlighted?: boolean
  onSelectNode: (nodeId: number, options: { additive: boolean }) => void
  onRequestRefresh?: () => void
}

export type FlowBranchNodeType = Node<FlowBranchNodeData, "branch">

export function FlowBranchNode({
  data,
  selected,
}: NodeProps<FlowBranchNodeType>) {
  const {
    node,
    task,
    agent,
    mergeRun,
    blockingMergeRun,
    activity,
    epicSlug,
    harnessCommand,
    detach,
    edgeHighlighted,
    onSelectNode,
    onRequestRefresh,
  } = data
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
        mergeRun={mergeRun}
        blockingMergeRun={blockingMergeRun}
        activity={activity}
        branchLabel={branchLabel}
        harnessCommand={harnessCommand}
        detach={detach}
        isSelected={selected}
        isHighlighted={edgeHighlighted}
        onSelect={(options) => onSelectNode(node.id, options)}
        onRequestRefresh={onRequestRefresh}
      />
    </>
  )
}
