import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import { TaskCard } from "@/components/graph/TaskCard"
import {
  displayBranchLabel,
  type AgentSession,
  type GraphNode,
  type MergeRun,
  type Task,
} from "@/lib/graph-utils"

export type FlowBranchNodeData = Record<string, unknown> & {
  node: GraphNode
  task?: Task
  mergeRun?: MergeRun
  blockingMergeRun?: MergeRun
  agentSession?: AgentSession
  epicSlug?: string | null
  gitMutationsDisabledReason?: string | null
  stackProjectionsFresh: boolean
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
    mergeRun,
    blockingMergeRun,
    agentSession,
    epicSlug,
    gitMutationsDisabledReason,
    stackProjectionsFresh,
    harnessCommand,
    detach,
    edgeHighlighted,
    onSelectNode,
    onRequestRefresh,
  } = data
  const { label: branchLabel, provisional: branchLabelProvisional } =
    displayBranchLabel(node, epicSlug)

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
      <TaskCard
        node={node}
        task={task}
        agentSession={agentSession}
        mergeRun={mergeRun}
        blockingMergeRun={blockingMergeRun}
        branchLabel={branchLabel}
        branchLabelProvisional={branchLabelProvisional}
        gitMutationsDisabledReason={gitMutationsDisabledReason ?? null}
        stackProjectionsFresh={stackProjectionsFresh}
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
