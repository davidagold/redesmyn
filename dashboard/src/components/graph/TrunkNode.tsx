import { Handle, Position, type Node, type NodeProps } from "@xyflow/react"
import { cn } from "@/lib/utils"

export type TrunkNodeData = Record<string, unknown> & {
  className?: string
}

export type TrunkNodeType = Node<TrunkNodeData, "trunk">

export function TrunkNode({ data }: NodeProps<TrunkNodeType>) {
  return (
    <div
      className={cn(
        "relative h-full w-full rounded-full bg-border/60",
        data.className,
      )}
    >
      <Handle
        type="source"
        position={Position.Bottom}
        className="h-2 w-2 border-0 bg-transparent opacity-0"
      />
    </div>
  )
}
