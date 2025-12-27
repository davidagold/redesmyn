import type { Node, NodeProps } from "@xyflow/react"
import { cn } from "@/lib/utils"

export type TrunkNodeData = Record<string, unknown> & {
  className?: string
}

export type TrunkNodeType = Node<TrunkNodeData, "trunk">

export function TrunkNode({ data }: NodeProps<TrunkNodeType>) {
  return (
    <div
      className={cn("h-full w-full rounded-full bg-border/60", data.className)}
    />
  )
}
