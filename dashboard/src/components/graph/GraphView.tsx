import { DotGrid } from "@/components/ui/dot-grid"
import {
  ReactFlow,
  type DefaultEdgeOptions,
  type Edge,
  type Node,
  type NodeMouseHandler,
  type ReactFlowInstance,
} from "@xyflow/react"
import { useEffect, useMemo, useState, type CSSProperties } from "react"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"
import { FlowBranchNode, type FlowBranchNodeData } from "./FlowBranchNode"
import { layoutTree } from "./flowLayout"

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
  const [flow, setFlow] = useState<ReactFlowInstance | null>(null)

  const defaultEdgeOptions: DefaultEdgeOptions = useMemo(
    () => ({
      type: "smoothstep",
      style: {
        stroke: "var(--border)",
        strokeWidth: 1.25,
      },
    }),
    [],
  )

  const positions = useMemo(
    () =>
      layoutTree(rootNodes, childrenByParent, {
        xSpacing: 360,
        ySpacing: 140,
        xOffset: 40,
        yOffset: 40,
      }),
    [childrenByParent, rootNodes],
  )

  const nodes = useMemo(() => {
    const mapped: Node<FlowBranchNodeData>[] = []
    for (const [nodeId, pos] of positions) {
      const node = rootNodes.find((n) => n.id === nodeId) ?? null
      // Note: positions is computed from the same root/children inputs, so every id
      // should be present in those arrays; this fallback is defensive only.
      const graphNode =
        node ??
        Array.from(childrenByParent.values())
          .flat()
          .find((n) => n.id === nodeId) ??
        null
      if (!graphNode) {
        continue
      }

      const task =
        graphNode.primaryTaskId !== null
          ? tasksById.get(graphNode.primaryTaskId)
          : undefined
      const agent =
        graphNode.agentId !== null
          ? agentsById.get(graphNode.agentId)
          : undefined

      mapped.push({
        id: String(graphNode.id),
        type: "branch",
        position: pos,
        data: {
          node: graphNode,
          task,
          agent,
          epicSlug,
          onSelectNode,
        },
        selectable: true,
        draggable: false,
        focusable: true,
        selected: selectedNodeId === graphNode.id,
      })
    }
    return mapped
  }, [
    agentsById,
    childrenByParent,
    epicSlug,
    onSelectNode,
    positions,
    rootNodes,
    selectedNodeId,
    tasksById,
  ])

  const edges = useMemo(() => {
    const mapped: Edge[] = []
    for (const node of nodes) {
      const graphNode = node.data.node
      if (graphNode.parentNodeId === null) {
        continue
      }
      mapped.push({
        id: `parent:${graphNode.parentNodeId}->${graphNode.id}`,
        source: String(graphNode.parentNodeId),
        target: String(graphNode.id),
        type: "smoothstep",
      })
    }
    return mapped
  }, [nodes])

  useEffect(() => {
    if (!flow || !nodes.length) {
      return
    }
    flow.fitView({ padding: 0.2, duration: 200 })
  }, [flow, nodes.length])

  const onNodeClick: NodeMouseHandler = (_, node) => {
    const nodeId = Number(node.id)
    if (Number.isNaN(nodeId)) {
      return
    }
    onSelectNode(nodeId)
  }

  return (
    <main className="relative min-w-0 flex-1 overflow-hidden">
      <DotGrid />
      {rootNodes.length > 0 ? (
        <div className="relative h-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={{ branch: FlowBranchNode }}
            fitView
            nodesDraggable={false}
            nodesConnectable={false}
            onInit={setFlow}
            onNodeClick={onNodeClick}
            onPaneClick={onClearSelection}
            elevateEdgesOnSelect
            defaultEdgeOptions={defaultEdgeOptions}
            style={
              {
                // Let our own background (surface + dot grid) show through.
                // React Flow's default dark background doesn't match our theme.
                "--xy-background-color": "transparent",
              } as CSSProperties
            }
          />
        </div>
      ) : (
        <div className="relative h-full overflow-auto p-6 text-sm text-muted-foreground">
          No nodes.
        </div>
      )}
    </main>
  )
}
