import { DotGrid } from "@/components/ui/dot-grid"
import {
  Position,
  ReactFlow,
  type DefaultEdgeOptions,
  type Edge,
  type ReactFlowInstance,
} from "@xyflow/react"
import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react"
import type { Agent, GraphNode, Task } from "@/lib/graph-utils"
import { makeEdgeId } from "@/lib/graph-utils"
import { FlowBranchNode, type FlowBranchNodeType } from "./FlowBranchNode"
import { CommitStringEdge } from "./CommitStringEdge"
import { TrunkNode, type TrunkNodeType } from "./TrunkNode"
import {
  DETAILS_PANEL_WIDTH_PX,
  GRAPH_EDGE_STYLE_ANIMATION_MS,
  GRAPH_LAYOUT_ANIMATION_MS,
  GRAPH_FIT_MAX_ZOOM,
  GRAPH_FIT_MIN_ZOOM,
  GRAPH_FIT_PADDING_PX,
  GRAPH_NODE_HEIGHT,
  GRAPH_NODE_WIDTH,
  GRAPH_PADDING,
  TRUNK_GAP,
  TRUNK_HEIGHT,
} from "./graphConfig"
import { layoutWithElk } from "./elkLayout"
import { type FlowPosition, layoutTree } from "./flowLayout"
import { applySelectionLens, computeSelectionLens } from "./selectionLens"

interface GraphViewProps {
  rootNodes: GraphNode[]
  childrenByParent: Map<number | null, GraphNode[]>
  tasksById: Map<number, Task>
  agentsById: Map<number, Agent>
  selectedNodeId: number | null
  selectedEdgeId: string | null
  focusMode: boolean
  epicSlug?: string | null
  onSelectNode: (nodeId: number) => void
  onSelectEdge: (fromNodeId: number, toNodeId: number) => void
  onClearSelection: () => void
}

const TRUNK_NODE_ID = "trunk"

type GraphFlowNode = FlowBranchNodeType | TrunkNodeType

export function GraphView({
  rootNodes,
  childrenByParent,
  tasksById,
  agentsById,
  selectedNodeId,
  selectedEdgeId,
  focusMode,
  epicSlug,
  onSelectNode,
  onSelectEdge,
  onClearSelection,
}: GraphViewProps) {
  const [flow, setFlow] = useState<ReactFlowInstance | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null)
  const [elkPositions, setElkPositions] =
    useState<Map<number, FlowPosition> | null>(null)
  const [layoutVersion, setLayoutVersion] = useState(0)

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

  const nodesById = useMemo(() => {
    const map = new Map<number, GraphNode>()
    for (const node of rootNodes) {
      map.set(node.id, node)
    }
    for (const group of childrenByParent.values()) {
      for (const node of group) {
        map.set(node.id, node)
      }
    }
    return map
  }, [childrenByParent, rootNodes])

  const graphNodes = useMemo(
    () => [...nodesById.values()].sort((a, b) => a.id - b.id),
    [nodesById],
  )

  const focusPositions = useMemo(() => {
    if (!focusMode || selectedNodeId === null) {
      return null
    }

    const path: GraphNode[] = []
    const visited = new Set<number>()
    let currentId: number | null = selectedNodeId

    while (currentId !== null && !visited.has(currentId)) {
      visited.add(currentId)
      const node: GraphNode | null = nodesById.get(currentId) ?? null
      if (!node) {
        break
      }
      path.push(node)
      currentId = node.parentNodeId ?? null
    }

    if (!path.length) {
      return null
    }

    path.reverse()
    const positions = new Map<number, FlowPosition>()
    const xOffset = 40
    const yOffset = 40
    const xStep = 360
    const yStep = 180

    for (let i = 0; i < path.length; i += 1) {
      const node = path[i]
      positions.set(node.id, {
        x: xOffset + i * xStep,
        y: yOffset + i * yStep,
      })
    }

    return positions
  }, [focusMode, nodesById, selectedNodeId])

  useEffect(() => {
    if (focusPositions || !graphNodes.length) {
      setElkPositions(null)
      return
    }

    let cancelled = false
    const yOffset = GRAPH_PADDING + TRUNK_HEIGHT + TRUNK_GAP
    ;(async () => {
      try {
        const positions = await layoutWithElk(graphNodes, childrenByParent, {
          nodeWidth: GRAPH_NODE_WIDTH,
          nodeHeight: GRAPH_NODE_HEIGHT,
          xOffset: GRAPH_PADDING,
          yOffset,
        })
        if (!cancelled) {
          setElkPositions(positions)
          setLayoutVersion((v) => v + 1)
        }
      } catch {
        if (!cancelled) {
          setElkPositions(null)
          setLayoutVersion((v) => v + 1)
        }
      }
    })()

    return () => {
      cancelled = true
    }
  }, [childrenByParent, focusPositions, graphNodes])

  const basePositions = useMemo(() => {
    if (focusPositions) {
      return focusPositions
    }
    if (elkPositions) {
      return elkPositions
    }
    return layoutTree(rootNodes, childrenByParent, {
      xSpacing: 360,
      ySpacing: 140,
      xOffset: GRAPH_PADDING,
      yOffset: GRAPH_PADDING + TRUNK_HEIGHT + TRUNK_GAP,
    })
  }, [childrenByParent, elkPositions, focusPositions, rootNodes])

  const hasSelection = selectedNodeId !== null || selectedEdgeId !== null
  const hasSelectionRef = useRef(hasSelection)

  useEffect(() => {
    hasSelectionRef.current = hasSelection
  }, [hasSelection])

  const selectionLens = useMemo(() => {
    if (selectedNodeId === null || focusPositions) {
      return null
    }
    return computeSelectionLens(selectedNodeId, nodesById, childrenByParent)
  }, [childrenByParent, focusPositions, nodesById, selectedNodeId])

  const targetPositions = useMemo(() => {
    if (!selectionLens) {
      return basePositions
    }
    return applySelectionLens(basePositions, selectionLens, childrenByParent)
  }, [basePositions, childrenByParent, selectionLens])

  const [positions, setPositions] =
    useState<Map<number, FlowPosition>>(targetPositions)
  const positionsRef = useRef(targetPositions)

  useEffect(() => {
    positionsRef.current = positions
  }, [positions])

  useEffect(() => {
    const from = positionsRef.current
    const to = targetPositions

    if (from === to) {
      return
    }

    if (!from.size || !to.size || GRAPH_LAYOUT_ANIMATION_MS <= 0) {
      setPositions(to)
      return
    }

    let frame: number | null = null
    const startedAt = performance.now()

    function easeOutCubic(t: number) {
      return 1 - Math.pow(1 - t, 3)
    }

    function step(now: number) {
      const elapsed = now - startedAt
      const t = Math.min(1, elapsed / GRAPH_LAYOUT_ANIMATION_MS)
      const eased = easeOutCubic(t)
      const next = new Map<number, FlowPosition>()

      for (const [nodeId, target] of to) {
        const start = from.get(nodeId) ?? target
        next.set(nodeId, {
          x: start.x + (target.x - start.x) * eased,
          y: start.y + (target.y - start.y) * eased,
        })
      }

      setPositions(next)
      if (t < 1) {
        frame = requestAnimationFrame(step)
      }
    }

    frame = requestAnimationFrame(step)
    return () => {
      if (frame !== null) {
        cancelAnimationFrame(frame)
      }
    }
  }, [targetPositions])

  const selectedEdgeNodeIds = useMemo(() => {
    if (!selectedEdgeId || !selectedEdgeId.startsWith("edge:")) {
      return null
    }
    const parts = selectedEdgeId.split(":")
    if (parts.length !== 3) {
      return null
    }
    const fromId = Number(parts[1])
    const toId = Number(parts[2])
    if (Number.isNaN(fromId) || Number.isNaN(toId)) {
      return null
    }
    return new Set([fromId, toId])
  }, [selectedEdgeId])

  const nodes = useMemo(() => {
    const mapped: GraphFlowNode[] = []
    const includeTrunk = !focusPositions && positions.size > 0
    if (includeTrunk) {
      let maxX = 0
      for (const pos of positions.values()) {
        maxX = Math.max(maxX, pos.x)
      }
      const trunkWidth = Math.max(
        GRAPH_NODE_WIDTH,
        maxX + GRAPH_NODE_WIDTH - GRAPH_PADDING,
      )

      mapped.push({
        id: TRUNK_NODE_ID,
        type: "trunk",
        position: { x: GRAPH_PADDING, y: GRAPH_PADDING },
        data: {},
        draggable: false,
        selectable: false,
        focusable: false,
        style: {
          width: trunkWidth,
          height: TRUNK_HEIGHT,
        },
        className: "pointer-events-none",
      } satisfies TrunkNodeType)
    }

    for (const [nodeId, pos] of positions) {
      const graphNode = nodesById.get(nodeId) ?? null
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
          edgeHighlighted: selectedEdgeNodeIds?.has(graphNode.id) ?? false,
          onSelectNode,
        },
        selectable: true,
        draggable: false,
        focusable: true,
        selected: selectedNodeId === graphNode.id,
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          width: GRAPH_NODE_WIDTH,
          height: GRAPH_NODE_HEIGHT,
        },
      } satisfies FlowBranchNodeType)
    }
    return mapped
  }, [
    agentsById,
    epicSlug,
    focusPositions,
    onSelectNode,
    nodesById,
    positions,
    selectedEdgeNodeIds,
    selectedNodeId,
    tasksById,
  ])

  const edges = useMemo(() => {
    const mapped: Edge[] = []
    if (!focusPositions) {
      for (const root of rootNodes) {
        mapped.push({
          id: `trunk:${root.id}`,
          source: TRUNK_NODE_ID,
          target: String(root.id),
          type: "smoothstep",
          selectable: false,
          focusable: false,
          interactionWidth: 0,
          style: {
            stroke: "var(--border)",
            strokeOpacity: 0.35,
            strokeWidth: 1.25,
            transition: `stroke ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-opacity ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease`,
          },
        })
      }
    }

    const visibleNodeIds = new Set(nodes.map((node) => node.id))
    for (const node of nodes) {
      if (node.type !== "branch") {
        continue
      }
      const graphNode = node.data.node
      if (!graphNode) {
        continue
      }
      if (graphNode.parentNodeId === null) {
        continue
      }
      if (!visibleNodeIds.has(String(graphNode.parentNodeId))) {
        continue
      }
      const edgeId = makeEdgeId(graphNode.parentNodeId, graphNode.id)
      const isSelected = selectedEdgeId === edgeId
      const isHovered = hoveredEdgeId === edgeId
      mapped.push({
        id: edgeId,
        source: String(graphNode.parentNodeId),
        target: String(graphNode.id),
        type: "commitString",
        selectable: true,
        focusable: true,
        selected: isSelected,
        interactionWidth: 24,
        className: "cursor-pointer",
        style: {
          stroke: isSelected || isHovered ? "var(--ring)" : "var(--border)",
          strokeWidth: isSelected ? 2.5 : isHovered ? 2 : 1.25,
          strokeOpacity: isSelected ? 1 : isHovered ? 0.75 : 0.45,
          transition: `stroke ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-width ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-opacity ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease`,
        },
      })
    }
    return mapped
  }, [focusPositions, hoveredEdgeId, nodes, rootNodes, selectedEdgeId])

  useEffect(() => {
    if (!flow || selectedNodeId === null || focusPositions) {
      return
    }

    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect || rect.width <= 0 || rect.height <= 0) {
      return
    }

    const branchNodeIds = selectionLens?.focusPath ?? [selectedNodeId]
    let minX = Number.POSITIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY

    for (const nodeId of branchNodeIds) {
      const pos = targetPositions.get(nodeId)
      if (!pos) {
        continue
      }
      minX = Math.min(minX, pos.x)
      minY = Math.min(minY, pos.y)
      maxX = Math.max(maxX, pos.x + GRAPH_NODE_WIDTH)
      maxY = Math.max(maxY, pos.y + GRAPH_NODE_HEIGHT)
    }

    if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
      return
    }

    const padding = GRAPH_FIT_PADDING_PX
    const visibleWidth = Math.max(
      1,
      rect.width - DETAILS_PANEL_WIDTH_PX - padding * 2,
    )
    const visibleHeight = Math.max(1, rect.height - padding * 2)
    const boundsWidth = Math.max(1, maxX - minX)
    const boundsHeight = Math.max(1, maxY - minY)

    const fitZoom = Math.min(
      visibleWidth / boundsWidth,
      visibleHeight / boundsHeight,
    )
    const zoom = Math.min(
      GRAPH_FIT_MAX_ZOOM,
      Math.max(GRAPH_FIT_MIN_ZOOM, fitZoom),
    )

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const visibleCenterX = (rect.width - DETAILS_PANEL_WIDTH_PX) / 2
    const visibleCenterY = rect.height / 2

    const x = visibleCenterX - centerX * zoom
    const desiredTrunkScreenY = GRAPH_PADDING
    const trunkWorldY = GRAPH_PADDING
    const trunkAnchorY = desiredTrunkScreenY - trunkWorldY * zoom

    const yMin = padding - minY * zoom
    const yMax = rect.height - padding - maxY * zoom
    const y =
      yMin <= yMax
        ? Math.min(yMax, Math.max(yMin, trunkAnchorY))
        : visibleCenterY - centerY * zoom

    flow.setViewport({ x, y, zoom }, { duration: GRAPH_LAYOUT_ANIMATION_MS })
  }, [flow, focusPositions, selectedNodeId, selectionLens, targetPositions])

  useEffect(() => {
    if (!flow || !nodes.length) {
      return
    }

    if (hasSelectionRef.current) {
      return
    }

    flow.fitView({ padding: 0.2, duration: 200 })
  }, [flow, layoutVersion, nodes.length])

  return (
    <main className="relative min-w-0 flex-1 overflow-hidden">
      <DotGrid />
      {rootNodes.length > 0 ? (
        <div ref={containerRef} className="relative h-full">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={{ branch: FlowBranchNode, trunk: TrunkNode }}
            edgeTypes={{ commitString: CommitStringEdge }}
            nodesDraggable={false}
            nodesConnectable={false}
            onInit={setFlow}
            onPaneClick={onClearSelection}
            onEdgeClick={(e, edge) => {
              if (!edge.id.startsWith("edge:")) {
                return
              }
              e.stopPropagation()
              const fromId = Number(edge.source)
              const toId = Number(edge.target)
              if (Number.isNaN(fromId) || Number.isNaN(toId)) {
                return
              }
              onSelectEdge(fromId, toId)
            }}
            onEdgeMouseEnter={(_, edge) => {
              if (edge.id.startsWith("edge:")) {
                setHoveredEdgeId(edge.id)
              }
            }}
            onEdgeMouseLeave={(_, edge) => {
              if (!edge.id.startsWith("edge:")) {
                return
              }
              setHoveredEdgeId((current) =>
                current === edge.id ? null : current,
              )
            }}
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
