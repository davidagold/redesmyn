import { DotGrid } from "@/components/ui/dot-grid"
import { Button } from "@/components/ui/button"
import {
  Position,
  ReactFlow,
  type DefaultEdgeOptions,
  type Edge,
  type ReactFlowInstance,
} from "@xyflow/react"
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from "react"
import type {
  Agent,
  GraphNode,
  MergeRun,
  Task,
  TrunkTimeline,
} from "@/lib/graph-utils"
import { makeEdgeId } from "@/lib/graph-utils"
import { copyToClipboard } from "@/lib/clipboard"
import { FlowBranchNode, type FlowBranchNodeType } from "./FlowBranchNode"
import { CommitStringEdge } from "./CommitStringEdge"
import { TrunkNode, type TrunkNodeType } from "./TrunkNode"
import { RoundedSmoothStepEdge } from "./RoundedSmoothStepEdge"
import {
  pulseDurationMs,
  type EdgePulseData,
  type EdgePulseKind,
} from "./edgePulse"
import {
  DETAILS_PANEL_WIDTH_PX,
  GRAPH_EDGE_STYLE_ANIMATION_MS,
  GRAPH_LAYOUT_ANIMATION_MS,
  GRAPH_SELECTION_ANIMATION_MS,
  GRAPH_FIT_MAX_ZOOM,
  GRAPH_FIT_MIN_ZOOM,
  GRAPH_FIT_PADDING_PX,
  GRAPH_NODE_HEIGHT,
  GRAPH_NODE_WIDTH,
  GRAPH_PADDING,
  TRUNK_GAP,
  TRUNK_COMMIT_PADDING,
  TRUNK_COMMIT_ROW_HEIGHT,
  TRUNK_COMMIT_SPACING,
  TRUNK_EDGE_NUDGE_PX,
  TRUNK_LABEL_COLUMN,
  TRUNK_THICKNESS,
} from "./graphConfig"
import { layoutWithElk } from "./elkLayout"
import { type FlowPosition, layoutTree } from "./flowLayout"
import { applySelectionLens, computeSelectionLens } from "./selectionLens"
import type { NodeActivity } from "@/lib/presence"

interface GraphViewProps {
  rootNodes: GraphNode[]
  childrenByParent: Map<number | null, GraphNode[]>
  tasksById: Map<number, Task>
  agentsById: Map<number, Agent>
  mergeRunsByTaskId: Map<number, MergeRun>
  activityByNodeId: Map<number, NodeActivity>
  harnessCommand: string
  detach: boolean
  trunk?: TrunkTimeline | null
  selectedNodeIds: ReadonlySet<number>
  selectedNodeId: number | null
  selectedEdgeId: string | null
  focusMode: boolean
  mergeStepCue?: {
    id: string
    nodeId: number
    kind: "rebase" | "merge_ff"
  } | null
  epicSlug?: string | null
  onSelectNode: (nodeId: number, options: { additive: boolean }) => void
  onSelectEdge: (fromNodeId: number, toNodeId: number) => void
  onClearSelection: () => void
  onRequestRefresh?: () => void
}

const TRUNK_NODE_ID = "trunk"

type GraphFlowNode = FlowBranchNodeType | TrunkNodeType

const POSITION_EPSILON_PX = 0.25
const VIEWPORT_EPSILON_PX = 0.5
const VIEWPORT_EPSILON_ZOOM = 0.001

type TrunkMark = {
  type: "commit" | "base" | "ellipsis"
  sha?: string
  authorName?: string | null
  authorEmail?: string | null
  authoredAt?: string | null
}

type TrunkLayout = {
  x: number
  y: number
  width: number
  height: number
  marks: TrunkMark[]
  baseOffset: number
  commitSpacing: number
  commitPadding: number
}

interface Viewport {
  x: number
  y: number
  zoom: number
}

interface ViewportAnimation {
  id: number
  to: Viewport
  duration: number
}

interface GraphBounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
}

function positionsMatch(
  from: Map<number, FlowPosition>,
  to: Map<number, FlowPosition>,
) {
  if (from === to) {
    return true
  }
  if (from.size !== to.size) {
    return false
  }
  for (const [nodeId, pos] of from) {
    const next = to.get(nodeId)
    if (!next) {
      return false
    }
    if (
      Math.abs(pos.x - next.x) > POSITION_EPSILON_PX ||
      Math.abs(pos.y - next.y) > POSITION_EPSILON_PX
    ) {
      return false
    }
  }
  return true
}

function viewportMatches(current: Viewport, next: Viewport) {
  return (
    Math.abs(current.x - next.x) <= VIEWPORT_EPSILON_PX &&
    Math.abs(current.y - next.y) <= VIEWPORT_EPSILON_PX &&
    Math.abs(current.zoom - next.zoom) <= VIEWPORT_EPSILON_ZOOM
  )
}

function computeGraphBounds(
  positions: Map<number, FlowPosition>,
  trunkLayout: TrunkLayout | null,
) {
  if (!positions.size && !trunkLayout) {
    return null
  }
  let minX = Number.POSITIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY

  for (const pos of positions.values()) {
    minX = Math.min(minX, pos.x)
    minY = Math.min(minY, pos.y)
    maxX = Math.max(maxX, pos.x + GRAPH_NODE_WIDTH)
    maxY = Math.max(maxY, pos.y + GRAPH_NODE_HEIGHT)
  }

  if (trunkLayout) {
    minX = trunkLayout.x
    minY = trunkLayout.y
    maxX = Math.max(maxX, trunkLayout.x + trunkLayout.width)
    maxY = Math.max(maxY, trunkLayout.y + trunkLayout.height)
  }

  return { minX, minY, maxX, maxY }
}

function computeFitViewport(
  bounds: GraphBounds,
  rect: DOMRect,
  reserveWidth: number,
) {
  const padding = GRAPH_FIT_PADDING_PX
  const visibleWidth = Math.max(1, rect.width - reserveWidth - padding * 2)
  const visibleHeight = Math.max(1, rect.height - padding * 2)
  const boundsWidth = Math.max(1, bounds.maxX - bounds.minX)
  const boundsHeight = Math.max(1, bounds.maxY - bounds.minY)

  const fitZoom = Math.min(
    visibleWidth / boundsWidth,
    visibleHeight / boundsHeight,
  )
  const zoom = Math.min(
    GRAPH_FIT_MAX_ZOOM,
    Math.max(GRAPH_FIT_MIN_ZOOM, fitZoom),
  )

  const centerX = (bounds.minX + bounds.maxX) / 2
  const centerY = (bounds.minY + bounds.maxY) / 2
  const visibleCenterX = (rect.width - reserveWidth) / 2
  const visibleCenterY = rect.height / 2

  return {
    x: visibleCenterX - centerX * zoom,
    y: visibleCenterY - centerY * zoom,
    zoom,
  }
}

export function GraphView({
  rootNodes,
  childrenByParent,
  tasksById,
  agentsById,
  mergeRunsByTaskId,
  activityByNodeId,
  harnessCommand,
  detach,
  trunk,
  selectedNodeIds,
  selectedNodeId,
  selectedEdgeId,
  focusMode,
  mergeStepCue = null,
  epicSlug,
  onSelectNode,
  onSelectEdge,
  onClearSelection,
  onRequestRefresh,
}: GraphViewProps) {
  const [flow, setFlow] = useState<ReactFlowInstance | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [hoveredEdgeId, setHoveredEdgeId] = useState<string | null>(null)
  const [elkPositions, setElkPositions] =
    useState<Map<number, FlowPosition> | null>(null)
  const [layoutVersion, setLayoutVersion] = useState(0)
  const didInitialFitRef = useRef(false)
  const viewportAnimationIdRef = useRef(0)
  const [viewportAnimation, setViewportAnimation] =
    useState<ViewportAnimation | null>(null)
  const [selectionNotice, setSelectionNotice] = useState<string | null>(null)
  const [selectionBarMounted, setSelectionBarMounted] = useState(false)
  const [selectionBarVisible, setSelectionBarVisible] = useState(false)
  const selectionBarHideTimerRef = useRef<number | null>(null)

  const [edgePulseById, setEdgePulseById] =
    useState<Record<string, EdgePulseData>>({})
  const edgePulseTimeoutsRef = useRef<Map<string, number>>(new Map())

  const triggerEdgePulse = useCallback(
    (edgeId: string, kind: EdgePulseKind, delayMs = 0) => {
      window.setTimeout(() => {
        setEdgePulseById((prev) => {
          const previous = prev[edgeId]
          const token = previous ? previous.token + 1 : 1
          return { ...prev, [edgeId]: { kind, token } }
        })

        const existingTimeout = edgePulseTimeoutsRef.current.get(edgeId)
        if (existingTimeout !== undefined) {
          window.clearTimeout(existingTimeout)
        }
        const timeoutId = window.setTimeout(() => {
          edgePulseTimeoutsRef.current.delete(edgeId)
          setEdgePulseById((prev) => {
            if (!(edgeId in prev)) {
              return prev
            }
            const next = { ...prev }
            delete next[edgeId]
            return next
          })
        }, pulseDurationMs(kind) + 200)
        edgePulseTimeoutsRef.current.set(edgeId, timeoutId)
      }, delayMs)
    },
    [],
  )

  const defaultEdgeOptions: DefaultEdgeOptions = useMemo(
    () => ({
      type: "roundedSmoothStep",
      style: {
        stroke: "var(--border)",
        strokeWidth: 1.25,
      },
    }),
    [],
  )

  useEffect(() => {
    if (!selectionNotice) {
      return
    }
    const id = window.setTimeout(() => setSelectionNotice(null), 2_000)
    return () => window.clearTimeout(id)
  }, [selectionNotice])

  useEffect(() => {
    const edgePulseTimeouts = edgePulseTimeoutsRef.current
    return () => {
      if (selectionBarHideTimerRef.current !== null) {
        window.clearTimeout(selectionBarHideTimerRef.current)
      }
      for (const timeoutId of edgePulseTimeouts.values()) {
        window.clearTimeout(timeoutId)
      }
      edgePulseTimeouts.clear()
    }
  }, [])

  const queueViewportAnimation = useCallback(
    (nextViewport: Viewport, duration: number) => {
      if (!flow) {
        return
      }
      const currentViewport = flow.getViewport()
      if (viewportMatches(currentViewport, nextViewport)) {
        return
      }
      setViewportAnimation((current) => {
        if (
          current &&
          current.duration === duration &&
          viewportMatches(current.to, nextViewport)
        ) {
          return current
        }
        viewportAnimationIdRef.current += 1
        return {
          id: viewportAnimationIdRef.current,
          to: nextViewport,
          duration,
        }
      })
    },
    [flow],
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

  useEffect(() => {
    if (!mergeStepCue) {
      return
    }
    const node = nodesById.get(mergeStepCue.nodeId)
    if (!node) {
      return
    }

    const edgeId =
      node.parentNodeId === null
        ? `trunk:${node.id}`
        : makeEdgeId(node.parentNodeId, node.id)

    triggerEdgePulse(
      edgeId,
      mergeStepCue.kind === "merge_ff" ? "merge" : "rebase",
    )
  }, [mergeStepCue, nodesById, triggerEdgePulse])

  const selectedRunningTaskIds = useMemo(() => {
    const taskIds = new Set<number>()
    for (const selectedNodeId of selectedNodeIds) {
      const node = nodesById.get(selectedNodeId) ?? null
      if (!node || node.primaryTaskId === null) {
        continue
      }
      const task = tasksById.get(node.primaryTaskId) ?? null
      if (!task || task.state === "blocked" || task.state === "done") {
        continue
      }

      const agent =
        node.agentId !== null ? (agentsById.get(node.agentId) ?? null) : null
      const status = agent?.status ?? null

      if (status === "running" || status === "blocked") {
        taskIds.add(task.id)
      }
    }

    return [...taskIds].sort((a, b) => a - b)
  }, [agentsById, nodesById, selectedNodeIds, tasksById])

  const trunkMarks = useMemo(() => {
    if (!trunk || !trunk.baseSha) {
      return null
    }
    const marks: TrunkMark[] = []
    const commitsAfter = trunk.commitsAfter ?? []
    const commitsBefore = trunk.commitsBefore ?? []
    const baseCommit = trunk.baseCommit ?? { sha: trunk.baseSha }

    if (trunk.hasMoreAfter) {
      marks.push({ type: "ellipsis" })
    }
    for (const commit of [...commitsAfter].reverse()) {
      marks.push({
        type: "commit",
        sha: commit.sha,
        authorName: commit.authorName ?? null,
        authorEmail: commit.authorEmail ?? null,
        authoredAt: commit.authoredAt ?? null,
      })
    }
    marks.push({
      type: "base",
      sha: baseCommit.sha,
      authorName: baseCommit.authorName ?? null,
      authorEmail: baseCommit.authorEmail ?? null,
      authoredAt: baseCommit.authoredAt ?? null,
    })
    for (const commit of commitsBefore) {
      marks.push({
        type: "commit",
        sha: commit.sha,
        authorName: commit.authorName ?? null,
        authorEmail: commit.authorEmail ?? null,
        authoredAt: commit.authoredAt ?? null,
      })
    }
    if (trunk.hasMoreBefore) {
      marks.push({ type: "ellipsis" })
    }
    const baseIndex = marks.findIndex((mark) => mark.type === "base")
    return {
      marks,
      baseIndex: baseIndex >= 0 ? baseIndex : Math.floor(marks.length / 2),
    }
  }, [trunk])

  const trunkColumnWidth = useMemo(
    () => TRUNK_THICKNESS + TRUNK_LABEL_COLUMN,
    [],
  )

  const trunkMetrics = useMemo(() => {
    const marks = trunkMarks?.marks ?? []
    const markCount = marks.length
    const commitSpacing = TRUNK_COMMIT_SPACING
    const commitPadding = TRUNK_COMMIT_PADDING
    const rowHeight = TRUNK_COMMIT_ROW_HEIGHT
    const rowCenterOffset = rowHeight / 2
    const spanHeight =
      markCount > 1
        ? commitPadding * 2 + (markCount - 1) * commitSpacing + rowHeight
        : commitPadding * 2 + rowHeight
    const baseIndex = trunkMarks?.baseIndex ?? 0
    const edgeOffset = baseIndex > 0 ? commitSpacing / 2 : 0
    const baseOffset =
      markCount > 0
        ? commitPadding +
          baseIndex * commitSpacing +
          rowCenterOffset -
          edgeOffset -
          TRUNK_EDGE_NUDGE_PX
        : commitPadding + rowCenterOffset

    return {
      marks,
      commitSpacing,
      commitPadding,
      rowHeight,
      spanHeight,
      baseOffset,
    }
  }, [trunkMarks])

  const layoutAnchorY = useMemo(() => {
    const offset = trunkMetrics.baseOffset
    const desired = GRAPH_PADDING + offset - GRAPH_NODE_HEIGHT / 2
    return Math.max(GRAPH_PADDING, desired)
  }, [trunkMetrics.baseOffset])

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
    const yOffset = layoutAnchorY
    const xOffset = GRAPH_PADDING + trunkColumnWidth + TRUNK_GAP
    ;(async () => {
      try {
        const positions = await layoutWithElk(graphNodes, childrenByParent, {
          nodeWidth: GRAPH_NODE_WIDTH,
          nodeHeight: GRAPH_NODE_HEIGHT,
          xOffset,
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
  }, [
    childrenByParent,
    focusPositions,
    graphNodes,
    layoutAnchorY,
    trunkColumnWidth,
  ])

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
      xOffset: GRAPH_PADDING + trunkColumnWidth + TRUNK_GAP,
      yOffset: layoutAnchorY,
    })
  }, [
    childrenByParent,
    elkPositions,
    focusPositions,
    rootNodes,
    layoutAnchorY,
    trunkColumnWidth,
  ])

  const hasSelection = selectedNodeIds.size > 0 || selectedEdgeId !== null
  const hasSelectionRef = useRef(hasSelection)
  const previousHasSelectionRef = useRef(hasSelection)

  useEffect(() => {
    hasSelectionRef.current = hasSelection
  }, [hasSelection])

  useEffect(() => {
    didInitialFitRef.current = false
  }, [epicSlug])

  const selectionLensRef = useRef<{
    key: string
    lens: ReturnType<typeof computeSelectionLens>
    nodesById: Map<number, GraphNode>
    childrenByParent: Map<number | null, GraphNode[]>
  } | null>(null)

  const rawSelectionLens = useMemo(() => {
    if (selectedNodeId === null || focusPositions) {
      return null
    }
    return computeSelectionLens(selectedNodeId, nodesById, childrenByParent)
  }, [childrenByParent, focusPositions, nodesById, selectedNodeId])

  const selectionLens = useMemo(() => {
    if (!rawSelectionLens) {
      selectionLensRef.current = null
      return null
    }
    const key = `${rawSelectionLens.focusRootId}:${rawSelectionLens.focusEndId}`
    const cached = selectionLensRef.current
    if (
      cached &&
      cached.key === key &&
      cached.nodesById === nodesById &&
      cached.childrenByParent === childrenByParent
    ) {
      return cached.lens
    }
    selectionLensRef.current = {
      key,
      lens: rawSelectionLens,
      nodesById,
      childrenByParent,
    }
    return rawSelectionLens
  }, [childrenByParent, nodesById, rawSelectionLens])

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
    const positionsAreSame = positionsMatch(from, to)
    const canAnimatePositions =
      !positionsAreSame &&
      from.size > 0 &&
      to.size > 0 &&
      GRAPH_LAYOUT_ANIMATION_MS > 0
    const shouldSetPositionsImmediately =
      !positionsAreSame && !canAnimatePositions
    const canAnimateViewport = Boolean(flow && viewportAnimation)

    if (!canAnimatePositions && !canAnimateViewport) {
      if (shouldSetPositionsImmediately) {
        setPositions(to)
      }
      return
    }

    if (shouldSetPositionsImmediately) {
      setPositions(to)
    }

    const viewportRequest = viewportAnimation
    const fromViewport = canAnimateViewport && flow ? flow.getViewport() : null
    const toViewport = canAnimateViewport ? (viewportRequest?.to ?? null) : null
    const viewportDuration =
      canAnimatePositions && viewportRequest
        ? GRAPH_LAYOUT_ANIMATION_MS
        : (viewportRequest?.duration ?? 0)
    const requestId = viewportRequest?.id

    let frame: number | null = null
    const startedAt = performance.now()

    function easeOutCubic(t: number) {
      return 1 - Math.pow(1 - t, 3)
    }

    function step(now: number) {
      const elapsed = now - startedAt
      let positionsDone = true
      let viewportDone = true

      if (canAnimatePositions) {
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
        positionsDone = t >= 1
      }

      if (canAnimateViewport && fromViewport && toViewport && flow) {
        const t =
          viewportDuration > 0 ? Math.min(1, elapsed / viewportDuration) : 1
        const eased = easeOutCubic(t)
        flow.setViewport(
          {
            x: fromViewport.x + (toViewport.x - fromViewport.x) * eased,
            y: fromViewport.y + (toViewport.y - fromViewport.y) * eased,
            zoom:
              fromViewport.zoom + (toViewport.zoom - fromViewport.zoom) * eased,
          },
          { duration: 0 },
        )
        viewportDone = t >= 1
      }

      if (!positionsDone || !viewportDone) {
        frame = requestAnimationFrame(step)
        return
      }

      if (canAnimateViewport && requestId !== undefined) {
        setViewportAnimation((current) =>
          current?.id === requestId ? null : current,
        )
      }
      if (canAnimatePositions) {
        setPositions(to)
      }
    }

    frame = requestAnimationFrame(step)
    return () => {
      if (frame !== null) {
        cancelAnimationFrame(frame)
      }
    }
  }, [flow, targetPositions, viewportAnimation])

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

  const trunkLayout = useMemo<TrunkLayout | null>(() => {
    if (focusPositions || basePositions.size === 0) {
      return null
    }
    let maxY = GRAPH_PADDING
    for (const pos of basePositions.values()) {
      maxY = Math.max(maxY, pos.y)
    }
    const graphBottom = maxY + GRAPH_NODE_HEIGHT
    const trunkHeight = Math.max(
      trunkMetrics.spanHeight,
      graphBottom - GRAPH_PADDING,
    )

    return {
      x: GRAPH_PADDING,
      y: GRAPH_PADDING,
      width: trunkColumnWidth,
      height: trunkHeight,
      marks: trunkMetrics.marks,
      baseOffset: trunkMetrics.baseOffset,
      commitSpacing: trunkMetrics.commitSpacing,
      commitPadding: trunkMetrics.commitPadding,
    }
  }, [basePositions, focusPositions, trunkColumnWidth, trunkMetrics])

  const nodes = useMemo(() => {
    const mapped: GraphFlowNode[] = []
    const includeTrunk = !focusPositions && positions.size > 0 && trunkLayout
    if (includeTrunk && trunkLayout) {
      mapped.push({
        id: TRUNK_NODE_ID,
        type: "trunk",
        position: { x: trunkLayout.x, y: trunkLayout.y },
        data: {
          marks: trunkLayout.marks,
          baseOffset: trunkLayout.baseOffset,
          commitSpacing: trunkLayout.commitSpacing,
          commitPadding: trunkLayout.commitPadding,
          lineWidth: TRUNK_THICKNESS,
          labelOffset: TRUNK_THICKNESS + 12,
        },
        draggable: false,
        selectable: false,
        focusable: false,
        sourcePosition: Position.Right,
        style: {
          width: trunkLayout.width,
          height: trunkLayout.height,
          // React Flow disables pointer events on inert nodes; keep these on so trunk tooltips work.
          pointerEvents: "all",
        },
        className: "select-none",
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
      const mergeRun = task ? mergeRunsByTaskId.get(task.id) : undefined
      const activity = activityByNodeId.get(graphNode.id)

      mapped.push({
        id: String(graphNode.id),
        type: "branch",
        position: pos,
        data: {
          node: graphNode,
          task,
          agent,
          mergeRun,
          activity,
          epicSlug,
          harnessCommand,
          detach,
          edgeHighlighted: selectedEdgeNodeIds?.has(graphNode.id) ?? false,
          onSelectNode,
          onRequestRefresh,
        },
        selectable: true,
        draggable: false,
        focusable: true,
        selected: selectedNodeIds.has(graphNode.id),
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
    activityByNodeId,
    agentsById,
    detach,
    epicSlug,
    focusPositions,
    harnessCommand,
    mergeRunsByTaskId,
    onSelectNode,
    onRequestRefresh,
    nodesById,
    positions,
    selectedEdgeNodeIds,
    selectedNodeIds,
    tasksById,
    trunkLayout,
  ])

  const edges = useMemo(() => {
    const mapped: Edge[] = []
    if (!focusPositions) {
      for (const root of rootNodes) {
        const edgeId = `trunk:${root.id}`
        const pulse = edgePulseById[edgeId] ?? null
        mapped.push({
          id: edgeId,
          source: TRUNK_NODE_ID,
          sourceHandle: "base",
          target: String(root.id),
          type: "roundedSmoothStep",
          selectable: false,
          focusable: false,
          interactionWidth: 0,
          data: pulse ? { pulse } : undefined,
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
      const pulse = edgePulseById[edgeId] ?? null
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
        data: pulse ? { pulse } : undefined,
        style: {
          stroke: isSelected || isHovered ? "var(--ring)" : "var(--border)",
          strokeWidth: isSelected ? 2.5 : isHovered ? 2 : 1.25,
          strokeOpacity: isSelected ? 1 : isHovered ? 0.75 : 0.45,
          transition: `stroke ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-width ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease, stroke-opacity ${GRAPH_EDGE_STYLE_ANIMATION_MS}ms ease`,
        },
      })
    }
    return mapped
  }, [
    edgePulseById,
    focusPositions,
    hoveredEdgeId,
    nodes,
    rootNodes,
    selectedEdgeId,
  ])

  useLayoutEffect(() => {
    const wasSelected = previousHasSelectionRef.current
    previousHasSelectionRef.current = hasSelection

    if (!flow || nodes.length === 0) {
      return
    }

    if (wasSelected && !hasSelection) {
      const rect = containerRef.current?.getBoundingClientRect()
      if (!rect || rect.width <= 0 || rect.height <= 0) {
        return
      }
      const bounds = computeGraphBounds(targetPositions, trunkLayout)
      if (!bounds) {
        return
      }
      const nextViewport = computeFitViewport(bounds, rect, 0)
      queueViewportAnimation(nextViewport, GRAPH_LAYOUT_ANIMATION_MS)
    }
  }, [
    flow,
    hasSelection,
    nodes.length,
    queueViewportAnimation,
    targetPositions,
    trunkLayout,
  ])

  useLayoutEffect(() => {
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

    const desiredTrunkScreenX = GRAPH_PADDING
    const trunkWorldX = trunkLayout?.x ?? GRAPH_PADDING
    const trunkAnchorX = desiredTrunkScreenX - trunkWorldX * zoom

    const xMin = padding - minX * zoom
    const xMax = rect.width - DETAILS_PANEL_WIDTH_PX - padding - maxX * zoom
    const x =
      xMin <= xMax
        ? Math.min(xMax, Math.max(xMin, trunkAnchorX))
        : visibleCenterX - centerX * zoom

    const yMin = padding - minY * zoom
    const yMax = rect.height - padding - maxY * zoom
    const y =
      yMin <= yMax
        ? Math.min(yMax, Math.max(yMin, visibleCenterY - centerY * zoom))
        : visibleCenterY - centerY * zoom

    const nextViewport = { x, y, zoom }
    queueViewportAnimation(nextViewport, GRAPH_SELECTION_ANIMATION_MS)
  }, [
    flow,
    focusPositions,
    selectedNodeId,
    selectionLens,
    queueViewportAnimation,
    targetPositions,
    trunkLayout,
  ])

  useEffect(() => {
    if (!flow || !nodes.length) {
      return
    }

    if (didInitialFitRef.current || hasSelectionRef.current) {
      return
    }

    flow.fitView({ padding: 0.2, duration: 200 })
    didInitialFitRef.current = true
  }, [flow, epicSlug, layoutVersion, nodes.length])

  const selectionBarTargetVisible = selectedNodeIds.size > 1

  useEffect(() => {
    if (selectionBarTargetVisible) {
      if (selectionBarHideTimerRef.current !== null) {
        window.clearTimeout(selectionBarHideTimerRef.current)
        selectionBarHideTimerRef.current = null
      }
      setSelectionBarMounted(true)
      window.requestAnimationFrame(() => setSelectionBarVisible(true))
      return
    }

    setSelectionBarVisible(false)
    if (selectionBarHideTimerRef.current !== null) {
      window.clearTimeout(selectionBarHideTimerRef.current)
    }
    selectionBarHideTimerRef.current = window.setTimeout(() => {
      setSelectionBarMounted(false)
      selectionBarHideTimerRef.current = null
    }, 200)
  }, [selectionBarTargetVisible])

  return (
    <main className="relative min-w-0 flex-1 overflow-hidden">
      <DotGrid />
      <div ref={containerRef} className="relative h-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={{ branch: FlowBranchNode, trunk: TrunkNode }}
          edgeTypes={{
            commitString: CommitStringEdge,
            roundedSmoothStep: RoundedSmoothStepEdge,
          }}
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
        {nodes.length === 0 ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-6 text-sm text-muted-foreground">
            No nodes.
          </div>
        ) : null}
        {selectionBarMounted ? (
          <div className="pointer-events-none absolute inset-x-0 top-4 z-20 flex justify-center px-4">
            <div
              className={
                "pointer-events-auto flex items-center gap-3 rounded-lg bg-background/80 px-3 py-2 shadow-sm ring-1 ring-foreground/10 backdrop-blur transition-all duration-200 will-change-transform " +
                (selectionBarVisible
                  ? "translate-y-0 opacity-100"
                  : "-translate-y-12 opacity-0")
              }
              onClick={(e) => e.stopPropagation()}
            >
              <div className="text-xs text-muted-foreground">
                {selectedNodeIds.size} selected
              </div>
              <div className="flex overflow-hidden rounded-md border border-border/60">
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-none border-0"
                  disabledReason={
                    selectedRunningTaskIds.length === 0
                      ? "No running selected agents"
                      : null
                  }
                  title={
                    selectedRunningTaskIds.length
                      ? "Copy a tmux command to attach (sequentially) to running selected agents"
                      : "No running selected agents"
                  }
                  onClick={() => {
                    const sessions = selectedRunningTaskIds.map(
                      (taskId) => `rn-a-${taskId}`,
                    )
                    const cmd =
                      `for s in ${sessions.map((s) => `'${s}'`).join(" ")}; do ` +
                      `tmux has-session -t "$s" 2>/dev/null && tmux attach -t "$s"; ` +
                      "done"
                    void copyToClipboard(cmd)
                      .then(() => setSelectionNotice("Attach command copied"))
                      .catch((e: unknown) =>
                        setSelectionNotice(
                          e instanceof Error ? e.message : String(e),
                        ),
                      )
                  }}
                >
                  Copy attach
                </Button>
              </div>
              <Button variant="ghost" size="sm" onClick={onClearSelection}>
                Clear
              </Button>
              {selectionNotice ? (
                <div
                  className="max-w-[24rem] truncate text-xs text-destructive"
                  title={selectionNotice}
                >
                  {selectionNotice}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </main>
  )
}
