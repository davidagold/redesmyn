export const GRAPH_LAYOUT_ANIMATION_MS = 260
export const GRAPH_EDGE_STYLE_ANIMATION_MS = 150
export const GRAPH_EDGE_BORDER_RADIUS = 29

export const GRAPH_NODE_WIDTH = 320
export const GRAPH_NODE_HEIGHT = 96
export const GRAPH_PADDING = 40

// Task cards can render "chrome" outside the base card (badges above, action
// buttons, callouts below). The graph layout treats these as additional spacing
// so they don't collide with neighboring nodes.
export const GRAPH_NODE_VERTICAL_GAP = 160
export const GRAPH_NODE_HORIZONTAL_GAP = 140

export const TRUNK_THICKNESS = 2
export const TRUNK_GAP = 56
export const TRUNK_COMMIT_PADDING = 18
export const TRUNK_COMMIT_SPACING = 30
export const TRUNK_COMMIT_ROW_HEIGHT = 16
export const TRUNK_COMMIT_TITLE_COLUMN = 200
export const TRUNK_COMMIT_MARK_COLUMN = 28
export const TRUNK_COMMIT_SHA_COLUMN = 110

export const DETAILS_PANEL_WIDTH_PX = 32 * 16
export const GRAPH_FIT_PADDING_PX = 72
export const GRAPH_FIT_MIN_ZOOM = 0.55
export const GRAPH_FIT_MAX_ZOOM = 1.2

/**
 * Commit-string LOD is tied to zoom to keep large graphs responsive.
 *
 * Bands:
 * - `zoom < EDGE_LABEL_ZOOM`: simple edges (no labels, no ticks).
 * - `EDGE_LABEL_ZOOM <= zoom < EDGE_TICKS_ZOOM`: show commit-count label only.
 * - `zoom >= EDGE_TICKS_ZOOM`: show label + tick/dash affordance.
 *
 * We deliberately select bands (not raw zoom) in edge components so panning/zooming
 * doesn't re-render every edge on every fractional zoom change.
 */
export const EDGE_LABEL_ZOOM = 0.8
export const EDGE_TICKS_ZOOM = 1.05

/**
 * v0 perf budget (qualitative, but explicit):
 * - No visible jank for ~100 nodes / ~100 edges on a modern laptop.
 * - LOD must cap per-edge glyph density so long commit ranges don't explode DOM cost.
 */
export const GRAPH_PERF_TARGET_NODES = 100
export const GRAPH_PERF_TARGET_EDGES = 100

export type EdgeLodBand = 0 | 1 | 2

export function edgeLodBand(zoom: number): EdgeLodBand {
  if (zoom < EDGE_LABEL_ZOOM) {
    return 0
  }
  if (zoom < EDGE_TICKS_ZOOM) {
    return 1
  }
  return 2
}
