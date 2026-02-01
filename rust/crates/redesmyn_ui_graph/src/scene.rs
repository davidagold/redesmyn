use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use gpui::SharedString;
use redesmyn_graph_layout::{ForestLayoutEngine, LayoutConfig, LayoutNode, LayoutOptions};
use redesmyn_ids::{SessionEventId, SessionId, TaskId};
use redesmyn_protocol::client::{CommandState, MergeReadiness, TaskState};

pub(crate) const COLLAPSED_TASK_NODE_SIZE: redesmyn_graph_layout::Size =
    redesmyn_graph_layout::Size {
        width: 320,
        height: 112,
    };

pub(crate) const EXPANDED_TASK_NODE_SIZE: redesmyn_graph_layout::Size =
    redesmyn_graph_layout::Size {
        width: 960,
        height: 640,
    };

use crate::constants::{
    GRAPH_PADDING, TRUNK_COMMIT_PADDING, TRUNK_COMMIT_ROW_HEIGHT, TRUNK_COMMIT_SPACING, TRUNK_GAP,
    TRUNK_MARKER_WIDTH, TRUNK_SHA_WIDTH, TRUNK_TITLE_WIDTH,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GraphNodeId {
    Task(TaskId),
    Trunk,
}

impl fmt::Display for GraphNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task(id) => id.fmt(f),
            Self::Trunk => write!(f, "trunk"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GraphEdgeId {
    pub from: GraphNodeId,
    pub to: GraphNodeId,
}

impl fmt::Display for GraphEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "edge:{from}:{to}", from = self.from, to = self.to)
    }
}

#[derive(Debug, Clone)]
pub struct GraphSceneNode {
    pub id: GraphNodeId,
    pub task_slug: SharedString,
    pub branch_slug: SharedString,
    pub title: SharedString,
    pub parent_id: Option<GraphNodeId>,
    pub state: TaskState,
    pub merge_readiness: MergeReadiness,
    pub agent_status: AgentStatus,
    pub branch_name: Option<SharedString>,
    pub latest_session: Option<TaskSessionSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSessionSummary {
    pub session_id: SessionId,
    pub session_event_id: SessionEventId,
    pub message_preview: Option<SharedString>,
}

#[derive(Debug, Clone)]
pub struct GraphSceneEdge {
    pub id: GraphEdgeId,
    pub commit_count: Option<u32>,
    pub commit_count_label: Option<SharedString>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Unknown,
    Running,
    Blocked,
    Stopped,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrunkCommit {
    pub sha: SharedString,
    pub title: Option<SharedString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrunkTimeline {
    pub base_sha: SharedString,
    pub base_commit: Option<TrunkCommit>,
    pub commits_before: Vec<TrunkCommit>,
    pub commits_after: Vec<TrunkCommit>,
    pub has_more_before: bool,
    pub has_more_after: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrunkMarkKind {
    Commit,
    Base,
    Connector,
    Ellipsis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrunkMark {
    pub(crate) kind: TrunkMarkKind,
    pub(crate) sha: Option<SharedString>,
    pub(crate) title: Option<SharedString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrunkLayout {
    pub(crate) marks: Vec<TrunkMark>,
    /// World-space Y offset (relative to the trunk node origin) for the trunk→graph connector.
    pub(crate) base_offset: i32,
    pub(crate) commit_spacing: i32,
    pub(crate) commit_padding: i32,
    pub(crate) row_height: i32,
    pub(crate) span_height: i32,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GraphSelection {
    pub selected_node: Option<GraphNodeId>,
    pub selected_nodes: BTreeSet<GraphNodeId>,
    pub selected_edge: Option<GraphEdgeId>,
    pub hovered_node: Option<GraphNodeId>,
    pub hovered_edge: Option<GraphEdgeId>,
}

#[derive(Debug, Clone)]
pub struct GraphScene {
    nodes: BTreeMap<GraphNodeId, GraphSceneNode>,
    edges: BTreeMap<GraphEdgeId, GraphSceneEdge>,
    selection: GraphSelection,
    node_sizes: BTreeMap<GraphNodeId, redesmyn_graph_layout::Size>,
    node_positions: BTreeMap<GraphNodeId, redesmyn_graph_layout::Point>,
    bounds: redesmyn_graph_layout::Rect,
    trunk_timeline: Option<TrunkTimeline>,
    trunk_layout: Option<TrunkLayout>,
    layout_nodes_scratch: Vec<LayoutNode<GraphNodeId>>,
    layout_engine: Option<ForestLayoutEngine<GraphNodeId>>,
    layout_engine_dirty: bool,
}

impl GraphScene {
    #[must_use]
    pub fn demo() -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_scene.demo_build");
        let _guard = span.enter();

        let mut scene = Self::empty_demo();
        scene.trunk_timeline = Some(demo_trunk_timeline());
        let a = GraphNodeId::Task(TaskId::from_bytes([1; 16]));
        let b = GraphNodeId::Task(TaskId::from_bytes([2; 16]));
        let c = GraphNodeId::Task(TaskId::from_bytes([3; 16]));
        let d = GraphNodeId::Task(TaskId::from_bytes([4; 16]));

        scene.insert_node(GraphSceneNode {
            id: a,
            task_slug: "T-1".into(),
            branch_slug: "root".into(),
            title: "T-1 Root".into(),
            parent_id: None,
            state: TaskState::InProgress,
            merge_readiness: MergeReadiness::Unknown,
            agent_status: AgentStatus::Running,
            branch_name: Some("feat/root".into()),
            latest_session: None,
        });
        scene.insert_node(GraphSceneNode {
            id: b,
            task_slug: "T-2".into(),
            branch_slug: "child-a".into(),
            title: "T-2 Child A".into(),
            parent_id: Some(a),
            state: TaskState::Blocked,
            merge_readiness: MergeReadiness::Blocked,
            agent_status: AgentStatus::Blocked,
            branch_name: Some("feat/child-a".into()),
            latest_session: None,
        });
        scene.insert_node(GraphSceneNode {
            id: c,
            task_slug: "T-3".into(),
            branch_slug: "child-b".into(),
            title: "T-3 Child B".into(),
            parent_id: Some(a),
            state: TaskState::Todo,
            merge_readiness: MergeReadiness::Ready,
            agent_status: AgentStatus::Stopped,
            branch_name: Some("feat/child-b".into()),
            latest_session: None,
        });
        scene.insert_node(GraphSceneNode {
            id: d,
            task_slug: "T-4".into(),
            branch_slug: "grandchild".into(),
            title: "T-4 Grandchild".into(),
            parent_id: Some(b),
            state: TaskState::Done,
            merge_readiness: MergeReadiness::Ready,
            agent_status: AgentStatus::Stopped,
            branch_name: Some("feat/grandchild".into()),
            latest_session: None,
        });

        scene.insert_edge(GraphEdgeId { from: a, to: b }, Some(12));
        scene.insert_edge(GraphEdgeId { from: a, to: c }, Some(3));
        scene.insert_edge(GraphEdgeId { from: b, to: d }, Some(27));

        scene.sync_trunk_overlay();
        scene.relayout();
        scene
    }

    #[must_use]
    pub(crate) fn empty_demo() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            selection: GraphSelection::default(),
            node_sizes: BTreeMap::new(),
            node_positions: BTreeMap::new(),
            bounds: redesmyn_graph_layout::Rect {
                origin: redesmyn_graph_layout::Point { x: 0, y: 0 },
                size: redesmyn_graph_layout::Size {
                    width: 0,
                    height: 0,
                },
            },
            trunk_timeline: None,
            trunk_layout: None,
            layout_nodes_scratch: Vec::new(),
            layout_engine: None,
            layout_engine_dirty: true,
        }
    }

    pub fn set_trunk_timeline(&mut self, trunk: Option<TrunkTimeline>) {
        if self.trunk_timeline == trunk {
            return;
        }

        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.graph_scene.set_trunk_timeline",
            enabled = trunk.is_some(),
        );
        let _guard = span.enter();

        self.trunk_timeline = trunk;
        self.sync_trunk_overlay();
        self.relayout();
    }

    pub(crate) fn trunk_layout(&self) -> Option<&TrunkLayout> {
        self.trunk_layout.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn insert_demo_node(&mut self, id: GraphNodeId) {
        self.insert_node(GraphSceneNode {
            id,
            task_slug: "demo".into(),
            branch_slug: "demo".into(),
            title: "demo".into(),
            parent_id: None,
            state: TaskState::Unknown,
            merge_readiness: MergeReadiness::Unknown,
            agent_status: AgentStatus::Unknown,
            branch_name: None,
            latest_session: None,
        });
        self.relayout();
    }

    pub fn nodes(&self) -> impl Iterator<Item = &GraphSceneNode> {
        self.nodes.values()
    }

    pub fn edges(&self) -> impl Iterator<Item = &GraphSceneEdge> {
        self.edges.values()
    }

    #[must_use]
    pub fn node(&self, id: GraphNodeId) -> Option<&GraphSceneNode> {
        self.nodes.get(&id)
    }

    #[must_use]
    pub fn selection(&self) -> &GraphSelection {
        &self.selection
    }

    pub fn select_node(&mut self, id: GraphNodeId) {
        self.set_selection(Some(id), None);
    }

    pub fn select_edge(&mut self, id: GraphEdgeId) {
        self.set_selection(None, Some(id));
    }

    pub fn toggle_node(&mut self, id: GraphNodeId) {
        let previous_selected_node = self.selection.selected_node;
        self.selection.selected_edge = None;

        let removed = self.selection.selected_nodes.remove(&id);
        if removed {
            if self.selection.selected_node == Some(id) {
                self.selection.selected_node = None;
            }
        } else {
            self.selection.selected_nodes.insert(id);
        }

        if let Some(selected_node) = self.selection.selected_node {
            self.selection.selected_nodes.insert(selected_node);
        }

        let sizes_changed = self.apply_expandedness_policy();
        if previous_selected_node != self.selection.selected_node || sizes_changed {
            self.relayout();
        }
    }

    pub fn clear_selection(&mut self) {
        self.set_selection(None, None);
    }

    fn set_selection(
        &mut self,
        selected_node: Option<GraphNodeId>,
        selected_edge: Option<GraphEdgeId>,
    ) {
        let selection_changed = self.selection.selected_node != selected_node
            || self.selection.selected_edge != selected_edge;

        self.selection.selected_node = selected_node;
        self.selection.selected_edge = selected_edge;
        self.selection.selected_nodes.clear();
        if let Some(selected_node) = selected_node
            && selected_edge.is_none()
        {
            self.selection.selected_nodes.insert(selected_node);
        }

        let sizes_changed = self.apply_expandedness_policy();

        if selection_changed || sizes_changed {
            self.relayout();
        }
    }

    #[must_use]
    pub fn layout_bounds(&self) -> redesmyn_graph_layout::Rect {
        self.bounds
    }

    fn apply_expandedness_policy(&mut self) -> bool {
        self.node_sizes.retain(|id, _| self.nodes.contains_key(id));

        let expanded = self.selection.selected_node;
        let mut changed = false;

        for id in self.nodes.keys().copied() {
            if id == GraphNodeId::Trunk {
                continue;
            }
            let desired = if Some(id) == expanded {
                EXPANDED_TASK_NODE_SIZE
            } else {
                COLLAPSED_TASK_NODE_SIZE
            };

            let size = self
                .node_sizes
                .entry(id)
                .or_insert(COLLAPSED_TASK_NODE_SIZE);

            if *size != desired {
                *size = desired;
                changed = true;
            }
        }

        changed
    }

    pub fn set_hovered_node(&mut self, id: Option<GraphNodeId>) -> bool {
        if self.selection.hovered_node == id && self.selection.hovered_edge.is_none() {
            return false;
        }

        self.selection.hovered_node = id;
        self.selection.hovered_edge = None;
        true
    }

    pub fn set_hovered_edge(&mut self, id: Option<GraphEdgeId>) -> bool {
        if self.selection.hovered_edge == id && self.selection.hovered_node.is_none() {
            return false;
        }

        self.selection.hovered_edge = id;
        self.selection.hovered_node = None;
        true
    }

    pub fn clear_hover(&mut self) -> bool {
        if self.selection.hovered_node.is_none() && self.selection.hovered_edge.is_none() {
            return false;
        }

        self.selection.hovered_node = None;
        self.selection.hovered_edge = None;
        true
    }

    pub fn replace_from_epic_graph(&mut self, graph: &redesmyn_protocol::client::EpicGraph) {
        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.graph_scene.replace_from_epic_graph",
            epic_slug = %graph.epic_slug,
            node_count = graph.nodes.len(),
            edge_count = graph.edges.len(),
        );
        let _guard = span.enter();

        // TODO(T-57): Wire trunk timeline data in once the control-plane read model exposes it on
        // `redesmyn_protocol::client::EpicGraph` by translating into `TrunkTimeline` and calling
        // `set_trunk_timeline(Some(..))` (or having the caller do it) so trunk data present ⇒ trunk
        // renders. Until then, trunk rendering can be toggled explicitly via `set_trunk_timeline`.

        // Prefer stable protocol identifiers when available, but fall back to deterministic
        // slug hashing for older protocol versions (or partial payloads) that omit ids.
        let mut node_id_by_slug: BTreeMap<String, GraphNodeId> = BTreeMap::new();
        for node in &graph.nodes {
            let id = match node.task_id {
                Some(id) => GraphNodeId::Task(id),
                None => GraphNodeId::Task(temporary_task_id_for_slug(&node.task_slug)),
            };
            node_id_by_slug.insert(node.task_slug.clone(), id);
        }

        let known_node_ids: BTreeSet<GraphNodeId> = node_id_by_slug.values().copied().collect();
        let mut parent_by_child = BTreeMap::new();

        let agent_status_by_task_id = agent_status_by_task_id(graph);
        let latest_session_by_task_id = latest_session_by_task_id(graph);

        let mut edges = BTreeMap::new();
        for edge in &graph.edges {
            let from = match edge.from_task_id {
                Some(id) => GraphNodeId::Task(id),
                None => node_id_by_slug
                    .get(&edge.from_task_slug)
                    .copied()
                    .unwrap_or_else(|| {
                        GraphNodeId::Task(temporary_task_id_for_slug(&edge.from_task_slug))
                    }),
            };
            let to = match edge.to_task_id {
                Some(id) => GraphNodeId::Task(id),
                None => node_id_by_slug
                    .get(&edge.to_task_slug)
                    .copied()
                    .unwrap_or_else(|| {
                        GraphNodeId::Task(temporary_task_id_for_slug(&edge.to_task_slug))
                    }),
            };
            let edge_id = GraphEdgeId { from, to };
            edges.insert(
                edge_id,
                GraphSceneEdge {
                    id: edge_id,
                    commit_count: None,
                    commit_count_label: None,
                },
            );
            parent_by_child
                .entry(to)
                .and_modify(|existing| {
                    if from < *existing {
                        *existing = from;
                    }
                })
                .or_insert(from);
        }

        let mut nodes = BTreeMap::new();
        for node in &graph.nodes {
            let id = node_id_by_slug
                .get(&node.task_slug)
                .copied()
                .unwrap_or_else(|| GraphNodeId::Task(temporary_task_id_for_slug(&node.task_slug)));

            let parent_id = node
                .parent_task_id
                .map(GraphNodeId::Task)
                .or_else(|| parent_by_child.get(&id).copied())
                .filter(|parent_id| known_node_ids.contains(parent_id));

            nodes.insert(
                id,
                GraphSceneNode {
                    id,
                    task_slug: SharedString::new(node.task_slug.clone()),
                    branch_slug: node
                        .branch_name
                        .as_ref()
                        .and_then(|name| name.rsplit('/').next())
                        .filter(|slug| !slug.is_empty())
                        .map(|slug| SharedString::new(slug.to_string()))
                        .unwrap_or_else(|| SharedString::new(node.task_slug.clone())),
                    title: SharedString::new(node.title.clone()),
                    parent_id,
                    state: node.state,
                    merge_readiness: node.merge_readiness,
                    agent_status: node
                        .task_id
                        .and_then(|id| agent_status_by_task_id.get(&id).copied())
                        .unwrap_or(AgentStatus::Unknown),
                    branch_name: node
                        .branch_name
                        .as_ref()
                        .map(|name| SharedString::new(name.clone())),
                    latest_session: node
                        .task_id
                        .and_then(|id| latest_session_by_task_id.get(&id).cloned()),
                },
            );
        }

        self.nodes = nodes;
        self.edges = edges;
        self.sync_trunk_overlay();
        self.layout_engine_dirty = true;

        self.selection
            .selected_nodes
            .retain(|selected| self.nodes.contains_key(selected));

        if let Some(selected) = self.selection.selected_node
            && !self.nodes.contains_key(&selected)
        {
            self.selection.selected_node = None;
        }

        if let Some(selected) = self.selection.selected_node
            && !self.selection.selected_nodes.contains(&selected)
        {
            self.selection.selected_node = None;
        }

        if let Some(selected) = self.selection.selected_node
            && self.selection.selected_edge.is_none()
        {
            self.selection.selected_nodes.insert(selected);
        }

        if let Some(selected) = self.selection.selected_edge
            && !self.edges.contains_key(&selected)
        {
            self.selection.selected_edge = None;
        }

        if let Some(hovered) = self.selection.hovered_node
            && !self.nodes.contains_key(&hovered)
        {
            self.selection.hovered_node = None;
        }

        if let Some(hovered) = self.selection.hovered_edge
            && !self.edges.contains_key(&hovered)
        {
            self.selection.hovered_edge = None;
        }

        self.apply_expandedness_policy();
        self.relayout();
    }

    pub(crate) fn node_world_origin(&self, id: GraphNodeId) -> gpui::Point<f32> {
        let origin = self
            .node_positions
            .get(&id)
            .copied()
            .unwrap_or(redesmyn_graph_layout::Point { x: 0, y: 0 });
        gpui::point(origin.x as f32, origin.y as f32)
    }

    pub(crate) fn node_world_size(&self, id: GraphNodeId) -> redesmyn_graph_layout::Size {
        self.node_sizes
            .get(&id)
            .copied()
            .unwrap_or_else(default_node_size)
    }

    fn insert_node(&mut self, node: GraphSceneNode) {
        let id = node.id;
        self.nodes.insert(id, node);
        self.node_sizes.entry(id).or_insert_with(default_node_size);
        self.layout_engine_dirty = true;
    }

    fn insert_edge(&mut self, id: GraphEdgeId, commit_count: Option<u32>) {
        self.edges.insert(
            id,
            GraphSceneEdge {
                id,
                commit_count,
                commit_count_label: commit_count.map(|count| count.to_string().into()),
            },
        );
    }

    fn sync_trunk_overlay(&mut self) {
        let trunk_id = GraphNodeId::Trunk;

        self.nodes.remove(&trunk_id);
        self.node_sizes.remove(&trunk_id);
        self.node_positions.remove(&trunk_id);
        self.trunk_layout = None;

        self.edges
            .retain(|edge_id, _| edge_id.from != trunk_id && edge_id.to != trunk_id);

        if self.trunk_timeline.is_none() {
            return;
        }

        self.nodes.insert(
            trunk_id,
            GraphSceneNode {
                id: trunk_id,
                task_slug: "trunk".into(),
                branch_slug: "trunk".into(),
                title: "Trunk".into(),
                parent_id: None,
                state: TaskState::Unknown,
                merge_readiness: MergeReadiness::Unknown,
                agent_status: AgentStatus::Unknown,
                branch_name: None,
                latest_session: None,
            },
        );
        self.node_sizes.insert(trunk_id, trunk_default_size());

        let root_ids: Vec<GraphNodeId> = self
            .nodes
            .values()
            .filter(|node| node.id != trunk_id && node.parent_id.is_none())
            .map(|node| node.id)
            .collect();

        for root_id in root_ids {
            self.insert_edge(
                GraphEdgeId {
                    from: trunk_id,
                    to: root_id,
                },
                None,
            );
        }
    }

    fn relayout(&mut self) {
        let trunk_layout = self.trunk_timeline.as_ref().map(trunk_layout_for_timeline);
        let x_offset = trunk_layout
            .as_ref()
            .map(|_| GRAPH_PADDING + trunk_column_width() + TRUNK_GAP)
            .unwrap_or(GRAPH_PADDING);

        let y_offset = trunk_layout
            .as_ref()
            .map(|layout| layout_anchor_y(layout.base_offset))
            .unwrap_or(GRAPH_PADDING);

        let config = LayoutConfig {
            origin: redesmyn_graph_layout::Point {
                x: x_offset,
                y: y_offset,
            },
            ..LayoutConfig::default()
        };

        if self.layout_engine_dirty || self.layout_engine.is_none() {
            self.layout_nodes_scratch.clear();
            self.layout_nodes_scratch.reserve(
                self.nodes
                    .len()
                    .saturating_sub(self.layout_nodes_scratch.capacity()),
            );

            for node in self.nodes.values() {
                if node.id == GraphNodeId::Trunk {
                    continue;
                }
                self.layout_nodes_scratch.push(LayoutNode {
                    id: node.id,
                    parent_id: node.parent_id,
                    size: self
                        .node_sizes
                        .get(&node.id)
                        .copied()
                        .unwrap_or_else(default_node_size),
                });
            }

            match ForestLayoutEngine::new(
                self.layout_nodes_scratch.iter().copied(),
                LayoutOptions::default(),
            ) {
                Ok(engine) => {
                    self.layout_engine = Some(engine);
                    self.layout_engine_dirty = false;
                }
                Err(error) => {
                    redesmyn_logging::tracing::error!(
                        ?error,
                        "graph layout engine build failed; keeping previous positions"
                    );
                    return;
                }
            }
        }

        let Some(engine) = self.layout_engine.as_mut() else {
            return;
        };

        if let Err(error) = engine.set_sizes(self.nodes.values().filter_map(|node| {
            if node.id == GraphNodeId::Trunk {
                return None;
            }

            Some((
                node.id,
                self.node_sizes
                    .get(&node.id)
                    .copied()
                    .unwrap_or_else(default_node_size),
            ))
        })) {
            redesmyn_logging::tracing::error!(
                ?error,
                "graph layout engine rejected node sizes; keeping previous positions"
            );
            return;
        }

        engine.layout_in_place(config);
        let view = engine.output_view();
        self.node_positions.clear();
        for (id, point) in view.ids.iter().copied().zip(view.positions.iter().copied()) {
            self.node_positions.insert(id, point);
        }
        self.bounds = view.bounds;

        if let Some(layout) = trunk_layout {
            let trunk_span_height = layout.span_height;
            self.trunk_layout = Some(layout);
            self.node_positions.insert(
                GraphNodeId::Trunk,
                redesmyn_graph_layout::Point {
                    x: GRAPH_PADDING,
                    y: GRAPH_PADDING,
                },
            );

            let trunk_height =
                trunk_height_for_scene(&self.node_positions, &self.node_sizes, trunk_span_height);
            self.node_sizes.insert(
                GraphNodeId::Trunk,
                redesmyn_graph_layout::Size {
                    width: trunk_column_width(),
                    height: trunk_height,
                },
            );

            self.bounds = bounds_with_trunk(self.bounds, trunk_column_width(), trunk_height);
        } else {
            self.trunk_layout = None;
        }
    }
}

fn default_node_size() -> redesmyn_graph_layout::Size {
    COLLAPSED_TASK_NODE_SIZE
}

fn trunk_column_width() -> i32 {
    TRUNK_TITLE_WIDTH + TRUNK_MARKER_WIDTH + TRUNK_SHA_WIDTH
}

fn trunk_default_size() -> redesmyn_graph_layout::Size {
    redesmyn_graph_layout::Size {
        width: trunk_column_width(),
        height: trunk_layout_default_span_height(),
    }
}

fn trunk_layout_default_span_height() -> i32 {
    TRUNK_COMMIT_PADDING * 2 + TRUNK_COMMIT_ROW_HEIGHT
}

fn trunk_height_for_scene(
    positions: &BTreeMap<GraphNodeId, redesmyn_graph_layout::Point>,
    sizes: &BTreeMap<GraphNodeId, redesmyn_graph_layout::Size>,
    trunk_span_height: i32,
) -> i32 {
    let mut graph_bottom = GRAPH_PADDING;
    for (id, pos) in positions {
        if *id == GraphNodeId::Trunk {
            continue;
        }
        let size = sizes.get(id).copied().unwrap_or_else(default_node_size);
        graph_bottom = graph_bottom.max(pos.y + size.height);
    }
    trunk_span_height.max(graph_bottom - GRAPH_PADDING)
}

fn bounds_with_trunk(
    bounds: redesmyn_graph_layout::Rect,
    trunk_width: i32,
    trunk_height: i32,
) -> redesmyn_graph_layout::Rect {
    let left = bounds.origin.x.min(GRAPH_PADDING);
    let top = bounds.origin.y.min(GRAPH_PADDING);
    let right = bounds.right().max(GRAPH_PADDING + trunk_width);
    let bottom = bounds.bottom().max(GRAPH_PADDING + trunk_height);
    redesmyn_graph_layout::Rect {
        origin: redesmyn_graph_layout::Point { x: left, y: top },
        size: redesmyn_graph_layout::Size {
            width: right - left,
            height: bottom - top,
        },
    }
}

fn layout_anchor_y(trunk_base_offset: i32) -> i32 {
    let root_center_y = GRAPH_PADDING + trunk_base_offset;
    let desired = root_center_y - COLLAPSED_TASK_NODE_SIZE.height / 2;
    GRAPH_PADDING.max(desired)
}

fn trunk_layout_for_timeline(timeline: &TrunkTimeline) -> TrunkLayout {
    let mut marks: Vec<TrunkMark> = Vec::new();

    if timeline.base_sha.is_empty() {
        return TrunkLayout {
            marks,
            base_offset: TRUNK_COMMIT_PADDING + TRUNK_COMMIT_ROW_HEIGHT / 2,
            commit_spacing: TRUNK_COMMIT_SPACING,
            commit_padding: TRUNK_COMMIT_PADDING,
            row_height: TRUNK_COMMIT_ROW_HEIGHT,
            span_height: trunk_layout_default_span_height(),
        };
    }

    if timeline.has_more_after {
        marks.push(TrunkMark {
            kind: TrunkMarkKind::Ellipsis,
            sha: None,
            title: None,
        });
    }

    for commit in timeline.commits_after.iter().rev() {
        marks.push(TrunkMark {
            kind: TrunkMarkKind::Commit,
            sha: Some(commit.sha.clone()),
            title: commit.title.clone(),
        });
    }

    let base_commit = timeline.base_commit.clone().unwrap_or_else(|| TrunkCommit {
        sha: timeline.base_sha.clone(),
        title: None,
    });

    marks.push(TrunkMark {
        kind: TrunkMarkKind::Base,
        sha: Some(base_commit.sha),
        title: base_commit.title,
    });

    for commit in &timeline.commits_before {
        marks.push(TrunkMark {
            kind: TrunkMarkKind::Commit,
            sha: Some(commit.sha.clone()),
            title: commit.title.clone(),
        });
    }

    if timeline.has_more_before {
        marks.push(TrunkMark {
            kind: TrunkMarkKind::Ellipsis,
            sha: None,
            title: None,
        });
    }

    let base_index = marks
        .iter()
        .position(|mark| mark.kind == TrunkMarkKind::Base)
        .unwrap_or_else(|| marks.len() / 2);

    let connector_index = base_index;
    marks.insert(
        connector_index,
        TrunkMark {
            kind: TrunkMarkKind::Connector,
            sha: None,
            title: None,
        },
    );

    let mark_count = marks.len() as i32;
    let span_height = if mark_count > 1 {
        TRUNK_COMMIT_PADDING * 2 + (mark_count - 1) * TRUNK_COMMIT_SPACING + TRUNK_COMMIT_ROW_HEIGHT
    } else {
        trunk_layout_default_span_height()
    };

    let base_offset = if mark_count > 0 {
        TRUNK_COMMIT_PADDING
            + connector_index as i32 * TRUNK_COMMIT_SPACING
            + TRUNK_COMMIT_ROW_HEIGHT / 2
    } else {
        TRUNK_COMMIT_PADDING + TRUNK_COMMIT_ROW_HEIGHT / 2
    };

    TrunkLayout {
        marks,
        base_offset,
        commit_spacing: TRUNK_COMMIT_SPACING,
        commit_padding: TRUNK_COMMIT_PADDING,
        row_height: TRUNK_COMMIT_ROW_HEIGHT,
        span_height,
    }
}

fn agent_status_by_task_id(
    graph: &redesmyn_protocol::client::EpicGraph,
) -> BTreeMap<TaskId, AgentStatus> {
    fn priority(status: AgentStatus) -> u8 {
        match status {
            AgentStatus::Error => 4,
            AgentStatus::Blocked => 3,
            AgentStatus::Running => 2,
            AgentStatus::Stopped => 1,
            AgentStatus::Unknown => 0,
        }
    }

    fn from_command_state(state: CommandState) -> AgentStatus {
        match state {
            CommandState::Queued | CommandState::Running | CommandState::Accepted => {
                AgentStatus::Running
            }
            CommandState::Blocked | CommandState::Resumable => AgentStatus::Blocked,
            CommandState::Failed => AgentStatus::Error,
            CommandState::Succeeded | CommandState::Canceled => AgentStatus::Stopped,
            CommandState::Unknown => AgentStatus::Unknown,
        }
    }

    let mut out: BTreeMap<TaskId, AgentStatus> = BTreeMap::new();
    for summary in &graph.command_summaries {
        let Some(task_id) = summary.target_task_id else {
            continue;
        };
        let status = from_command_state(summary.state);
        out.entry(task_id)
            .and_modify(|existing| {
                if priority(status) > priority(*existing) {
                    *existing = status;
                }
            })
            .or_insert(status);
    }
    out
}

fn latest_session_by_task_id(
    graph: &redesmyn_protocol::client::EpicGraph,
) -> BTreeMap<TaskId, TaskSessionSummary> {
    fn collapse_whitespace(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        let mut saw_space = false;
        for ch in value.chars() {
            if ch.is_whitespace() {
                if !saw_space {
                    out.push(' ');
                    saw_space = true;
                }
                continue;
            }
            out.push(ch);
            saw_space = false;
        }
        out.trim().to_string()
    }

    let mut out: BTreeMap<TaskId, (redesmyn_protocol::Timestamp, TaskSessionSummary)> =
        BTreeMap::new();
    for session in &graph.session_summaries {
        let summary = TaskSessionSummary {
            session_id: session.session_id,
            session_event_id: session.session_event_id,
            message_preview: session
                .message_preview
                .as_deref()
                .map(collapse_whitespace)
                .and_then(|preview| (!preview.is_empty()).then(|| SharedString::new(preview))),
        };

        out.entry(session.task_id)
            .and_modify(|(existing_at, existing)| {
                if session.last_event_at > *existing_at {
                    *existing_at = session.last_event_at;
                    *existing = summary.clone();
                }
            })
            .or_insert((session.last_event_at, summary));
    }

    out.into_iter()
        .map(|(task_id, (_, summary))| (task_id, summary))
        .collect()
}

fn temporary_task_id_for_slug(slug: &str) -> TaskId {
    // TEMPORARY: The client protocol currently provides task identifiers as slugs (strings). This
    // adapter keeps graph IDs stable and deterministic until the protocol exports real identifiers.
    // When protocol `TaskId` is available, delete this function and use the provided id directly.
    // (Avoid std::collections hashing because RandomState is non-deterministic.)
    let bytes = stable_128bit_hash(slug.as_bytes());
    TaskId::from_bytes(bytes)
}

fn stable_128bit_hash(input: &[u8]) -> [u8; 16] {
    fn fnv1a(seed: u64, input: &[u8]) -> u64 {
        let mut hash = seed;
        for byte in input {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    let hi = fnv1a(0xcbf29ce484222325, input);
    let lo = fnv1a(0xaf63dc4c8601ec8c, input);

    let mut out = [0_u8; 16];
    out[..8].copy_from_slice(&hi.to_be_bytes());
    out[8..].copy_from_slice(&lo.to_be_bytes());
    out
}

fn demo_trunk_timeline() -> TrunkTimeline {
    TrunkTimeline {
        base_sha: SharedString::new("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()),
        base_commit: Some(TrunkCommit {
            sha: SharedString::new("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()),
            title: Some("base".into()),
        }),
        commits_before: vec![
            TrunkCommit {
                sha: SharedString::new("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string()),
                title: Some("before".into()),
            },
            TrunkCommit {
                sha: SharedString::new("ccccccccccccccccccccccccbbbbbbbbbbbbbbbb".to_string()),
                title: Some("older".into()),
            },
        ],
        commits_after: vec![
            TrunkCommit {
                sha: SharedString::new("dddddddddddddddddddddddddddddddddddddddd".to_string()),
                title: Some("after".into()),
            },
            TrunkCommit {
                sha: SharedString::new("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string()),
                title: Some("head".into()),
            },
        ],
        has_more_before: true,
        has_more_after: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporary_task_id_hash_is_deterministic() {
        let a = temporary_task_id_for_slug("T-50");
        let b = temporary_task_id_for_slug("T-50");
        assert_eq!(a, b);
    }

    #[test]
    fn selection_expands_only_selected_node() {
        let mut scene = GraphScene::demo();
        let ids: Vec<_> = scene
            .nodes()
            .map(|node| node.id)
            .filter(|id| *id != GraphNodeId::Trunk)
            .collect();
        let first = ids[0];

        scene.select_node(first);
        assert_eq!(scene.node_world_size(first), EXPANDED_TASK_NODE_SIZE);
        for id in ids.iter().copied().filter(|id| *id != first) {
            assert_eq!(scene.node_world_size(id), COLLAPSED_TASK_NODE_SIZE);
        }

        scene.select_edge(GraphEdgeId {
            from: first,
            to: ids[1],
        });
        for id in ids.iter().copied() {
            assert_eq!(scene.node_world_size(id), COLLAPSED_TASK_NODE_SIZE);
        }

        scene.clear_selection();
        for id in ids {
            assert_eq!(scene.node_world_size(id), COLLAPSED_TASK_NODE_SIZE);
        }
    }

    #[test]
    fn toggle_selection_adds_and_removes_nodes() {
        let mut scene = GraphScene::empty_demo();
        let a = GraphNodeId::Task(TaskId::from_bytes([1; 16]));
        let b = GraphNodeId::Task(TaskId::from_bytes([2; 16]));
        scene.insert_demo_node(a);
        scene.insert_demo_node(b);

        scene.toggle_node(a);
        assert_eq!(
            scene
                .selection()
                .selected_nodes
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![a]
        );
        assert_eq!(scene.selection().selected_node, None);

        scene.select_node(a);
        assert_eq!(scene.selection().selected_node, Some(a));
        assert_eq!(
            scene
                .selection()
                .selected_nodes
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![a]
        );

        scene.toggle_node(b);
        assert_eq!(
            scene
                .selection()
                .selected_nodes
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![a, b]
        );
        assert_eq!(scene.selection().selected_node, Some(a));

        scene.toggle_node(a);
        assert_eq!(
            scene
                .selection()
                .selected_nodes
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![b]
        );
        assert_eq!(scene.selection().selected_node, None);

        scene.toggle_node(b);
        assert!(scene.selection().selected_nodes.is_empty());
        assert_eq!(scene.selection().selected_node, None);
    }

    #[test]
    fn selecting_edge_clears_node_selection() {
        let mut scene = GraphScene::empty_demo();
        let a = GraphNodeId::Task(TaskId::from_bytes([1; 16]));
        let b = GraphNodeId::Task(TaskId::from_bytes([2; 16]));
        scene.insert_demo_node(a);
        scene.insert_demo_node(b);

        scene.toggle_node(a);
        scene.toggle_node(b);
        assert_eq!(scene.selection().selected_nodes.len(), 2);

        let edge = GraphEdgeId { from: a, to: b };
        scene.select_edge(edge);
        assert!(scene.selection().selected_nodes.is_empty());
        assert_eq!(scene.selection().selected_node, None);
        assert_eq!(scene.selection().selected_edge, Some(edge));
    }

    #[test]
    fn trunk_layout_offsets_root_nodes() {
        let mut scene = GraphScene::empty_demo();
        let root_id = GraphNodeId::Task(TaskId::from_bytes([42; 16]));
        scene.insert_demo_node(root_id);

        scene.set_trunk_timeline(Some(demo_trunk_timeline()));

        let root_pos = scene.node_positions.get(&root_id).copied().unwrap();
        assert_eq!(root_pos.x, GRAPH_PADDING + trunk_column_width() + TRUNK_GAP);

        let trunk_pos = scene
            .node_positions
            .get(&GraphNodeId::Trunk)
            .copied()
            .unwrap();
        assert_eq!(trunk_pos.x, GRAPH_PADDING);
        assert_eq!(trunk_pos.y, GRAPH_PADDING);

        let trunk_layout = scene.trunk_layout().unwrap();
        let root_size = scene.node_sizes.get(&root_id).copied().unwrap();
        let root_center_y = root_pos.y + root_size.height / 2;
        let trunk_base_anchor_y = trunk_pos.y + trunk_layout.base_offset;
        assert_eq!(root_center_y, trunk_base_anchor_y);
    }

    #[test]
    fn replace_from_epic_graph_preserves_selection_when_possible() {
        let a_id = TaskId::from_bytes([1; 16]);
        let b_id = TaskId::from_bytes([2; 16]);

        let mut graph = redesmyn_protocol::client::EpicGraph {
            epic_slug: "demo".to_string(),
            nodes: vec![
                redesmyn_protocol::client::EpicTaskNode {
                    task_slug: "T-1".to_string(),
                    title: "First".to_string(),
                    task_id: Some(a_id),
                    parent_task_id: None,
                    state: TaskState::Todo,
                    branch_name: None,
                    merge_readiness: MergeReadiness::Unknown,
                },
                redesmyn_protocol::client::EpicTaskNode {
                    task_slug: "T-2".to_string(),
                    title: "Second".to_string(),
                    task_id: Some(b_id),
                    parent_task_id: Some(a_id),
                    state: TaskState::Todo,
                    branch_name: None,
                    merge_readiness: MergeReadiness::Unknown,
                },
            ],
            edges: vec![redesmyn_protocol::client::EpicTaskEdge {
                from_task_slug: "T-1".to_string(),
                to_task_slug: "T-2".to_string(),
                from_task_id: Some(a_id),
                to_task_id: Some(b_id),
            }],
            epic_id: None,
            epic_title: None,
            workspace_id: None,
            repo_id: None,
            command_summaries: Vec::new(),
            daemon_presences: Vec::new(),
            session_summaries: Vec::new(),
            as_of_event_id: None,
        };

        let mut scene = GraphScene::empty_demo();
        scene.replace_from_epic_graph(&graph);
        let selected = GraphNodeId::Task(a_id);
        scene.select_node(selected);

        graph.nodes[0].title = "First (updated)".to_string();
        scene.replace_from_epic_graph(&graph);

        assert_eq!(scene.selection().selected_node, Some(selected));
        assert!(scene.selection().selected_nodes.contains(&selected));
    }

    #[test]
    fn replace_from_epic_graph_clears_selection_when_node_missing() {
        let a_id = TaskId::from_bytes([1; 16]);

        let mut scene = GraphScene::empty_demo();
        let graph = redesmyn_protocol::client::EpicGraph {
            epic_slug: "demo".to_string(),
            nodes: vec![redesmyn_protocol::client::EpicTaskNode {
                task_slug: "T-1".to_string(),
                title: "First".to_string(),
                task_id: Some(a_id),
                parent_task_id: None,
                state: TaskState::Todo,
                branch_name: None,
                merge_readiness: MergeReadiness::Unknown,
            }],
            edges: Vec::new(),
            epic_id: None,
            epic_title: None,
            workspace_id: None,
            repo_id: None,
            command_summaries: Vec::new(),
            daemon_presences: Vec::new(),
            session_summaries: Vec::new(),
            as_of_event_id: None,
        };

        scene.replace_from_epic_graph(&graph);
        scene.select_node(GraphNodeId::Task(a_id));

        let empty = redesmyn_protocol::client::EpicGraph {
            epic_slug: "demo".to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            epic_id: None,
            epic_title: None,
            workspace_id: None,
            repo_id: None,
            command_summaries: Vec::new(),
            daemon_presences: Vec::new(),
            session_summaries: Vec::new(),
            as_of_event_id: None,
        };

        scene.replace_from_epic_graph(&empty);
        assert!(scene.selection().selected_node.is_none());
        assert!(scene.selection().selected_nodes.is_empty());
    }

    #[test]
    fn replace_from_epic_graph_single_node_has_bounds() {
        let a_id = TaskId::from_bytes([1; 16]);
        let mut scene = GraphScene::empty_demo();
        let graph = redesmyn_protocol::client::EpicGraph {
            epic_slug: "demo".to_string(),
            nodes: vec![redesmyn_protocol::client::EpicTaskNode {
                task_slug: "T-1".to_string(),
                title: "First".to_string(),
                task_id: Some(a_id),
                parent_task_id: None,
                state: TaskState::Todo,
                branch_name: None,
                merge_readiness: MergeReadiness::Unknown,
            }],
            edges: Vec::new(),
            epic_id: None,
            epic_title: None,
            workspace_id: None,
            repo_id: None,
            command_summaries: Vec::new(),
            daemon_presences: Vec::new(),
            session_summaries: Vec::new(),
            as_of_event_id: None,
        };

        scene.replace_from_epic_graph(&graph);
        let bounds = scene.layout_bounds();
        assert!(bounds.size.width > 0);
        assert!(bounds.size.height > 0);
    }
}
