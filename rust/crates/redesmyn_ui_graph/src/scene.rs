use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use gpui::SharedString;
use redesmyn_graph_layout::{LayoutConfig, LayoutNode, layout_forest};
use redesmyn_ids::TaskId;
use redesmyn_protocol::client::{CommandState, MergeReadiness, TaskState};

pub(crate) const COLLAPSED_TASK_NODE_SIZE: redesmyn_graph_layout::Size =
    redesmyn_graph_layout::Size {
        width: 320,
        height: 96,
    };

pub(crate) const EXPANDED_TASK_NODE_SIZE: redesmyn_graph_layout::Size =
    redesmyn_graph_layout::Size {
        width: 960,
        height: 560,
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
    pub title: SharedString,
    pub parent_id: Option<GraphNodeId>,
    pub state: TaskState,
    pub merge_readiness: MergeReadiness,
    pub agent_status: AgentStatus,
    pub branch_name: Option<SharedString>,
}

#[derive(Debug, Clone)]
pub struct GraphSceneEdge {
    pub id: GraphEdgeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Unknown,
    Running,
    Blocked,
    Stopped,
    Error,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GraphSelection {
    pub selected_node: Option<GraphNodeId>,
    pub selected_edge: Option<GraphEdgeId>,
}

#[derive(Debug, Clone)]
pub struct GraphScene {
    nodes: BTreeMap<GraphNodeId, GraphSceneNode>,
    edges: BTreeMap<GraphEdgeId, GraphSceneEdge>,
    selection: GraphSelection,
    node_sizes: BTreeMap<GraphNodeId, redesmyn_graph_layout::Size>,
    node_positions: BTreeMap<GraphNodeId, redesmyn_graph_layout::Point>,
    bounds: redesmyn_graph_layout::Rect,
    layout_nodes_scratch: Vec<LayoutNode<GraphNodeId>>,
}

impl GraphScene {
    #[must_use]
    pub fn demo() -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_scene.demo_build");
        let _guard = span.enter();

        let mut scene = Self::empty_demo();
        let a = GraphNodeId::Task(TaskId::from_bytes([1; 16]));
        let b = GraphNodeId::Task(TaskId::from_bytes([2; 16]));
        let c = GraphNodeId::Task(TaskId::from_bytes([3; 16]));
        let d = GraphNodeId::Task(TaskId::from_bytes([4; 16]));

        scene.insert_node(GraphSceneNode {
            id: a,
            task_slug: "T-1".into(),
            title: "T-1 Root".into(),
            parent_id: None,
            state: TaskState::InProgress,
            merge_readiness: MergeReadiness::Unknown,
            agent_status: AgentStatus::Running,
            branch_name: Some("feat/root".into()),
        });
        scene.insert_node(GraphSceneNode {
            id: b,
            task_slug: "T-2".into(),
            title: "T-2 Child A".into(),
            parent_id: Some(a),
            state: TaskState::Blocked,
            merge_readiness: MergeReadiness::Blocked,
            agent_status: AgentStatus::Blocked,
            branch_name: Some("feat/child-a".into()),
        });
        scene.insert_node(GraphSceneNode {
            id: c,
            task_slug: "T-3".into(),
            title: "T-3 Child B".into(),
            parent_id: Some(a),
            state: TaskState::Todo,
            merge_readiness: MergeReadiness::Ready,
            agent_status: AgentStatus::Stopped,
            branch_name: Some("feat/child-b".into()),
        });
        scene.insert_node(GraphSceneNode {
            id: d,
            task_slug: "T-4".into(),
            title: "T-4 Grandchild".into(),
            parent_id: Some(b),
            state: TaskState::Done,
            merge_readiness: MergeReadiness::Ready,
            agent_status: AgentStatus::Stopped,
            branch_name: Some("feat/grandchild".into()),
        });

        scene.insert_edge(GraphEdgeId { from: a, to: b });
        scene.insert_edge(GraphEdgeId { from: a, to: c });
        scene.insert_edge(GraphEdgeId { from: b, to: d });

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
            layout_nodes_scratch: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_demo_node(&mut self, id: GraphNodeId) {
        self.insert_node(GraphSceneNode {
            id,
            task_slug: "demo".into(),
            title: "demo".into(),
            parent_id: None,
            state: TaskState::Unknown,
            merge_readiness: MergeReadiness::Unknown,
            agent_status: AgentStatus::Unknown,
            branch_name: None,
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

        let sizes_changed = self.apply_expandedness_policy();
        if selection_changed || sizes_changed {
            self.relayout();
        }
    }

    fn apply_expandedness_policy(&mut self) -> bool {
        self.node_sizes.retain(|id, _| self.nodes.contains_key(id));

        let expanded = self.selection.selected_node;
        let mut changed = false;

        for id in self.nodes.keys().copied() {
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

    pub fn replace_from_epic_graph(&mut self, graph: &redesmyn_protocol::client::EpicGraph) {
        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.graph_scene.replace_from_epic_graph",
            epic_slug = %graph.epic_slug,
            node_count = graph.nodes.len(),
            edge_count = graph.edges.len(),
        );
        let _guard = span.enter();

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
            edges.insert(edge_id, GraphSceneEdge { id: edge_id });
            parent_by_child.entry(to).or_insert(from);
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
                    title: SharedString::new(node.title.clone()),
                    parent_id,
                    state: node.state,
                    merge_readiness: node.merge_readiness,
                    agent_status: node
                        .task_id
                        .and_then(|id| agent_status_by_task_id.get(&id).copied())
                        .unwrap_or(AgentStatus::Unknown),
                    branch_name: node.branch_name.as_ref().map(|name| SharedString::new(name.clone())),
                },
            );
        }

        self.nodes = nodes;
        self.edges = edges;

        if let Some(selected) = self.selection.selected_node
            && !self.nodes.contains_key(&selected)
        {
            self.selection.selected_node = None;
        }

        if let Some(selected) = self.selection.selected_edge
            && !self.edges.contains_key(&selected)
        {
            self.selection.selected_edge = None;
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
    }

    fn insert_edge(&mut self, id: GraphEdgeId) {
        self.edges.insert(id, GraphSceneEdge { id });
    }

    fn relayout(&mut self) {
        self.layout_nodes_scratch.clear();
        self.layout_nodes_scratch.reserve(
            self.nodes
                .len()
                .saturating_sub(self.layout_nodes_scratch.capacity()),
        );

        for node in self.nodes.values() {
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

        let config = LayoutConfig {
            origin: redesmyn_graph_layout::Point { x: 40, y: 40 },
            ..LayoutConfig::default()
        };

        match layout_forest(self.layout_nodes_scratch.iter().copied(), config) {
            Ok(output) => {
                self.node_positions = output.positions;
                self.bounds = output.bounds;
            }
            Err(error) => {
                redesmyn_logging::tracing::error!(
                    ?error,
                    "graph layout failed; keeping previous positions"
                );
            }
        }
    }
}

fn default_node_size() -> redesmyn_graph_layout::Size {
    COLLAPSED_TASK_NODE_SIZE
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
            CommandState::Running | CommandState::Accepted => AgentStatus::Running,
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
        let ids: Vec<_> = scene.nodes().map(|node| node.id).collect();
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
}
