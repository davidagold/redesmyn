use std::collections::BTreeMap;
use std::fmt;

use gpui::SharedString;
use redesmyn_graph_layout::{layout_forest, LayoutConfig, LayoutNode};
use redesmyn_ids::TaskId;

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
    pub title: SharedString,
    pub parent_id: Option<GraphNodeId>,
}

#[derive(Debug, Clone)]
pub struct GraphSceneEdge {
    pub id: GraphEdgeId,
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
            title: "T-1 Root".into(),
            parent_id: None,
        });
        scene.insert_node(GraphSceneNode {
            id: b,
            title: "T-2 Child A".into(),
            parent_id: Some(a),
        });
        scene.insert_node(GraphSceneNode {
            id: c,
            title: "T-3 Child B".into(),
            parent_id: Some(a),
        });
        scene.insert_node(GraphSceneNode {
            id: d,
            title: "T-4 Grandchild".into(),
            parent_id: Some(b),
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
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_demo_node(&mut self, id: GraphNodeId) {
        self.insert_node(GraphSceneNode {
            id,
            title: "demo".into(),
            parent_id: None,
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
        self.selection.selected_edge = None;
        self.selection.selected_node = Some(id);
    }

    pub fn select_edge(&mut self, id: GraphEdgeId) {
        self.selection.selected_node = None;
        self.selection.selected_edge = Some(id);
    }

    pub fn clear_selection(&mut self) {
        self.selection = GraphSelection::default();
    }

    pub fn replace_from_epic_graph(&mut self, graph: &redesmyn_protocol::client::EpicGraph) {
        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.graph_scene.replace_from_epic_graph",
            epic_slug = %graph.epic_slug,
            node_count = graph.nodes.len(),
            edge_count = graph.edges.len(),
        );
        let _guard = span.enter();

        // TEMPORARY: `EpicGraph` currently identifies tasks by slug (string) instead of a stable
        // protocol id (e.g. ULID-backed `TaskId`). Hash slugs into deterministic IDs so node/edge
        // IDs remain stable across refreshes until the protocol exports real identifiers.
        let mut nodes = BTreeMap::new();
        let mut parent_by_child = BTreeMap::new();

        let mut edges = BTreeMap::new();
        for edge in &graph.edges {
            let from = GraphNodeId::Task(temporary_task_id_for_slug(&edge.from_task_slug));
            let to = GraphNodeId::Task(temporary_task_id_for_slug(&edge.to_task_slug));
            let edge_id = GraphEdgeId { from, to };
            edges.insert(edge_id, GraphSceneEdge { id: edge_id });
            parent_by_child.entry(to).or_insert(from);
        }

        for node in &graph.nodes {
            let id = GraphNodeId::Task(temporary_task_id_for_slug(&node.task_slug));
            nodes.insert(
                id,
                GraphSceneNode {
                    id,
                    title: SharedString::new(node.title.clone()),
                    parent_id: parent_by_child.get(&id).copied(),
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
        let nodes = self
            .nodes
            .values()
            .map(|node| LayoutNode {
                id: node.id,
                parent_id: node.parent_id,
                size: self
                    .node_sizes
                    .get(&node.id)
                    .copied()
                    .unwrap_or_else(default_node_size),
            })
            .collect::<Vec<_>>();

        let config = LayoutConfig {
            origin: redesmyn_graph_layout::Point { x: 40, y: 40 },
            ..LayoutConfig::default()
        };

        match layout_forest(nodes, config) {
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
    redesmyn_graph_layout::Size {
        width: 320,
        height: 96,
    }
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
}
