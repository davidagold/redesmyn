//! Deterministic layout for a rooted forest of variable-size nodes.
//!
//! v1 focuses on the primary structure we have today: a forest defined by a
//! `parent_id` pointer (tree edges). The layout is pure and deterministic:
//! identical inputs produce identical outputs.
//!
//! Call [`layout_forest`] whenever topology or node sizes change. v1 only uses
//! tree edges; non-tree edges (blockers, constraints) can be layered on later.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    #[must_use]
    pub const fn is_non_negative(self) -> bool {
        self.width >= 0 && self.height >= 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub origin: Point,
    pub size: Size,
}

impl Rect {
    #[must_use]
    pub const fn right(self) -> i32 {
        self.origin.x + self.size.width
    }

    #[must_use]
    pub const fn bottom(self) -> i32 {
        self.origin.y + self.size.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutConfig {
    /// Horizontal spacing between a parent node and its children.
    pub layer_spacing: i32,
    /// Vertical spacing between sibling subtrees.
    pub sibling_spacing: i32,
    /// Vertical spacing between separate root subtrees.
    pub root_spacing: i32,
    /// Top-left origin for the forest layout.
    pub origin: Point,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            // Mirrors the web constants (roughly); UI can override as needed.
            layer_spacing: 140,
            sibling_spacing: 160,
            root_spacing: 160,
            origin: Point { x: 0, y: 0 },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutNode<Id> {
    pub id: Id,
    pub parent_id: Option<Id>,
    pub size: Size,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutOutput<Id> {
    pub positions: BTreeMap<Id, Point>,
    pub bounds: Rect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError<Id> {
    DuplicateNodeId { id: Id },
    NegativeNodeSize { id: Id, size: Size },
    CycleDetected { id: Id },
}

#[must_use]
pub fn layout_forest<Id>(
    nodes: impl IntoIterator<Item = LayoutNode<Id>>,
    config: LayoutConfig,
) -> Result<LayoutOutput<Id>, LayoutError<Id>>
where
    Id: Copy + Ord + std::fmt::Debug,
{
    let input_nodes: Vec<LayoutNode<Id>> = nodes.into_iter().collect();
    let span = tracing::debug_span!(
        "graph_layout.layout_forest",
        node_count = input_nodes.len(),
        layer_spacing = config.layer_spacing,
        sibling_spacing = config.sibling_spacing,
        root_spacing = config.root_spacing,
    );
    let _guard = span.enter();

    let nodes_by_id = build_nodes_by_id(&input_nodes)?;
    let children_by_parent = build_children_by_parent(&nodes_by_id);
    detect_cycles(&nodes_by_id, &children_by_parent)?;

    let roots = roots_in_stable_order(&nodes_by_id);
    let subtree_heights = compute_subtree_heights(&nodes_by_id, &children_by_parent, config);

    let mut positions = BTreeMap::new();
    let mut next_root_top_y = config.origin.y;
    for root_id in roots {
        let subtree_height = subtree_heights.get(&root_id).copied().unwrap_or_default();
        layout_subtree(
            root_id,
            &nodes_by_id,
            &children_by_parent,
            &subtree_heights,
            config,
            config.origin.x,
            next_root_top_y,
            &mut positions,
        );
        next_root_top_y =
            next_root_top_y.saturating_add(subtree_height.saturating_add(config.root_spacing));
    }

    let bounds = compute_bounds(&nodes_by_id, &positions).unwrap_or(Rect {
        origin: config.origin,
        size: Size {
            width: 0,
            height: 0,
        },
    });

    Ok(LayoutOutput { positions, bounds })
}

fn build_nodes_by_id<Id: Copy + Ord>(
    input_nodes: &[LayoutNode<Id>],
) -> Result<BTreeMap<Id, LayoutNode<Id>>, LayoutError<Id>> {
    let mut nodes_by_id = BTreeMap::new();
    for node in input_nodes {
        if !node.size.is_non_negative() {
            return Err(LayoutError::NegativeNodeSize {
                id: node.id,
                size: node.size,
            });
        }
        if nodes_by_id.insert(node.id, *node).is_some() {
            return Err(LayoutError::DuplicateNodeId { id: node.id });
        }
    }
    Ok(nodes_by_id)
}

fn build_children_by_parent<Id: Copy + Ord + std::fmt::Debug>(
    nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>,
) -> BTreeMap<Id, Vec<Id>> {
    let mut children_by_parent: BTreeMap<Id, Vec<Id>> = BTreeMap::new();

    for node in nodes_by_id.values() {
        let Some(parent_id) = node.parent_id else {
            continue;
        };
        if !nodes_by_id.contains_key(&parent_id) {
            tracing::debug!(
                ?parent_id,
                node_id = ?node.id,
                "parent_id not found; treating node as root"
            );
            continue;
        }

        children_by_parent
            .entry(parent_id)
            .or_default()
            .push(node.id);
    }

    for children in children_by_parent.values_mut() {
        children.sort_unstable();
    }

    children_by_parent
}

fn roots_in_stable_order<Id: Copy + Ord>(nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>) -> Vec<Id> {
    let mut roots = Vec::new();
    for node in nodes_by_id.values() {
        match node.parent_id {
            None => roots.push(node.id),
            Some(parent_id) if !nodes_by_id.contains_key(&parent_id) => roots.push(node.id),
            Some(_) => {}
        }
    }
    roots
}

fn detect_cycles<Id: Copy + Ord + std::fmt::Debug>(
    nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>,
    children_by_parent: &BTreeMap<Id, Vec<Id>>,
) -> Result<(), LayoutError<Id>> {
    let mut state: BTreeMap<Id, VisitState> = BTreeMap::new();
    for node_id in nodes_by_id.keys().copied() {
        if state.contains_key(&node_id) {
            continue;
        }
        detect_cycles_dfs(node_id, children_by_parent, &mut state)?;
    }
    Ok(())
}

fn detect_cycles_dfs<Id: Copy + Ord + std::fmt::Debug>(
    node_id: Id,
    children_by_parent: &BTreeMap<Id, Vec<Id>>,
    state: &mut BTreeMap<Id, VisitState>,
) -> Result<(), LayoutError<Id>> {
    match state.get(&node_id).copied() {
        Some(VisitState::Visiting) => {
            tracing::debug!(node_id = ?node_id, "cycle detected");
            return Err(LayoutError::CycleDetected { id: node_id });
        }
        Some(VisitState::Visited) => return Ok(()),
        None => {}
    }

    state.insert(node_id, VisitState::Visiting);
    if let Some(children) = children_by_parent.get(&node_id) {
        for child_id in children {
            detect_cycles_dfs(*child_id, children_by_parent, state)?;
        }
    }
    state.insert(node_id, VisitState::Visited);
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Visiting,
    Visited,
}

fn compute_subtree_heights<Id: Copy + Ord>(
    nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>,
    children_by_parent: &BTreeMap<Id, Vec<Id>>,
    config: LayoutConfig,
) -> BTreeMap<Id, i32> {
    let mut memo: BTreeMap<Id, i32> = BTreeMap::new();
    for node_id in nodes_by_id.keys().copied() {
        compute_subtree_height(node_id, nodes_by_id, children_by_parent, config, &mut memo);
    }
    memo
}

fn compute_subtree_height<Id: Copy + Ord>(
    node_id: Id,
    nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>,
    children_by_parent: &BTreeMap<Id, Vec<Id>>,
    config: LayoutConfig,
    memo: &mut BTreeMap<Id, i32>,
) -> i32 {
    if let Some(height) = memo.get(&node_id).copied() {
        return height;
    }

    let node_height = nodes_by_id
        .get(&node_id)
        .map(|node| node.size.height)
        .unwrap_or_default();
    let Some(children) = children_by_parent.get(&node_id) else {
        memo.insert(node_id, node_height);
        return node_height;
    };

    let mut children_total_height = 0_i32;
    for (idx, child_id) in children.iter().enumerate() {
        let child_height =
            compute_subtree_height(*child_id, nodes_by_id, children_by_parent, config, memo);
        children_total_height = children_total_height.saturating_add(child_height);
        if idx + 1 < children.len() {
            children_total_height = children_total_height.saturating_add(config.sibling_spacing);
        }
    }

    let subtree_height = node_height.max(children_total_height);
    memo.insert(node_id, subtree_height);
    subtree_height
}

fn layout_subtree<Id: Copy + Ord>(
    node_id: Id,
    nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>,
    children_by_parent: &BTreeMap<Id, Vec<Id>>,
    subtree_heights: &BTreeMap<Id, i32>,
    config: LayoutConfig,
    x: i32,
    subtree_top_y: i32,
    positions: &mut BTreeMap<Id, Point>,
) {
    let Some(node) = nodes_by_id.get(&node_id).copied() else {
        return;
    };

    let subtree_height = subtree_heights.get(&node_id).copied().unwrap_or_default();
    let node_y = subtree_top_y.saturating_add((subtree_height - node.size.height) / 2);
    positions.insert(node_id, Point { x, y: node_y });

    let Some(children) = children_by_parent.get(&node_id) else {
        return;
    };
    if children.is_empty() {
        return;
    }

    let mut children_total_height = 0_i32;
    for (idx, child_id) in children.iter().enumerate() {
        let child_subtree_height = subtree_heights.get(child_id).copied().unwrap_or_default();
        children_total_height = children_total_height.saturating_add(child_subtree_height);
        if idx + 1 < children.len() {
            children_total_height = children_total_height.saturating_add(config.sibling_spacing);
        }
    }

    let mut next_child_top_y =
        subtree_top_y.saturating_add((subtree_height - children_total_height) / 2);
    let child_x = x.saturating_add(node.size.width.saturating_add(config.layer_spacing));
    for child_id in children {
        let child_subtree_height = subtree_heights.get(child_id).copied().unwrap_or_default();
        layout_subtree(
            *child_id,
            nodes_by_id,
            children_by_parent,
            subtree_heights,
            config,
            child_x,
            next_child_top_y,
            positions,
        );
        next_child_top_y = next_child_top_y
            .saturating_add(child_subtree_height.saturating_add(config.sibling_spacing));
    }
}

fn compute_bounds<Id: Copy + Ord>(
    nodes_by_id: &BTreeMap<Id, LayoutNode<Id>>,
    positions: &BTreeMap<Id, Point>,
) -> Option<Rect> {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for (node_id, position) in positions {
        let Some(node) = nodes_by_id.get(node_id) else {
            continue;
        };
        min_x = min_x.min(position.x);
        min_y = min_y.min(position.y);
        max_x = max_x.max(position.x.saturating_add(node.size.width));
        max_y = max_y.max(position.y.saturating_add(node.size.height));
    }

    if min_x == i32::MAX {
        return None;
    }

    Some(Rect {
        origin: Point { x: min_x, y: min_y },
        size: Size {
            width: max_x.saturating_sub(min_x),
            height: max_y.saturating_sub(min_y),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_intersects(a_origin: Point, a_size: Size, b_origin: Point, b_size: Size) -> bool {
        let a = Rect {
            origin: a_origin,
            size: a_size,
        };
        let b = Rect {
            origin: b_origin,
            size: b_size,
        };

        let a_left = a.origin.x;
        let a_right = a.right();
        let a_top = a.origin.y;
        let a_bottom = a.bottom();

        let b_left = b.origin.x;
        let b_right = b.right();
        let b_top = b.origin.y;
        let b_bottom = b.bottom();

        a_left < b_right && a_right > b_left && a_top < b_bottom && a_bottom > b_top
    }

    #[test]
    fn lays_out_a_tree_with_variable_sizes_deterministically() {
        let config = LayoutConfig {
            layer_spacing: 20,
            sibling_spacing: 10,
            root_spacing: 15,
            origin: Point { x: 0, y: 0 },
        };

        let nodes = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: None,
                size: Size {
                    width: 100,
                    height: 50,
                },
            },
            LayoutNode {
                id: 2_u32,
                parent_id: Some(1),
                size: Size {
                    width: 80,
                    height: 40,
                },
            },
            LayoutNode {
                id: 3_u32,
                parent_id: Some(1),
                size: Size {
                    width: 120,
                    height: 60,
                },
            },
            LayoutNode {
                id: 4_u32,
                parent_id: Some(2),
                size: Size {
                    width: 70,
                    height: 30,
                },
            },
        ];

        let output = layout_forest(nodes, config).expect("layout should succeed");

        let expected_positions: BTreeMap<u32, Point> = [
            (1, Point { x: 0, y: 30 }),
            (2, Point { x: 120, y: 0 }),
            (3, Point { x: 120, y: 50 }),
            (4, Point { x: 220, y: 5 }),
        ]
        .into_iter()
        .collect();

        assert_eq!(output.positions, expected_positions);
        assert_eq!(
            output.bounds,
            Rect {
                origin: Point { x: 0, y: 0 },
                size: Size {
                    width: 290,
                    height: 110
                }
            }
        );
    }

    #[test]
    fn output_is_stable_across_input_ordering() {
        let config = LayoutConfig {
            layer_spacing: 20,
            sibling_spacing: 10,
            root_spacing: 15,
            origin: Point { x: 0, y: 0 },
        };

        let nodes_a = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: None,
                size: Size {
                    width: 100,
                    height: 50,
                },
            },
            LayoutNode {
                id: 2_u32,
                parent_id: Some(1),
                size: Size {
                    width: 80,
                    height: 40,
                },
            },
            LayoutNode {
                id: 3_u32,
                parent_id: Some(1),
                size: Size {
                    width: 120,
                    height: 60,
                },
            },
            LayoutNode {
                id: 4_u32,
                parent_id: Some(2),
                size: Size {
                    width: 70,
                    height: 30,
                },
            },
        ];

        let nodes_b = vec![nodes_a[2], nodes_a[3], nodes_a[1], nodes_a[0]];

        let output_a = layout_forest(nodes_a.clone(), config).expect("layout should succeed");
        let output_a_second_run = layout_forest(nodes_a, config).expect("layout should succeed");
        let output_b = layout_forest(nodes_b, config).expect("layout should succeed");
        assert_eq!(output_a, output_a_second_run);
        assert_eq!(output_a, output_b);
    }

    #[test]
    fn does_not_produce_overlapping_nodes_for_a_simple_forest() {
        let config = LayoutConfig {
            layer_spacing: 30,
            sibling_spacing: 12,
            root_spacing: 40,
            origin: Point { x: 0, y: 0 },
        };

        let nodes = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: None,
                size: Size {
                    width: 100,
                    height: 50,
                },
            },
            LayoutNode {
                id: 2_u32,
                parent_id: Some(1),
                size: Size {
                    width: 80,
                    height: 120,
                },
            },
            LayoutNode {
                id: 3_u32,
                parent_id: Some(1),
                size: Size {
                    width: 60,
                    height: 40,
                },
            },
            LayoutNode {
                id: 10_u32,
                parent_id: None,
                size: Size {
                    width: 90,
                    height: 90,
                },
            },
            LayoutNode {
                id: 11_u32,
                parent_id: Some(10),
                size: Size {
                    width: 40,
                    height: 30,
                },
            },
        ];

        let sizes_by_id: BTreeMap<u32, Size> =
            nodes.iter().map(|node| (node.id, node.size)).collect();
        let output = layout_forest(nodes, config).expect("layout should succeed");
        let mut ids: Vec<u32> = output.positions.keys().copied().collect();
        ids.sort_unstable();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a_id = ids[i];
                let b_id = ids[j];
                let a_node = output.positions.get(&a_id).copied().unwrap();
                let b_node = output.positions.get(&b_id).copied().unwrap();
                let a_size = sizes_by_id[&a_id];
                let b_size = sizes_by_id[&b_id];

                assert!(
                    !rect_intersects(a_node, a_size, b_node, b_size),
                    "nodes {a_id} and {b_id} overlap: {a_node:?} {a_size:?} vs {b_node:?} {b_size:?}",
                );
            }
        }
    }

    #[test]
    fn rejects_cycles() {
        let nodes = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: Some(2),
                size: Size {
                    width: 10,
                    height: 10,
                },
            },
            LayoutNode {
                id: 2_u32,
                parent_id: Some(1),
                size: Size {
                    width: 10,
                    height: 10,
                },
            },
        ];

        let err = layout_forest(nodes, LayoutConfig::default()).expect_err("should detect cycle");
        assert_eq!(err, LayoutError::CycleDetected { id: 1 });
    }
}
