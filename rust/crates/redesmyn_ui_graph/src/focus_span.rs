use std::collections::{BTreeMap, BTreeSet};

use crate::scene::{GraphNodeId, GraphSceneNode};

/// Compute the focus span path for "focus mode".
///
/// The path is:
/// - `root → ... → selected` (walk parents),
/// - then `selected → ...` while each node has exactly one child.
///
/// Returns `None` if `selected_node_id` is not present or the upstream walk fails to reach its
/// detected root (e.g. a cycle).
pub(crate) fn compute_focus_span(
    selected_node_id: GraphNodeId,
    nodes_by_id: &BTreeMap<GraphNodeId, GraphSceneNode>,
) -> Option<Vec<GraphNodeId>> {
    if !nodes_by_id.contains_key(&selected_node_id) {
        return None;
    }

    let children_by_parent = build_children_by_parent(nodes_by_id);

    // Find the upstream root for the selected node, bailing on cycles.
    let mut visited_upstream = BTreeSet::new();
    let mut root_id = selected_node_id;
    while visited_upstream.insert(root_id) {
        let parent_id = nodes_by_id.get(&root_id).and_then(|node| node.parent_id);
        let Some(parent_id) = parent_id else {
            break;
        };
        root_id = parent_id;
    }

    // Walk from selected → root using the parent pointers.
    let mut focus_path_up = Vec::new();
    let mut visited_path = BTreeSet::new();
    let mut cursor = Some(selected_node_id);
    while let Some(node_id) = cursor {
        if !visited_path.insert(node_id) {
            break;
        }

        focus_path_up.push(node_id);
        if node_id == root_id {
            break;
        }

        cursor = nodes_by_id.get(&node_id).and_then(|node| node.parent_id);
    }

    if focus_path_up.is_empty() || focus_path_up.last().copied() != Some(root_id) {
        return None;
    }

    focus_path_up.reverse();

    // Walk downstream from selected while the chain remains single-child.
    let mut visited_downstream: BTreeSet<GraphNodeId> = focus_path_up.iter().copied().collect();
    let mut focus_path_down = Vec::new();
    let mut down_id = selected_node_id;
    loop {
        let children = children_by_parent
            .get(&down_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if children.len() != 1 {
            break;
        }

        let next_id = children[0];
        if !visited_downstream.insert(next_id) {
            break;
        }

        focus_path_down.push(next_id);
        down_id = next_id;
    }

    focus_path_up.extend(focus_path_down);
    Some(focus_path_up)
}

fn build_children_by_parent(
    nodes_by_id: &BTreeMap<GraphNodeId, GraphSceneNode>,
) -> BTreeMap<GraphNodeId, Vec<GraphNodeId>> {
    let mut children_by_parent: BTreeMap<GraphNodeId, Vec<GraphNodeId>> = BTreeMap::new();
    for (id, node) in nodes_by_id {
        let Some(parent_id) = node.parent_id else {
            continue;
        };
        children_by_parent.entry(parent_id).or_default().push(*id);
    }

    // Preserve determinism even if upstream data isn't sorted.
    for children in children_by_parent.values_mut() {
        children.sort_unstable();
    }

    children_by_parent
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::AgentStatus;
    use redesmyn_ids::TaskId;
    use redesmyn_protocol::client::{MergeReadiness, TaskState};

    fn task(bytes: u8) -> GraphNodeId {
        GraphNodeId::Task(TaskId::from_bytes([bytes; 16]))
    }

    fn insert(
        nodes: &mut BTreeMap<GraphNodeId, GraphSceneNode>,
        id: GraphNodeId,
        parent: Option<GraphNodeId>,
    ) {
        nodes.insert(
            id,
            GraphSceneNode {
                id,
                task_slug: "demo".into(),
                title: "demo".into(),
                parent_id: parent,
                state: TaskState::Unknown,
                merge_readiness: MergeReadiness::Unknown,
                agent_status: AgentStatus::Unknown,
                branch_name: None,
            },
        );
    }

    #[test]
    fn focus_span_includes_root_selected_and_single_child_chain() {
        // root -> a -> b -> c
        let root = task(1);
        let a = task(2);
        let b = task(3);
        let c = task(4);

        let mut nodes = BTreeMap::new();
        insert(&mut nodes, root, None);
        insert(&mut nodes, a, Some(root));
        insert(&mut nodes, b, Some(a));
        insert(&mut nodes, c, Some(b));

        assert_eq!(compute_focus_span(b, &nodes), Some(vec![root, a, b, c]));
    }

    #[test]
    fn focus_span_stops_when_multiple_children() {
        // root -> a and root -> b
        let root = task(1);
        let a = task(2);
        let b = task(3);

        let mut nodes = BTreeMap::new();
        insert(&mut nodes, root, None);
        insert(&mut nodes, a, Some(root));
        insert(&mut nodes, b, Some(root));

        assert_eq!(compute_focus_span(root, &nodes), Some(vec![root]));
    }

    #[test]
    fn focus_span_is_none_when_selected_missing() {
        let nodes = BTreeMap::new();
        assert_eq!(compute_focus_span(task(1), &nodes), None);
    }

    #[test]
    fn focus_span_does_not_loop_for_parent_cycle() {
        // a -> b -> a
        let a = task(1);
        let b = task(2);

        let mut nodes = BTreeMap::new();
        insert(&mut nodes, a, Some(b));
        insert(&mut nodes, b, Some(a));

        assert_eq!(compute_focus_span(a, &nodes), Some(vec![a, b]));
    }
}
