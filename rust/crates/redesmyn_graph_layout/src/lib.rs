//! Deterministic layout for a rooted forest of variable-size nodes.
//!
//! v1 focuses on the primary structure we have today: a forest defined by a
//! `parent_id` pointer (tree edges). The layout is pure and deterministic:
//! identical inputs produce identical outputs.
//!
//! For high-performance use (e.g. per-frame expand/collapse relayout), prefer
//! [`ForestLayoutEngine`]: build topology once, update sizes cheaply, and rerun
//! layout without heap allocations.
//!
//! [`layout_forest`] remains available as a convenience wrapper that allocates
//! and rebuilds topology each call.

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::collections::BTreeMap;
use std::ops::Range;

/// A 2D point in logical pixels.
///
/// In [`LayoutOutput`], points represent the top-left origin of a node's rectangle.
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

/// How to treat a node whose `parent_id` does not appear in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnknownParentPolicy {
    /// Treat the node as a root (useful for partial / filtered forests).
    TreatAsRoot,
    /// Reject the input as invalid.
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutOptions {
    /// Input validation / interpretation policy.
    pub unknown_parent_policy: UnknownParentPolicy,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            unknown_parent_policy: UnknownParentPolicy::TreatAsRoot,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutOutputView<'a, Id> {
    /// Stable node order (sorted by id).
    ///
    /// The position for `ids[i]` is `positions[i]`.
    pub ids: &'a [Id],
    pub positions: &'a [Point],
    pub bounds: Rect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError<Id> {
    DuplicateNodeId { id: Id },
    NegativeNodeSize { id: Id, size: Size },
    UnknownNodeId { id: Id },
    UnknownParentId { id: Id, parent_id: Id },
    CycleDetected { id: Id },
}

/// A topology-cached, allocation-free-per-layout forest layout engine.
///
/// `new(...)` builds and validates topology once. Subsequent calls to
/// [`Self::layout`] reuse internal buffers and perform no heap allocations.
#[derive(Debug, Clone)]
pub struct ForestLayoutEngine<Id> {
    /// Stable node order (sorted by id).
    ids: Vec<Id>,
    /// Lookup-only map (determinism does not depend on iteration).
    index_by_id: BTreeMap<Id, usize>,

    parent_index: Vec<Option<usize>>,
    roots: Vec<usize>,

    children: Vec<usize>,
    child_range: Vec<Range<usize>>,

    postorder: Vec<usize>,

    sizes: Vec<Size>,

    subtree_heights: Vec<i32>,
    children_block_heights: Vec<i32>,
    positions: Vec<Point>,
    bounds: Rect,

    layout_stack: Vec<LayoutFrame>,
}

#[derive(Debug, Clone, Copy)]
struct LayoutFrame {
    node_index: usize,
    next_child_pos: usize,
    next_child_top_y: i32,
    child_x: i32,
}

impl<Id> ForestLayoutEngine<Id>
where
    Id: Copy + Ord + std::fmt::Debug,
{
    pub fn new(
        nodes: impl IntoIterator<Item = LayoutNode<Id>>,
        options: LayoutOptions,
    ) -> Result<Self, LayoutError<Id>> {
        let mut input_nodes: Vec<LayoutNode<Id>> = nodes.into_iter().collect();
        let span = redesmyn_logging::tracing::debug_span!(
            "graph_layout.engine_new",
            node_count = input_nodes.len(),
            unknown_parent_policy = ?options.unknown_parent_policy,
        );
        let _guard = span.enter();

        validate_node_sizes(&input_nodes)?;
        input_nodes.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        validate_unique_ids(&input_nodes)?;

        let node_count = input_nodes.len();
        let mut ids = Vec::with_capacity(node_count);
        let mut sizes = Vec::with_capacity(node_count);
        for node in &input_nodes {
            ids.push(node.id);
            sizes.push(node.size);
        }

        let index_by_id = build_index_by_id(&ids);
        let parent_index = build_parent_index(&input_nodes, &index_by_id, options)?;
        let roots = roots_in_stable_order(&parent_index);

        let (children, child_range) = build_children_adjacency(&parent_index, node_count);
        detect_cycles(&ids, &children, &child_range)?;

        let postorder = build_postorder(&roots, &children, &child_range, node_count);

        Ok(Self {
            ids,
            index_by_id,
            parent_index,
            roots,
            children,
            child_range,
            postorder,
            sizes,
            subtree_heights: vec![0; node_count],
            children_block_heights: vec![0; node_count],
            positions: vec![Point { x: 0, y: 0 }; node_count],
            bounds: Rect {
                origin: Point { x: 0, y: 0 },
                size: Size {
                    width: 0,
                    height: 0,
                },
            },
            layout_stack: Vec::with_capacity(node_count),
        })
    }

    #[must_use]
    pub fn ids(&self) -> &[Id] {
        &self.ids
    }

    #[must_use]
    pub fn sizes(&self) -> &[Size] {
        &self.sizes
    }

    #[must_use]
    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    #[must_use]
    pub fn parent_index(&self) -> &[Option<usize>] {
        &self.parent_index
    }

    #[must_use]
    pub fn positions(&self) -> &[Point] {
        &self.positions
    }

    pub fn set_size(&mut self, id: Id, size: Size) -> Result<(), LayoutError<Id>> {
        if !size.is_non_negative() {
            return Err(LayoutError::NegativeNodeSize { id, size });
        }

        let Some(&index) = self.index_by_id.get(&id) else {
            return Err(LayoutError::UnknownNodeId { id });
        };

        self.sizes[index] = size;
        Ok(())
    }

    pub fn set_sizes(
        &mut self,
        sizes: impl IntoIterator<Item = (Id, Size)>,
    ) -> Result<(), LayoutError<Id>> {
        for (id, size) in sizes {
            self.set_size(id, size)?;
        }
        Ok(())
    }

    pub fn layout(&mut self, config: LayoutConfig) -> LayoutOutputView<'_, Id> {
        self.recompute_subtree_heights(config);
        self.recompute_positions(config);
        self.bounds = compute_bounds(self.sizes.iter().copied(), self.positions.iter().copied())
            .unwrap_or(Rect {
                origin: config.origin,
                size: Size {
                    width: 0,
                    height: 0,
                },
            });

        LayoutOutputView {
            ids: &self.ids,
            positions: &self.positions,
            bounds: self.bounds,
        }
    }

    fn recompute_subtree_heights(&mut self, config: LayoutConfig) {
        for &node_index in &self.postorder {
            let range = self.child_range[node_index].clone();
            let mut children_total_height = 0_i32;
            let mut child_count = 0_usize;
            for &child_index in &self.children[range] {
                if child_count > 0 {
                    children_total_height =
                        children_total_height.saturating_add(config.sibling_spacing);
                }
                children_total_height =
                    children_total_height.saturating_add(self.subtree_heights[child_index]);
                child_count += 1;
            }

            self.children_block_heights[node_index] = children_total_height;
            let node_height = self.sizes[node_index].height;
            self.subtree_heights[node_index] = node_height.max(children_total_height);
        }
    }

    fn recompute_positions(&mut self, config: LayoutConfig) {
        self.layout_stack.clear();

        let roots_len = self.roots.len();
        let mut next_root_top_y = config.origin.y;
        for root_slot in 0..roots_len {
            let root_index = self.roots[root_slot];
            self.push_layout_frame(root_index, config.origin.x, next_root_top_y, config);

            while !self.layout_stack.is_empty() {
                let next_child = {
                    let frame = self.layout_stack.last_mut().expect("not empty");
                    let node_index = frame.node_index;
                    let child_end = self.child_range[node_index].end;
                    if frame.next_child_pos >= child_end {
                        None
                    } else {
                        let child_index = self.children[frame.next_child_pos];
                        frame.next_child_pos += 1;
                        let child_x = frame.child_x;
                        let child_subtree_top_y = frame.next_child_top_y;
                        frame.next_child_top_y = frame.next_child_top_y.saturating_add(
                            self.subtree_heights[child_index]
                                .saturating_add(config.sibling_spacing),
                        );
                        Some((child_index, child_x, child_subtree_top_y))
                    }
                };

                match next_child {
                    Some((child_index, child_x, child_subtree_top_y)) => {
                        self.push_layout_frame(child_index, child_x, child_subtree_top_y, config);
                    }
                    None => {
                        self.layout_stack.pop();
                    }
                }
            }

            next_root_top_y = next_root_top_y.saturating_add(
                self.subtree_heights[root_index].saturating_add(config.root_spacing),
            );
        }
    }

    fn push_layout_frame(
        &mut self,
        node_index: usize,
        x: i32,
        subtree_top_y: i32,
        config: LayoutConfig,
    ) {
        let subtree_height = self.subtree_heights[node_index];
        let node_height = self.sizes[node_index].height;
        let node_y = subtree_top_y.saturating_add((subtree_height - node_height) / 2);
        self.positions[node_index] = Point { x, y: node_y };

        let range = self.child_range[node_index].clone();
        if range.start == range.end {
            return;
        }

        let children_total_height = self.children_block_heights[node_index];
        let next_child_top_y =
            subtree_top_y.saturating_add((subtree_height - children_total_height) / 2);
        let child_x = x.saturating_add(
            self.sizes[node_index]
                .width
                .saturating_add(config.layer_spacing),
        );

        self.layout_stack.push(LayoutFrame {
            node_index,
            next_child_pos: range.start,
            next_child_top_y,
            child_x,
        });
    }
}

/// Computes a deterministic layout for a rooted forest.
///
/// Nodes whose `parent_id` is not present in the input are treated as roots. Use
/// [`layout_forest_with_options`] with [`UnknownParentPolicy::Error`] to reject such inputs.
#[must_use]
pub fn layout_forest<Id>(
    nodes: impl IntoIterator<Item = LayoutNode<Id>>,
    config: LayoutConfig,
) -> Result<LayoutOutput<Id>, LayoutError<Id>>
where
    Id: Copy + Ord + std::fmt::Debug,
{
    layout_forest_with_options(nodes, config, LayoutOptions::default())
}

#[must_use]
pub fn layout_forest_with_options<Id>(
    nodes: impl IntoIterator<Item = LayoutNode<Id>>,
    config: LayoutConfig,
    options: LayoutOptions,
) -> Result<LayoutOutput<Id>, LayoutError<Id>>
where
    Id: Copy + Ord + std::fmt::Debug,
{
    let mut engine = ForestLayoutEngine::new(nodes, options)?;
    let view = engine.layout(config);
    let mut positions = BTreeMap::new();
    for (id, point) in view.ids.iter().copied().zip(view.positions.iter().copied()) {
        positions.insert(id, point);
    }
    Ok(LayoutOutput {
        positions,
        bounds: view.bounds,
    })
}

fn validate_node_sizes<Id: Copy>(nodes: &[LayoutNode<Id>]) -> Result<(), LayoutError<Id>> {
    for node in nodes {
        if !node.size.is_non_negative() {
            return Err(LayoutError::NegativeNodeSize {
                id: node.id,
                size: node.size,
            });
        }
    }
    Ok(())
}

fn validate_unique_ids<Id: Copy + Ord>(
    nodes_sorted_by_id: &[LayoutNode<Id>],
) -> Result<(), LayoutError<Id>> {
    for window in nodes_sorted_by_id.windows(2) {
        if window[0].id == window[1].id {
            return Err(LayoutError::DuplicateNodeId { id: window[0].id });
        }
    }
    Ok(())
}

fn build_index_by_id<Id: Copy + Ord>(ids: &[Id]) -> BTreeMap<Id, usize> {
    let mut index_by_id = BTreeMap::new();
    for (index, id) in ids.iter().copied().enumerate() {
        index_by_id.insert(id, index);
    }
    index_by_id
}

fn build_parent_index<Id: Copy + Ord + std::fmt::Debug>(
    nodes_sorted_by_id: &[LayoutNode<Id>],
    index_by_id: &BTreeMap<Id, usize>,
    options: LayoutOptions,
) -> Result<Vec<Option<usize>>, LayoutError<Id>> {
    let mut parent_index = Vec::with_capacity(nodes_sorted_by_id.len());
    for node in nodes_sorted_by_id {
        let Some(parent_id) = node.parent_id else {
            parent_index.push(None);
            continue;
        };

        match index_by_id.get(&parent_id).copied() {
            Some(parent) => parent_index.push(Some(parent)),
            None => match options.unknown_parent_policy {
                UnknownParentPolicy::TreatAsRoot => {
                    redesmyn_logging::tracing::debug!(
                        ?parent_id,
                        node_id = ?node.id,
                        "parent_id not found; treating node as root"
                    );
                    parent_index.push(None);
                }
                UnknownParentPolicy::Error => {
                    return Err(LayoutError::UnknownParentId {
                        id: node.id,
                        parent_id,
                    });
                }
            },
        }
    }
    Ok(parent_index)
}

fn roots_in_stable_order(parent_index: &[Option<usize>]) -> Vec<usize> {
    let mut roots = Vec::new();
    for (index, parent) in parent_index.iter().copied().enumerate() {
        if parent.is_none() {
            roots.push(index);
        }
    }
    roots
}

fn build_children_adjacency(
    parent_index: &[Option<usize>],
    node_count: usize,
) -> (Vec<usize>, Vec<Range<usize>>) {
    let mut child_counts = vec![0_usize; node_count];
    for parent in parent_index.iter().copied().flatten() {
        child_counts[parent] += 1;
    }

    let mut child_range = Vec::with_capacity(node_count);
    let mut total_children = 0_usize;
    for count in &child_counts {
        let start = total_children;
        total_children += *count;
        child_range.push(start..total_children);
    }

    let mut next_child_slot: Vec<usize> = child_range.iter().map(|range| range.start).collect();
    let mut children = vec![0_usize; total_children];

    for (child_index, parent) in parent_index.iter().copied().enumerate() {
        let Some(parent_index) = parent else {
            continue;
        };
        let slot = &mut next_child_slot[parent_index];
        children[*slot] = child_index;
        *slot += 1;
    }

    (children, child_range)
}

fn detect_cycles<Id: Copy + std::fmt::Debug>(
    ids: &[Id],
    children: &[usize],
    child_range: &[Range<usize>],
) -> Result<(), LayoutError<Id>> {
    let mut state = vec![VisitState::Unvisited; ids.len()];
    let mut stack: Vec<CycleFrame> = Vec::with_capacity(ids.len());

    for start in 0..ids.len() {
        if state[start] != VisitState::Unvisited {
            continue;
        }

        state[start] = VisitState::Visiting;
        stack.push(CycleFrame {
            node_index: start,
            next_child_pos: child_range[start].start,
        });

        while let Some(frame) = stack.last_mut() {
            let node_index = frame.node_index;
            let child_end = child_range[node_index].end;
            if frame.next_child_pos >= child_end {
                state[node_index] = VisitState::Visited;
                stack.pop();
                continue;
            }

            let child_index = children[frame.next_child_pos];
            frame.next_child_pos += 1;
            match state[child_index] {
                VisitState::Unvisited => {
                    state[child_index] = VisitState::Visiting;
                    stack.push(CycleFrame {
                        node_index: child_index,
                        next_child_pos: child_range[child_index].start,
                    });
                }
                VisitState::Visiting => {
                    redesmyn_logging::tracing::debug!(
                        node_id = ?ids[child_index],
                        "cycle detected"
                    );
                    return Err(LayoutError::CycleDetected {
                        id: ids[child_index],
                    });
                }
                VisitState::Visited => {}
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Unvisited,
    Visiting,
    Visited,
}

#[derive(Clone, Copy)]
struct CycleFrame {
    node_index: usize,
    next_child_pos: usize,
}

fn build_postorder(
    roots: &[usize],
    children: &[usize],
    child_range: &[Range<usize>],
    node_count: usize,
) -> Vec<usize> {
    let mut postorder = Vec::with_capacity(node_count);
    let mut stack: Vec<PostorderFrame> = Vec::with_capacity(node_count);

    for &root_index in roots {
        stack.push(PostorderFrame {
            node_index: root_index,
            next_child_pos: child_range[root_index].start,
        });

        while let Some(frame) = stack.last_mut() {
            let node_index = frame.node_index;
            let child_end = child_range[node_index].end;
            if frame.next_child_pos >= child_end {
                postorder.push(node_index);
                stack.pop();
                continue;
            }

            let child_index = children[frame.next_child_pos];
            frame.next_child_pos += 1;
            stack.push(PostorderFrame {
                node_index: child_index,
                next_child_pos: child_range[child_index].start,
            });
        }
    }

    postorder
}

#[derive(Clone, Copy)]
struct PostorderFrame {
    node_index: usize,
    next_child_pos: usize,
}

fn compute_bounds(
    sizes: impl IntoIterator<Item = Size>,
    positions: impl IntoIterator<Item = Point>,
) -> Option<Rect> {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for (size, position) in sizes.into_iter().zip(positions) {
        min_x = min_x.min(position.x);
        min_y = min_y.min(position.y);
        max_x = max_x.max(position.x.saturating_add(size.width));
        max_y = max_y.max(position.y.saturating_add(size.height));
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

    #[cfg(test)]
    mod alloc_counter {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;
        use std::sync::atomic::{AtomicUsize, Ordering};

        pub struct CountingAllocator;

        static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
        thread_local! {
            static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
        }

        impl CountingAllocator {
            pub fn begin() {
                TRACK_ALLOCATIONS.with(|flag| flag.set(true));
                Self::reset();
            }

            pub fn end() {
                TRACK_ALLOCATIONS.with(|flag| flag.set(false));
            }

            pub fn reset() {
                ALLOCATION_CALLS.store(0, Ordering::Relaxed);
            }

            pub fn count() -> usize {
                ALLOCATION_CALLS.load(Ordering::Relaxed)
            }

            fn is_tracking() -> bool {
                TRACK_ALLOCATIONS.with(|flag| flag.get())
            }
        }

        unsafe impl GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                if CountingAllocator::is_tracking() {
                    ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                }
                unsafe { System.alloc(layout) }
            }

            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                if CountingAllocator::is_tracking() {
                    ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                }
                unsafe { System.alloc_zeroed(layout) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                if CountingAllocator::is_tracking() {
                    ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                }
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }
    }

    #[global_allocator]
    static GLOBAL_ALLOCATOR: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

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

    #[test]
    fn can_reject_unknown_parent_ids_when_configured() {
        let nodes = vec![LayoutNode {
            id: 1_u32,
            parent_id: Some(999),
            size: Size {
                width: 10,
                height: 10,
            },
        }];

        let err = layout_forest_with_options(
            nodes,
            LayoutConfig::default(),
            LayoutOptions {
                unknown_parent_policy: UnknownParentPolicy::Error,
            },
        )
        .expect_err("should reject unknown parent_id");

        assert_eq!(
            err,
            LayoutError::UnknownParentId {
                id: 1,
                parent_id: 999
            }
        );
    }

    #[test]
    fn layout_engine_does_not_allocate_on_layout_after_construction() {
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
                id: 4_u32,
                parent_id: Some(2),
                size: Size {
                    width: 70,
                    height: 30,
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

        let mut engine =
            ForestLayoutEngine::new(nodes, LayoutOptions::default()).expect("engine should build");

        alloc_counter::CountingAllocator::begin();
        engine.layout(config);
        let (idx_3, idx_4, idx_10, baseline_pos_3, baseline_pos_4, baseline_pos_10) = {
            let ids = engine.ids();
            let positions = engine.positions();
            assert_no_overlaps(ids, positions, engine.sizes());
            let idx_3 = index_of_id(ids, 3);
            let idx_4 = index_of_id(ids, 4);
            let idx_10 = index_of_id(ids, 10);
            (
                idx_3,
                idx_4,
                idx_10,
                positions[idx_3],
                positions[idx_4],
                positions[idx_10],
            )
        };

        let mut saw_sibling_push_away = false;
        let mut saw_child_push_away = false;
        let mut saw_root_stack_push_away = false;

        for size in [
            Size {
                width: 80,
                height: 40,
            },
            Size {
                width: 120,
                height: 40,
            },
            Size {
                width: 80,
                height: 110,
            },
            Size {
                width: 160,
                height: 70,
            },
        ] {
            engine.set_size(2, size).expect("set_size should succeed");
            engine.layout(config);
            {
                let positions = engine.positions();
                assert_no_overlaps(engine.ids(), positions, engine.sizes());
                let pos_3 = positions[idx_3];
                let pos_4 = positions[idx_4];
                let pos_10 = positions[idx_10];
                saw_sibling_push_away |= pos_3.y != baseline_pos_3.y;
                saw_child_push_away |= pos_4.x != baseline_pos_4.x;
                saw_root_stack_push_away |= pos_10.y != baseline_pos_10.y;
            }
        }

        assert!(
            saw_sibling_push_away && saw_child_push_away && saw_root_stack_push_away,
            "expected some neighbor nodes to move when node size changes"
        );

        alloc_counter::CountingAllocator::end();
        assert_eq!(alloc_counter::CountingAllocator::count(), 0);
    }

    fn index_of_id(ids: &[u32], id: u32) -> usize {
        ids.iter()
            .position(|candidate| *candidate == id)
            .expect("id should exist")
    }

    fn assert_no_overlaps(ids: &[u32], positions: &[Point], sizes: &[Size]) {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a_id = ids[i];
                let b_id = ids[j];
                let a_pos = positions[i];
                let b_pos = positions[j];
                let a_size = sizes[i];
                let b_size = sizes[j];
                assert!(
                    !rect_intersects(a_pos, a_size, b_pos, b_size),
                    "nodes {a_id} and {b_id} overlap: {a_pos:?} {a_size:?} vs {b_pos:?} {b_size:?}",
                );
            }
        }
    }
}
