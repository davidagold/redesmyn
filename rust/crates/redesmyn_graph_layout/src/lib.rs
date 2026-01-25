//! Deterministic layout for a rooted forest of variable-size nodes.
//!
//! This crate focuses on the primary structure we have today: a forest defined
//! by a `parent_id` pointer (tree edges). The layout is pure and deterministic:
//! identical inputs produce identical outputs.
//!
//! For high-performance use (e.g. per-frame expand/collapse relayout), prefer
//! [`ForestLayoutEngine`]:
//! - build and validate topology once,
//! - update node sizes/visibility cheaply,
//! - recompute layout without heap allocations.

#![cfg_attr(not(test), forbid(unsafe_code))]

use std::collections::BTreeMap;
use std::ops::Range;

/// A 2D point in logical pixels.
///
/// In layout outputs, points represent the top-left origin of a node's rectangle.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutEdge<Id> {
    pub from: Id,
    pub to: Id,
    /// Number of hidden intermediate nodes skipped between `from` and `to`.
    pub elided_hops: u32,
}

/// A stable, allocation-free view into a computed layout.
///
/// Prefer the pattern `layout_in_place(...); output_view()` to avoid holding a
/// mutable borrow of the engine while you iterate the results.
///
/// The id at index `i` corresponds to `positions[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutOutputView<'a, Id> {
    /// Stable node order (sorted by id) of *visible* nodes.
    ///
    /// The position for `ids[i]` is `positions[i]`.
    pub ids: &'a [Id],
    pub positions: &'a [Point],
    /// Derived parent→child edges between visible nodes.
    ///
    /// If intermediate ancestors are hidden, edges connect to the nearest
    /// visible ancestor and expose `elided_hops`.
    pub edges: &'a [LayoutEdge<Id>],
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
/// [`Self::layout_in_place`] reuse internal buffers and perform no heap
/// allocations.
///
/// For ergonomics (and to avoid borrow pitfalls), prefer:
/// - `engine.layout_in_place(config);`
/// - `let view = engine.output_view();`
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

    sizes: Vec<Size>,

    /// Whether a node is present in the derived view forest.
    visible: Vec<bool>,
    /// Whether a *visible* node prunes its descendants in the view forest.
    collapsed: Vec<bool>,

    /// Per-node derived parent after applying visibility/collapse.
    view_parent_original: Vec<Option<usize>>,
    /// Per-node hop count between derived parent and node (only meaningful when visible).
    view_elided_hops_original: Vec<u32>,
    /// Per-node flag indicating inclusion in the view (visible and not pruned by a collapsed
    /// visible ancestor).
    view_active_original: Vec<bool>,

    /// Visible nodes (original indices) in stable id order.
    view_nodes: Vec<usize>,
    /// Mapping original index → view index (or -1 if not visible).
    view_index_by_original: Vec<i32>,

    /// View parent index (view indices).
    view_parent: Vec<Option<usize>>,
    /// Hop count for each view node to its derived parent (roots have 0).
    view_elided_hops: Vec<u32>,
    view_roots: Vec<usize>,

    view_children: Vec<usize>,
    view_child_range: Vec<Range<usize>>,
    view_child_counts: Vec<usize>,
    view_next_child_slot: Vec<usize>,

    view_left_sibling: Vec<Option<usize>>,
    view_number: Vec<usize>,

    view_depth: Vec<usize>,
    depth_max_width: Vec<i32>,
    depth_x: Vec<i32>,

    // Tidy layout (Buchheim) working state.
    prelim: Vec<i32>,
    modifier: Vec<i32>,
    change: Vec<i32>,
    shift: Vec<i32>,
    ancestor: Vec<usize>,
    thread: Vec<Option<usize>>,
    y_center: Vec<i32>,

    // Output buffers (visible nodes only).
    view_ids: Vec<Id>,
    view_positions: Vec<Point>,
    view_edges: Vec<LayoutEdge<Id>>,
    bounds: Rect,

    // Scratch stacks for derived-view building and subtree traversals.
    walk_stack: Vec<ViewWalkItem>,
    subtree_nodes: Vec<usize>,
    depth_stack: Vec<DepthItem>,
}

#[derive(Debug, Clone, Copy)]
struct ViewWalkItem {
    node_index: usize,
    current_visible_ancestor: Option<usize>,
    hidden_hops: u32,
}

#[derive(Debug, Clone, Copy)]
struct DepthItem {
    node_index: usize,
    depth: usize,
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

        Ok(Self {
            ids,
            index_by_id,
            parent_index,
            roots,
            children,
            child_range,
            sizes,
            visible: vec![true; node_count],
            collapsed: vec![false; node_count],
            view_parent_original: vec![None; node_count],
            view_elided_hops_original: vec![0; node_count],
            view_active_original: vec![false; node_count],
            view_nodes: Vec::with_capacity(node_count),
            view_index_by_original: vec![-1; node_count],
            view_parent: Vec::with_capacity(node_count),
            view_elided_hops: Vec::with_capacity(node_count),
            view_roots: Vec::with_capacity(node_count),
            view_children: Vec::with_capacity(node_count),
            view_child_range: Vec::with_capacity(node_count),
            view_child_counts: vec![0; node_count],
            view_next_child_slot: vec![0; node_count],
            view_left_sibling: Vec::with_capacity(node_count),
            view_number: Vec::with_capacity(node_count),
            view_depth: Vec::with_capacity(node_count),
            depth_max_width: Vec::with_capacity(node_count),
            depth_x: Vec::with_capacity(node_count),
            prelim: Vec::with_capacity(node_count),
            modifier: Vec::with_capacity(node_count),
            change: Vec::with_capacity(node_count),
            shift: Vec::with_capacity(node_count),
            ancestor: Vec::with_capacity(node_count),
            thread: Vec::with_capacity(node_count),
            y_center: Vec::with_capacity(node_count),
            view_ids: Vec::with_capacity(node_count),
            view_positions: Vec::with_capacity(node_count),
            view_edges: Vec::with_capacity(node_count),
            bounds: Rect {
                origin: Point { x: 0, y: 0 },
                size: Size {
                    width: 0,
                    height: 0,
                },
            },
            walk_stack: Vec::with_capacity(node_count),
            subtree_nodes: Vec::with_capacity(node_count),
            depth_stack: Vec::with_capacity(node_count),
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
    pub fn visible(&self) -> &[bool] {
        &self.visible
    }

    #[must_use]
    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    #[must_use]
    pub fn parent_index(&self) -> &[Option<usize>] {
        &self.parent_index
    }

    /// Returns a lightweight view into the most recently computed layout.
    ///
    /// Prefer `layout_in_place(...); output_view()` if you want to compute layout
    /// and then freely call other `&self` getters while iterating the view.
    #[must_use]
    pub fn output_view(&self) -> LayoutOutputView<'_, Id> {
        LayoutOutputView {
            ids: &self.view_ids,
            positions: &self.view_positions,
            edges: &self.view_edges,
            bounds: self.bounds,
        }
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

    pub fn set_visible(&mut self, id: Id, visible: bool) -> Result<(), LayoutError<Id>> {
        let Some(&index) = self.index_by_id.get(&id) else {
            return Err(LayoutError::UnknownNodeId { id });
        };
        self.visible[index] = visible;
        Ok(())
    }

    pub fn set_collapsed(&mut self, id: Id, collapsed: bool) -> Result<(), LayoutError<Id>> {
        let Some(&index) = self.index_by_id.get(&id) else {
            return Err(LayoutError::UnknownNodeId { id });
        };
        self.collapsed[index] = collapsed;
        Ok(())
    }

    /// Recomputes positions in-place and returns the derived-view bounds.
    ///
    /// This is allocation-free after engine construction.
    pub fn layout_in_place(&mut self, config: LayoutConfig) -> Rect {
        self.rebuild_view();
        self.recompute_depths_and_x(config);
        self.recompute_y_positions(config);
        self.bounds = compute_bounds(
            self.view_nodes
                .iter()
                .copied()
                .map(|original| self.sizes[original]),
            self.view_positions.iter().copied(),
        )
        .unwrap_or(Rect {
            origin: config.origin,
            size: Size {
                width: 0,
                height: 0,
            },
        });
        self.bounds
    }

    /// Convenience API that recomputes layout and returns a view.
    ///
    /// The returned view is tied to the mutable borrow created by this call. If
    /// you want to compute layout and then freely call other getters while you
    /// iterate, prefer `layout_in_place(config); engine.output_view()`.
    pub fn layout(&mut self, config: LayoutConfig) -> LayoutOutputView<'_, Id> {
        self.layout_in_place(config);
        self.output_view()
    }
    fn rebuild_view(&mut self) {
        let node_count = self.ids.len();

        self.view_active_original.fill(false);
        self.view_parent_original.fill(None);
        self.view_elided_hops_original.fill(0);

        self.walk_stack.clear();
        for &root in &self.roots {
            self.walk_stack.push(ViewWalkItem {
                node_index: root,
                current_visible_ancestor: None,
                hidden_hops: 0,
            });
        }

        while let Some(item) = self.walk_stack.pop() {
            let node_index = item.node_index;
            let is_visible = self.visible[node_index];

            if is_visible {
                self.view_active_original[node_index] = true;
                self.view_parent_original[node_index] = item.current_visible_ancestor;
                self.view_elided_hops_original[node_index] =
                    if item.current_visible_ancestor.is_some() {
                        item.hidden_hops
                    } else {
                        0
                    };
            }

            let next_visible_ancestor = if is_visible {
                Some(node_index)
            } else {
                item.current_visible_ancestor
            };

            let next_hidden_hops = if is_visible {
                0
            } else if item.current_visible_ancestor.is_some() {
                item.hidden_hops.saturating_add(1)
            } else {
                0
            };

            let prune_descendants = is_visible && self.collapsed[node_index];
            if prune_descendants {
                continue;
            }

            let range = self.child_range[node_index].clone();
            for &child in self.children[range].iter().rev() {
                self.walk_stack.push(ViewWalkItem {
                    node_index: child,
                    current_visible_ancestor: next_visible_ancestor,
                    hidden_hops: next_hidden_hops,
                });
            }
        }

        self.view_nodes.clear();
        self.view_ids.clear();
        self.view_index_by_original.fill(-1);

        for original_index in 0..node_count {
            if !self.view_active_original[original_index] {
                continue;
            }
            let view_index = self.view_nodes.len();
            self.view_nodes.push(original_index);
            self.view_ids.push(self.ids[original_index]);
            self.view_index_by_original[original_index] = view_index as i32;
        }

        let view_count = self.view_nodes.len();
        self.view_parent.clear();
        self.view_parent.resize(view_count, None);
        self.view_elided_hops.clear();
        self.view_elided_hops.resize(view_count, 0);
        self.view_edges.clear();

        for view_index in 0..view_count {
            let original_index = self.view_nodes[view_index];
            let parent_original = self.view_parent_original[original_index];
            let parent_view = parent_original.and_then(|original_parent| {
                let idx = self.view_index_by_original[original_parent];
                if idx >= 0 { Some(idx as usize) } else { None }
            });

            self.view_parent[view_index] = parent_view;
            let elided_hops = self.view_elided_hops_original[original_index];
            self.view_elided_hops[view_index] = elided_hops;

            if let Some(parent_view) = parent_view {
                self.view_edges.push(LayoutEdge {
                    from: self.view_ids[parent_view],
                    to: self.view_ids[view_index],
                    elided_hops,
                });
            }
        }

        self.view_roots.clear();
        self.view_child_counts[..view_count].fill(0);

        for (child, parent) in self.view_parent.iter().copied().enumerate() {
            match parent {
                Some(parent) => self.view_child_counts[parent] += 1,
                None => self.view_roots.push(child),
            }
        }

        self.view_child_range.clear();
        self.view_child_range.resize(view_count, 0..0);
        self.view_next_child_slot[..view_count].fill(0);

        let mut total_children = 0_usize;
        for parent in 0..view_count {
            let count = self.view_child_counts[parent];
            let start = total_children;
            total_children += count;
            self.view_child_range[parent] = start..total_children;
            self.view_next_child_slot[parent] = start;
        }

        self.view_children.clear();
        self.view_children.resize(total_children, 0);
        for child in 0..view_count {
            let Some(parent) = self.view_parent[child] else {
                continue;
            };
            let slot = &mut self.view_next_child_slot[parent];
            self.view_children[*slot] = child;
            *slot += 1;
        }

        self.view_left_sibling.clear();
        self.view_left_sibling.resize(view_count, None);
        self.view_number.clear();
        self.view_number.resize(view_count, 1);

        for parent in 0..view_count {
            let range = self.view_child_range[parent].clone();
            let mut prev = None;
            let mut number = 1_usize;
            for slot in range {
                let child = self.view_children[slot];
                self.view_left_sibling[child] = prev;
                self.view_number[child] = number;
                prev = Some(child);
                number += 1;
            }
        }
    }

    fn recompute_depths_and_x(&mut self, config: LayoutConfig) {
        let view_count = self.view_nodes.len();
        self.view_depth.clear();
        self.view_depth.resize(view_count, 0);

        self.depth_stack.clear();
        for &root in &self.view_roots {
            self.depth_stack.push(DepthItem {
                node_index: root,
                depth: 0,
            });
        }

        let mut max_depth = 0_usize;
        while let Some(item) = self.depth_stack.pop() {
            self.view_depth[item.node_index] = item.depth;
            max_depth = max_depth.max(item.depth);

            let range = self.view_child_range[item.node_index].clone();
            for slot in range.rev() {
                let child = self.view_children[slot];
                self.depth_stack.push(DepthItem {
                    node_index: child,
                    depth: item.depth + 1,
                });
            }
        }

        self.depth_max_width.clear();
        self.depth_max_width.resize(max_depth.saturating_add(1), 0);
        for view_index in 0..view_count {
            let depth = self.view_depth[view_index];
            let width = self.sizes[self.view_nodes[view_index]].width;
            if width > self.depth_max_width[depth] {
                self.depth_max_width[depth] = width;
            }
        }

        self.depth_x.clear();
        self.depth_x.resize(max_depth.saturating_add(1), 0);
        let mut x = config.origin.x;
        for depth in 0..=max_depth {
            self.depth_x[depth] = x;
            let step = self
                .depth_max_width
                .get(depth)
                .copied()
                .unwrap_or(0)
                .saturating_add(config.layer_spacing);
            x = x.saturating_add(step);
        }

        self.view_positions.clear();
        self.view_positions.resize(
            view_count,
            Point {
                x: config.origin.x,
                y: config.origin.y,
            },
        );
        for view_index in 0..view_count {
            let depth = self.view_depth[view_index];
            self.view_positions[view_index].x = self.depth_x[depth];
        }
    }

    fn recompute_y_positions(&mut self, config: LayoutConfig) {
        let view_count = self.view_nodes.len();
        if view_count == 0 {
            return;
        }

        self.prelim.clear();
        self.prelim.resize(view_count, 0);
        self.modifier.clear();
        self.modifier.resize(view_count, 0);
        self.change.clear();
        self.change.resize(view_count, 0);
        self.shift.clear();
        self.shift.resize(view_count, 0);
        self.ancestor.clear();
        self.ancestor.resize_with(view_count, || 0);
        for i in 0..view_count {
            self.ancestor[i] = i;
        }
        self.thread.clear();
        self.thread.resize(view_count, None);
        self.y_center.clear();
        self.y_center.resize(view_count, 0);

        let roots_len = self.view_roots.len();
        let mut next_root_top = config.origin.y;
        for root_slot in 0..roots_len {
            let root = self.view_roots[root_slot];
            self.first_walk(root, config);
            self.second_walk(root, 0);

            self.subtree_nodes.clear();
            self.collect_subtree_nodes(root);

            let mut min_top = i32::MAX;
            let mut max_bottom = i32::MIN;
            for &node in &self.subtree_nodes {
                let (top, bottom) = self.node_vertical_bounds(node, self.y_center[node]);
                min_top = min_top.min(top);
                max_bottom = max_bottom.max(bottom);
            }

            let offset = next_root_top.saturating_sub(min_top);
            for &node in &self.subtree_nodes {
                let shifted_center = self.y_center[node].saturating_add(offset);
                let size = self.node_size(node);
                self.view_positions[node].y = shifted_center.saturating_sub(size.height / 2);
            }

            next_root_top = max_bottom
                .saturating_add(offset)
                .saturating_add(config.root_spacing);
        }
    }

    fn node_size(&self, view_index: usize) -> Size {
        self.sizes[self.view_nodes[view_index]]
    }

    fn node_vertical_bounds(&self, view_index: usize, y_center: i32) -> (i32, i32) {
        let size = self.node_size(view_index);
        let top = y_center.saturating_sub(size.height / 2);
        let bottom = top.saturating_add(size.height);
        (top, bottom)
    }

    fn sibling_separation(&self, a: usize, b: usize, config: LayoutConfig) -> i32 {
        let a_height = self.node_size(a).height;
        let b_height = self.node_size(b).height;
        (a_height / 2)
            .saturating_add(b_height / 2)
            .saturating_add(config.sibling_spacing)
    }

    fn leftmost_child(&self, v: usize) -> Option<usize> {
        let range = self.view_child_range[v].clone();
        (range.start < range.end).then(|| self.view_children[range.start])
    }

    fn rightmost_child(&self, v: usize) -> Option<usize> {
        let range = self.view_child_range[v].clone();
        (range.start < range.end).then(|| self.view_children[range.end - 1])
    }

    fn next_left(&self, v: usize) -> Option<usize> {
        self.leftmost_child(v).or(self.thread[v])
    }

    fn next_right(&self, v: usize) -> Option<usize> {
        self.rightmost_child(v).or(self.thread[v])
    }

    fn leftmost_sibling(&self, v: usize) -> Option<usize> {
        let parent = self.view_parent[v]?;
        self.leftmost_child(parent)
    }

    fn first_walk(&mut self, v: usize, config: LayoutConfig) {
        let range = self.view_child_range[v].clone();
        if range.start == range.end {
            if let Some(left) = self.view_left_sibling[v] {
                self.prelim[v] =
                    self.prelim[left].saturating_add(self.sibling_separation(left, v, config));
            } else {
                self.prelim[v] = 0;
            }
            return;
        }

        let mut default_ancestor = self.view_children[range.start];
        for slot in range.clone() {
            let w = self.view_children[slot];
            self.first_walk(w, config);
            default_ancestor = self.apportion(w, default_ancestor, config);
        }
        self.execute_shifts(v);

        let first = self.view_children[range.start];
        let last = self.view_children[range.end - 1];
        let midpoint = (self.prelim[first] + self.prelim[last]) / 2;

        if let Some(left) = self.view_left_sibling[v] {
            let prelim = self.prelim[left].saturating_add(self.sibling_separation(left, v, config));
            self.prelim[v] = prelim;
            self.modifier[v] = prelim.saturating_sub(midpoint);
        } else {
            self.prelim[v] = midpoint;
        }
    }

    fn apportion(&mut self, v: usize, default_ancestor: usize, config: LayoutConfig) -> usize {
        let Some(w) = self.view_left_sibling[v] else {
            return default_ancestor;
        };

        let mut vir = v;
        let mut vor = v;
        let mut vil = w;
        let mut vol = self.leftmost_sibling(v).unwrap_or(w);

        let mut sir = self.modifier[vir];
        let mut sor = self.modifier[vor];
        let mut sil = self.modifier[vil];
        let mut sol = self.modifier[vol];

        let mut default_ancestor = default_ancestor;

        while self.next_right(vil).is_some() && self.next_left(vir).is_some() {
            vil = self.next_right(vil).expect("checked");
            vir = self.next_left(vir).expect("checked");
            vol = self.next_left(vol).unwrap_or(vol);
            vor = self.next_right(vor).unwrap_or(vor);

            self.ancestor[vor] = v;

            let required = self.sibling_separation(vil, vir, config);
            let shift = (self.prelim[vil] as i64 + sil as i64 + required as i64)
                - (self.prelim[vir] as i64 + sir as i64);

            if shift > 0 {
                let a = self.ancestor_for(vil, v, default_ancestor);
                self.move_subtree(a, v, shift as i32);
                sir = sir.saturating_add(shift as i32);
                sor = sor.saturating_add(shift as i32);
            }

            sil = sil.saturating_add(self.modifier[vil]);
            sir = sir.saturating_add(self.modifier[vir]);
            sol = sol.saturating_add(self.modifier[vol]);
            sor = sor.saturating_add(self.modifier[vor]);
        }

        if self.next_right(vil).is_some() && self.next_right(vor).is_none() {
            self.thread[vor] = self.next_right(vil);
            self.modifier[vor] = self.modifier[vor].saturating_add(sil.saturating_sub(sor));
        }

        if self.next_left(vir).is_some() && self.next_left(vol).is_none() {
            self.thread[vol] = self.next_left(vir);
            self.modifier[vol] = self.modifier[vol].saturating_add(sir.saturating_sub(sol));
            default_ancestor = v;
        }

        default_ancestor
    }

    fn ancestor_for(&self, vil: usize, v: usize, default_ancestor: usize) -> usize {
        let candidate = self.ancestor[vil];
        if self.view_parent.get(candidate).copied().flatten() == self.view_parent[v] {
            candidate
        } else {
            default_ancestor
        }
    }

    fn move_subtree(&mut self, wl: usize, wr: usize, shift: i32) {
        let subtrees = self.view_number[wr].saturating_sub(self.view_number[wl]);
        if subtrees == 0 {
            return;
        }

        let shift_per_subtree = shift / subtrees as i32;
        self.change[wr] = self.change[wr].saturating_sub(shift_per_subtree);
        self.shift[wr] = self.shift[wr].saturating_add(shift);
        self.change[wl] = self.change[wl].saturating_add(shift_per_subtree);
        self.prelim[wr] = self.prelim[wr].saturating_add(shift);
        self.modifier[wr] = self.modifier[wr].saturating_add(shift);
    }

    fn execute_shifts(&mut self, v: usize) {
        let range = self.view_child_range[v].clone();
        let mut shift = 0_i32;
        let mut change = 0_i32;
        for slot in range.rev() {
            let w = self.view_children[slot];
            self.prelim[w] = self.prelim[w].saturating_add(shift);
            self.modifier[w] = self.modifier[w].saturating_add(shift);
            change = change.saturating_add(self.change[w]);
            shift = shift.saturating_add(self.shift[w]).saturating_add(change);
        }
    }

    fn second_walk(&mut self, v: usize, m: i32) {
        self.y_center[v] = self.prelim[v].saturating_add(m);
        let range = self.view_child_range[v].clone();
        let next_m = m.saturating_add(self.modifier[v]);
        for slot in range {
            let w = self.view_children[slot];
            self.second_walk(w, next_m);
        }
    }

    fn collect_subtree_nodes(&mut self, root: usize) {
        self.subtree_nodes.clear();
        self.depth_stack.clear();
        self.depth_stack.push(DepthItem {
            node_index: root,
            depth: 0,
        });
        while let Some(item) = self.depth_stack.pop() {
            self.subtree_nodes.push(item.node_index);
            let range = self.view_child_range[item.node_index].clone();
            for slot in range.rev() {
                let child = self.view_children[slot];
                self.depth_stack.push(DepthItem {
                    node_index: child,
                    depth: 0,
                });
            }
        }
    }
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

        let snapshot_a = layout_snapshot(nodes_a.clone(), config);
        let snapshot_a_second = layout_snapshot(nodes_a, config);
        let snapshot_b = layout_snapshot(nodes_b, config);
        assert_eq!(snapshot_a, snapshot_a_second);
        assert_eq!(snapshot_a, snapshot_b);
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
        let snapshot = layout_snapshot(nodes, config);
        assert_no_overlaps_view(&snapshot.ids, &snapshot.positions, &sizes_by_id);
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

        let err = ForestLayoutEngine::new(nodes, LayoutOptions::default())
            .expect_err("should detect cycle");
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

        let err = ForestLayoutEngine::new(
            nodes,
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
        engine.layout_in_place(config);
        let view = engine.output_view();
        assert_eq!(view.ids, engine.ids());
        assert_no_overlaps_aligned(view.ids, view.positions, engine.sizes());
        let idx_3 = index_of_id(view.ids, 3);
        let idx_4 = index_of_id(view.ids, 4);
        let idx_10 = index_of_id(view.ids, 10);
        let baseline_pos_3 = view.positions[idx_3];
        let baseline_pos_4 = view.positions[idx_4];
        let baseline_pos_10 = view.positions[idx_10];

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
            engine.layout_in_place(config);
            {
                let view = engine.output_view();
                assert_eq!(view.ids, engine.ids());
                assert_no_overlaps_aligned(view.ids, view.positions, engine.sizes());
                let pos_3 = view.positions[idx_3];
                let pos_4 = view.positions[idx_4];
                let pos_10 = view.positions[idx_10];
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

    #[test]
    fn hiding_intermediate_ancestors_elides_hops() {
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
            LayoutNode {
                id: 3_u32,
                parent_id: Some(2),
                size: Size {
                    width: 10,
                    height: 10,
                },
            },
            LayoutNode {
                id: 4_u32,
                parent_id: Some(3),
                size: Size {
                    width: 10,
                    height: 10,
                },
            },
        ];

        let mut engine =
            ForestLayoutEngine::new(nodes, LayoutOptions::default()).expect("engine should build");
        engine
            .set_visible(2, false)
            .expect("set_visible should succeed");
        engine
            .set_visible(3, false)
            .expect("set_visible should succeed");
        engine.layout_in_place(config);
        let view = engine.output_view();

        assert_eq!(view.ids, &[1, 4]);
        assert_eq!(
            view.edges,
            &[LayoutEdge {
                from: 1,
                to: 4,
                elided_hops: 2,
            }]
        );
    }

    #[test]
    fn collapsing_a_visible_node_prunes_descendants() {
        let config = LayoutConfig::default();
        let nodes = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: None,
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
            LayoutNode {
                id: 3_u32,
                parent_id: Some(2),
                size: Size {
                    width: 10,
                    height: 10,
                },
            },
        ];

        let mut engine =
            ForestLayoutEngine::new(nodes, LayoutOptions::default()).expect("engine should build");
        engine
            .set_collapsed(2, true)
            .expect("set_collapsed should succeed");
        engine.layout_in_place(config);
        let view = engine.output_view();

        assert_eq!(view.ids, &[1, 2]);
        assert_eq!(
            view.edges,
            &[LayoutEdge {
                from: 1,
                to: 2,
                elided_hops: 0,
            }]
        );
    }

    #[test]
    fn collapsing_a_hidden_node_does_not_prune_descendants() {
        let config = LayoutConfig::default();
        let nodes = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: None,
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
            LayoutNode {
                id: 3_u32,
                parent_id: Some(2),
                size: Size {
                    width: 10,
                    height: 10,
                },
            },
        ];

        let mut engine =
            ForestLayoutEngine::new(nodes, LayoutOptions::default()).expect("engine should build");
        engine
            .set_visible(2, false)
            .expect("set_visible should succeed");
        engine
            .set_collapsed(2, true)
            .expect("set_collapsed should succeed");
        engine.layout_in_place(config);
        let view = engine.output_view();

        assert_eq!(view.ids, &[1, 3]);
        assert_eq!(
            view.edges,
            &[LayoutEdge {
                from: 1,
                to: 3,
                elided_hops: 1,
            }]
        );
    }

    #[test]
    fn can_overlap_vertically_across_depths_when_x_is_disjoint() {
        let config = LayoutConfig {
            layer_spacing: 20,
            sibling_spacing: 10,
            root_spacing: 15,
            origin: Point { x: 0, y: 0 },
        };

        // 1
        // ├── 2
        // │   ├── 4
        // │   └── 5
        // └── 3
        let nodes = vec![
            LayoutNode {
                id: 1_u32,
                parent_id: None,
                size: Size {
                    width: 100,
                    height: 30,
                },
            },
            LayoutNode {
                id: 2_u32,
                parent_id: Some(1),
                size: Size {
                    width: 100,
                    height: 100,
                },
            },
            LayoutNode {
                id: 3_u32,
                parent_id: Some(1),
                size: Size {
                    width: 100,
                    height: 100,
                },
            },
            LayoutNode {
                id: 4_u32,
                parent_id: Some(2),
                size: Size {
                    width: 100,
                    height: 100,
                },
            },
            LayoutNode {
                id: 5_u32,
                parent_id: Some(2),
                size: Size {
                    width: 100,
                    height: 100,
                },
            },
        ];

        let sizes_by_id: BTreeMap<u32, Size> =
            nodes.iter().map(|node| (node.id, node.size)).collect();
        let snapshot = layout_snapshot(nodes, config);

        let idx_3 = index_of_id(&snapshot.ids, 3);
        let idx_4 = index_of_id(&snapshot.ids, 4);
        let idx_5 = index_of_id(&snapshot.ids, 5);

        let rect_3 = rect_for(&snapshot.ids, &snapshot.positions, &sizes_by_id, idx_3);
        let rect_4 = rect_for(&snapshot.ids, &snapshot.positions, &sizes_by_id, idx_4);
        let rect_5 = rect_for(&snapshot.ids, &snapshot.positions, &sizes_by_id, idx_5);

        let overlaps_y = intervals_overlap(
            (rect_3.origin.y, rect_3.bottom()),
            (rect_4.origin.y, rect_4.bottom()),
        ) || intervals_overlap(
            (rect_3.origin.y, rect_3.bottom()),
            (rect_5.origin.y, rect_5.bottom()),
        );
        assert!(
            overlaps_y,
            "expected node 3 (depth 1) to overlap vertically with at least one grandchild (depth 2)"
        );

        // Even if Y overlaps, the rectangles should not intersect because their depth columns
        // are disjoint in X.
        assert!(
            !rect_intersects(rect_3.origin, rect_3.size, rect_4.origin, rect_4.size),
            "expected node 3 and node 4 to be disjoint in 2D"
        );
        assert!(
            !rect_intersects(rect_3.origin, rect_3.size, rect_5.origin, rect_5.size),
            "expected node 3 and node 5 to be disjoint in 2D"
        );
    }

    fn layout_snapshot(nodes: Vec<LayoutNode<u32>>, config: LayoutConfig) -> LayoutSnapshot<u32> {
        let mut engine =
            ForestLayoutEngine::new(nodes, LayoutOptions::default()).expect("engine should build");
        engine.layout_in_place(config);
        let view = engine.output_view();
        LayoutSnapshot {
            ids: view.ids.to_vec(),
            positions: view.positions.to_vec(),
            edges: view.edges.to_vec(),
            bounds: view.bounds,
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct LayoutSnapshot<Id> {
        ids: Vec<Id>,
        positions: Vec<Point>,
        edges: Vec<LayoutEdge<Id>>,
        bounds: Rect,
    }

    fn index_of_id(ids: &[u32], id: u32) -> usize {
        ids.iter()
            .position(|candidate| *candidate == id)
            .expect("id should exist")
    }

    fn assert_no_overlaps_aligned(ids: &[u32], positions: &[Point], sizes: &[Size]) {
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

    fn assert_no_overlaps_view(
        ids: &[u32],
        positions: &[Point],
        sizes_by_id: &BTreeMap<u32, Size>,
    ) {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a_id = ids[i];
                let b_id = ids[j];
                let a_pos = positions[i];
                let b_pos = positions[j];
                let a_size = sizes_by_id[&a_id];
                let b_size = sizes_by_id[&b_id];
                assert!(
                    !rect_intersects(a_pos, a_size, b_pos, b_size),
                    "nodes {a_id} and {b_id} overlap: {a_pos:?} {a_size:?} vs {b_pos:?} {b_size:?}",
                );
            }
        }
    }

    fn rect_for(
        ids: &[u32],
        positions: &[Point],
        sizes_by_id: &BTreeMap<u32, Size>,
        index: usize,
    ) -> Rect {
        let id = ids[index];
        Rect {
            origin: positions[index],
            size: sizes_by_id[&id],
        }
    }

    fn intervals_overlap(a: (i32, i32), b: (i32, i32)) -> bool {
        a.0 < b.1 && a.1 > b.0
    }
}
