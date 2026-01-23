use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

use gpui::{
    App, AsyncApp, ClickEvent, Context, CursorStyle, FocusHandle, Focusable, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, ScrollHandle, ScrollWheelEvent, Task,
    TextRun, Window, canvas, div, fill, prelude::*, px, quad, rems,
};

use redesmyn_ids::CommandId;

use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ProgressPillKind, ScrollArea,
    TextButton,
};
use redesmyn_ui::utils::{UserActionState, theme_for_window};

use crate::camera::{GraphCamera, GraphCameraLimits};
use crate::geometry::{
    DEFAULT_EDGE_STROKE_PX, EdgeLodBand, EdgeRoute, edge_lod_band, edge_route_in_window,
};
use crate::hit_test::{GraphHit, hit_test};
use crate::scene::{AgentStatus, GraphEdgeId, GraphNodeId, GraphScene};

use redesmyn_protocol::client::{MergeReadiness, TaskState};

#[derive(Debug, Clone, Copy)]
struct PanDrag {
    start_mouse: gpui::Point<gpui::Pixels>,
    start_origin_world: gpui::Point<f32>,
    pending_clear_selection: bool,
    did_pan: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct NodeWorldRect {
    origin: gpui::Point<f32>,
    width: f32,
    height: f32,
}

impl NodeWorldRect {
    fn from_scene(scene: &GraphScene, node_id: GraphNodeId) -> Self {
        let origin = scene.node_world_origin(node_id);
        let size = scene.node_world_size(node_id);
        Self {
            origin,
            width: size.width as f32,
            height: size.height as f32,
        }
    }
}

#[derive(Debug, Clone)]
struct LayoutAnimation {
    start: Instant,
    duration: Duration,
    from_layout: BTreeMap<GraphNodeId, NodeWorldRect>,
    to_layout: BTreeMap<GraphNodeId, NodeWorldRect>,
    from_selected_node: Option<GraphNodeId>,
    to_selected_node: Option<GraphNodeId>,
}

impl LayoutAnimation {
    fn progress(&self) -> f32 {
        if self.duration == Duration::from_millis(0) {
            return 1.0;
        }

        let elapsed = self.start.elapsed().as_secs_f32();
        let total = self.duration.as_secs_f32();
        ease_out_cubic((elapsed / total).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone)]
struct CameraAnimation {
    start: Instant,
    duration: Duration,
    from_origin_world: gpui::Point<f32>,
    to_origin_world: gpui::Point<f32>,
    from_zoom: f32,
    to_zoom: f32,
}

impl CameraAnimation {
    fn progress(&self) -> f32 {
        if self.duration == Duration::from_millis(0) {
            return 1.0;
        }

        let elapsed = self.start.elapsed().as_secs_f32();
        let total = self.duration.as_secs_f32();
        ease_out_cubic((elapsed / total).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Copy)]
struct SelectionBarTransition {
    started_at: Instant,
    from: f32,
    to: f32,
    duration: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
struct BulkProgress {
    completed: usize,
    total: usize,
}

#[derive(Debug, Default)]
struct BulkCommandState {
    action: UserActionState,
    progress: Option<BulkProgress>,
    command_id: Option<CommandId>,
    task: Option<Task<()>>,
}

pub struct GraphView {
    focus_handle: FocusHandle,
    scene: GraphScene,
    camera: GraphCamera,
    camera_animation: Option<CameraAnimation>,
    pan_drag: Option<PanDrag>,
    last_canvas_bounds: Option<gpui::Bounds<gpui::Pixels>>,
    expanded_session_scroll: ScrollHandle,
    expanded_details_scroll: ScrollHandle,
    demo_action: UserActionState,
    demo_action_task: Option<Task<()>>,
    layout_animation: Option<LayoutAnimation>,
    pending_pan_to_selection: Option<GraphNodeId>,
    did_initial_fit: bool,
    fit_suppressed: bool,
    edge_label_cache: RefCell<HashMap<gpui::SharedString, gpui::ShapedLine>>,
    selection_bar_progress: f32,
    selection_bar_transition: Option<SelectionBarTransition>,
    bulk_start: BulkCommandState,
}

impl GraphView {
    pub fn new_demo(cx: &mut Context<Self>) -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_view.new_demo");
        let _guard = span.enter();

        Self {
            focus_handle: cx.focus_handle(),
            scene: GraphScene::demo(),
            camera: GraphCamera::new(GraphCameraLimits::default()),
            camera_animation: None,
            pan_drag: None,
            last_canvas_bounds: None,
            expanded_session_scroll: ScrollHandle::new(),
            expanded_details_scroll: ScrollHandle::new(),
            demo_action: UserActionState::default(),
            demo_action_task: None,
            layout_animation: None,
            pending_pan_to_selection: None,
            did_initial_fit: false,
            fit_suppressed: false,
            edge_label_cache: RefCell::new(HashMap::new()),
            selection_bar_progress: 0.0,
            selection_bar_transition: None,
            bulk_start: BulkCommandState::default(),
        }
    }

    fn visual_selected_node(&self) -> Option<GraphNodeId> {
        if let Some(animation) = self.layout_animation.as_ref()
            && animation.to_selected_node.is_none()
        {
            animation.from_selected_node
        } else {
            self.scene.selection().selected_node
        }
    }

    fn selection_label(&self) -> String {
        if let Some(edge) = self.scene.selection().selected_edge {
            return format!("Selected edge: {edge}");
        }
        let count = self.scene.selection().selected_nodes.len();
        if count > 1 {
            return format!("Selected nodes: {count}");
        }
        if let Some(node) = self
            .visual_selected_node()
            .or_else(|| self.scene.selection().selected_nodes.iter().next().copied())
        {
            return format!("Selected node: {node}");
        }
        "Selected: <none>".to_string()
    }

    fn update_selection(&mut self, mutate: impl FnOnce(&mut GraphScene), cx: &mut Context<Self>) {
        let previous_selection = self.scene.selection().clone();
        let from_layout = self.snapshot_displayed_layout();

        mutate(&mut self.scene);

        let next_selection = self.scene.selection().clone();
        if previous_selection.selected_node != next_selection.selected_node {
            self.reset_expanded_card_state();
            self.start_layout_animation(
                from_layout,
                self.snapshot_scene_layout(),
                previous_selection.selected_node,
                next_selection.selected_node,
            );

            if !self.did_initial_fit {
                self.fit_suppressed = true;
            }

            self.pending_pan_to_selection = next_selection.selected_node;
        }

        self.update_selection_bar_target();
        cx.notify();
    }

    fn select_node(&mut self, node_id: GraphNodeId, cx: &mut Context<Self>) {
        self.update_selection(|scene| scene.select_node(node_id), cx);
    }

    fn toggle_node(&mut self, node_id: GraphNodeId, cx: &mut Context<Self>) {
        self.update_selection(|scene| scene.toggle_node(node_id), cx);
    }

    fn select_edge(&mut self, edge_id: GraphEdgeId, cx: &mut Context<Self>) {
        self.update_selection(|scene| scene.select_edge(edge_id), cx);
    }

    fn clear_selection(&mut self, cx: &mut Context<Self>) {
        self.update_selection(|scene| scene.clear_selection(), cx);
    }

    fn reset_expanded_card_state(&mut self) {
        self.expanded_session_scroll = ScrollHandle::new();
        self.expanded_details_scroll = ScrollHandle::new();
        self.demo_action = UserActionState::default();
        self.demo_action_task = None;
    }

    fn snapshot_scene_layout(&self) -> BTreeMap<GraphNodeId, NodeWorldRect> {
        self.scene
            .nodes()
            .map(|node| (node.id, NodeWorldRect::from_scene(&self.scene, node.id)))
            .collect()
    }

    fn snapshot_displayed_layout(&self) -> BTreeMap<GraphNodeId, NodeWorldRect> {
        let t = self
            .layout_animation
            .as_ref()
            .map(LayoutAnimation::progress)
            .unwrap_or(1.0);

        self.scene
            .nodes()
            .map(|node| (node.id, self.node_world_rect_for_progress(node.id, t)))
            .collect()
    }

    fn start_layout_animation(
        &mut self,
        from_layout: BTreeMap<GraphNodeId, NodeWorldRect>,
        to_layout: BTreeMap<GraphNodeId, NodeWorldRect>,
        from_selected_node: Option<GraphNodeId>,
        to_selected_node: Option<GraphNodeId>,
    ) {
        self.layout_animation = Some(LayoutAnimation {
            start: Instant::now(),
            duration: Duration::from_millis(180),
            from_layout,
            to_layout,
            from_selected_node,
            to_selected_node,
        });
    }

    fn node_world_rect_for_progress(&self, node_id: GraphNodeId, t: f32) -> NodeWorldRect {
        let scene_rect = NodeWorldRect::from_scene(&self.scene, node_id);
        let Some(animation) = self.layout_animation.as_ref() else {
            return scene_rect;
        };
        if t >= 1.0 {
            return scene_rect;
        }

        let from = animation
            .from_layout
            .get(&node_id)
            .copied()
            .unwrap_or(scene_rect);
        let to = animation
            .to_layout
            .get(&node_id)
            .copied()
            .unwrap_or(scene_rect);
        lerp_world_rect(from, to, t)
    }

    fn node_bounds_in_window_for_progress(
        &self,
        node_id: GraphNodeId,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        t: f32,
    ) -> gpui::Bounds<gpui::Pixels> {
        let rect = self.node_world_rect_for_progress(node_id, t);
        let zoom = self.camera.zoom();

        let local_origin = self.camera.world_to_screen(rect.origin);
        let local_size = gpui::size(px(rect.width * zoom), px(rect.height * zoom));

        gpui::Bounds {
            origin: local_origin,
            size: local_size,
        } + canvas_bounds.origin
    }

    fn layout_animation_progress_for_render(&mut self, window: &Window) -> f32 {
        let Some(animation) = self.layout_animation.as_ref() else {
            return 1.0;
        };

        let t = animation.progress();
        if t >= 1.0 {
            self.layout_animation = None;
            1.0
        } else {
            window.request_animation_frame();
            t
        }
    }

    fn layout_animation_progress_for_paint(&self) -> f32 {
        self.layout_animation
            .as_ref()
            .map(LayoutAnimation::progress)
            .unwrap_or(1.0)
    }

    fn cancel_camera_animation(&mut self) -> bool {
        let Some(animation) = self.camera_animation.take() else {
            return false;
        };

        let t = animation.progress();
        if t <= 0.0 {
            return true;
        }

        self.camera.set_origin_world(lerp_point(
            animation.from_origin_world,
            animation.to_origin_world,
            t,
        ));
        self.camera
            .set_zoom(lerp_f32(animation.from_zoom, animation.to_zoom, t));
        true
    }

    fn step_camera_animation_for_render(&mut self, window: &Window) {
        let Some(animation) = self.camera_animation.as_ref() else {
            return;
        };

        let t = animation.progress();
        let origin = lerp_point(animation.from_origin_world, animation.to_origin_world, t);
        let zoom = lerp_f32(animation.from_zoom, animation.to_zoom, t);

        self.camera.set_origin_world(origin);
        self.camera.set_zoom(zoom);

        if t >= 1.0 {
            self.camera_animation = None;
        } else {
            window.request_animation_frame();
        }
    }

    fn try_start_camera_animation(
        &mut self,
        to_origin_world: gpui::Point<f32>,
        to_zoom: f32,
        duration: Duration,
    ) -> bool {
        self.cancel_camera_animation();

        let mut clamped_camera = self.camera;
        clamped_camera.set_zoom(to_zoom);
        let to_zoom = clamped_camera.zoom();

        if approx_eq_point(self.camera.origin_world(), to_origin_world)
            && (self.camera.zoom() - to_zoom).abs() < 1e-3
        {
            return false;
        }

        self.camera_animation = Some(CameraAnimation {
            start: Instant::now(),
            duration,
            from_origin_world: self.camera.origin_world(),
            to_origin_world,
            from_zoom: self.camera.zoom(),
            to_zoom,
        });
        true
    }

    fn viewport_margins_px(bounds: gpui::Bounds<gpui::Pixels>) -> (gpui::Pixels, gpui::Pixels) {
        const DESIRED_MARGIN_PX: f32 = 72.0;

        let width = f32::from(bounds.size.width);
        let height = f32::from(bounds.size.height);

        let margin_x = DESIRED_MARGIN_PX.min(width * 0.25);
        let margin_y = DESIRED_MARGIN_PX.min(height * 0.25);

        (px(margin_x), px(margin_y))
    }

    fn try_initial_fit_for_render(&mut self) -> bool {
        if self.did_initial_fit {
            return false;
        }

        let selection = self.scene.selection();
        if self.fit_suppressed
            || selection.selected_node.is_some()
            || selection.selected_edge.is_some()
        {
            if self.pending_pan_to_selection.is_none() {
                self.pending_pan_to_selection = selection.selected_node;
            }
            self.did_initial_fit = true;
            return false;
        }

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return false;
        };

        let graph_bounds = self.scene.layout_bounds();
        if graph_bounds.size.width <= 0 || graph_bounds.size.height <= 0 {
            return false;
        }

        let (margin_x, margin_y) = Self::viewport_margins_px(canvas_bounds);

        let available_width = f32::from(canvas_bounds.size.width - margin_x * 2.0);
        let available_height = f32::from(canvas_bounds.size.height - margin_y * 2.0);
        if available_width <= 0.0 || available_height <= 0.0 {
            return false;
        }

        let graph_width = graph_bounds.size.width as f32;
        let graph_height = graph_bounds.size.height as f32;
        if graph_width <= 0.0 || graph_height <= 0.0 {
            return false;
        }

        let mut target_zoom = (available_width / graph_width).min(available_height / graph_height);
        let mut camera = self.camera;
        camera.set_zoom(target_zoom);
        target_zoom = camera.zoom();

        let graph_center = gpui::point(
            graph_bounds.origin.x as f32 + (graph_width / 2.0),
            graph_bounds.origin.y as f32 + (graph_height / 2.0),
        );

        let viewport_center_screen = gpui::point(
            canvas_bounds.size.width / 2.0,
            canvas_bounds.size.height / 2.0,
        );
        let viewport_center_world = gpui::point(
            f32::from(viewport_center_screen.x) / target_zoom,
            f32::from(viewport_center_screen.y) / target_zoom,
        );

        let target_origin = gpui::point(
            graph_center.x - viewport_center_world.x,
            graph_center.y - viewport_center_world.y,
        );

        let did_start =
            self.try_start_camera_animation(target_origin, target_zoom, Duration::from_millis(260));
        if did_start {
            redesmyn_logging::tracing::info!(
                origin_world = ?target_origin,
                zoom = target_zoom,
                "graph initial fit-to-view"
            );
        }

        self.did_initial_fit = true;
        did_start
    }

    fn try_pan_to_selection_for_render(&mut self) -> bool {
        let Some(node_id) = self.pending_pan_to_selection else {
            return false;
        };

        if self.pan_drag.is_some() {
            self.pending_pan_to_selection = None;
            return false;
        }

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return false;
        };

        if self.scene.node(node_id).is_none() {
            self.pending_pan_to_selection = None;
            return false;
        }

        let node_rect = NodeWorldRect::from_scene(&self.scene, node_id);
        let zoom = self.camera.zoom();

        let (margin_x, margin_y) = Self::viewport_margins_px(canvas_bounds);
        // T-54 pan-to-selection defines a "safe rect" inside the canvas (screen space), then pans
        // the camera until the selected node's screen-space bounding box is fully inside it:
        //
        //   safe_min = (margin_x, margin_y)
        //   safe_max = (canvas_w - margin_x, canvas_h - margin_y)
        //
        // (Where `margin_*` is fixed-ish padding: ~72px or 25% of the viewport, whichever is
        // smaller.)
        //
        // This currently assumes the entire `canvas_bounds` region is usable/visible graph space.
        // That is not always true: we already (and will increasingly) draw UI overlays on top of
        // the canvas via absolutely positioned elements (multi-select selection bar, toasts,
        // floating toolbars, etc.). Those overlays occupy screen real estate but do not change
        // `canvas_bounds`, so the "safe rect" can include regions behind an overlay and a node can
        // be "in view" per this math while still being partially occluded.
        //
        // If we ever want a strict "never hidden" guarantee, this safe rect should incorporate
        // dynamic insets/exclusion zones, e.g.:
        //   safe_min.y += overlay_top_height;
        //   safe_max.y -= overlay_bottom_height;
        //   safe_max.x -= overlay_right_width;
        let safe_min = gpui::point(margin_x, margin_y);
        let safe_max = gpui::point(
            canvas_bounds.size.width - margin_x,
            canvas_bounds.size.height - margin_y,
        );

        let node_origin_screen = self.camera.world_to_screen(node_rect.origin);
        let node_size_screen = gpui::size(px(node_rect.width * zoom), px(node_rect.height * zoom));
        let node_max_screen = gpui::point(
            node_origin_screen.x + node_size_screen.width,
            node_origin_screen.y + node_size_screen.height,
        );

        let safe_width = safe_max.x - safe_min.x;
        let safe_height = safe_max.y - safe_min.y;

        let mut target_origin = self.camera.origin_world();

        let epsilon = px(1.0);
        let needs_center_x = node_size_screen.width + epsilon >= safe_width;
        let needs_center_y = node_size_screen.height + epsilon >= safe_height;

        if needs_center_x || needs_center_y {
            let node_center_world = gpui::point(
                node_rect.origin.x + node_rect.width / 2.0,
                node_rect.origin.y + node_rect.height / 2.0,
            );
            let viewport_center_world = gpui::point(
                f32::from(canvas_bounds.size.width / 2.0) / zoom,
                f32::from(canvas_bounds.size.height / 2.0) / zoom,
            );
            target_origin = gpui::point(
                node_center_world.x - viewport_center_world.x,
                node_center_world.y - viewport_center_world.y,
            );
        } else {
            if node_origin_screen.x < safe_min.x - epsilon {
                target_origin.x -= f32::from(safe_min.x - node_origin_screen.x) / zoom;
            } else if node_max_screen.x > safe_max.x + epsilon {
                target_origin.x += f32::from(node_max_screen.x - safe_max.x) / zoom;
            }

            if node_origin_screen.y < safe_min.y - epsilon {
                target_origin.y -= f32::from(safe_min.y - node_origin_screen.y) / zoom;
            } else if node_max_screen.y > safe_max.y + epsilon {
                target_origin.y += f32::from(node_max_screen.y - safe_max.y) / zoom;
            }
        }

        self.pending_pan_to_selection = None;

        if approx_eq_point(self.camera.origin_world(), target_origin) {
            return false;
        }

        let did_start =
            self.try_start_camera_animation(target_origin, zoom, Duration::from_millis(200));
        if did_start {
            redesmyn_logging::tracing::info!(
                node_id = %node_id,
                origin_world = ?target_origin,
                "graph pan-to-selection"
            );
        }
        did_start
    }

    fn start_demo_action(&mut self, node_id: GraphNodeId, cx: &mut Context<Self>) {
        if self.demo_action.in_flight {
            return;
        }

        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.task_card.demo_action.start",
            node_id = %node_id
        );
        let _guard = span.enter();

        self.demo_action.start();
        cx.notify();

        let view = cx.entity();
        self.demo_action_task = Some(cx.spawn(
            move |_: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    gpui::Timer::after(Duration::from_millis(900)).await;
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            this.demo_action.succeed();
                            this.demo_action_task = None;
                            cx.notify();

                            redesmyn_logging::tracing::info!(
                                node_id = %node_id,
                                "demo task card action finished"
                            );
                        })
                    });
                }
            },
        ));
    }

    fn on_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.button != MouseButton::Left {
            return;
        }

        self.cancel_camera_animation();
        self.pending_pan_to_selection = None;
        if !self.did_initial_fit {
            self.fit_suppressed = true;
        }

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return;
        };

        match hit_test(&self.scene, &self.camera, canvas_bounds, event.position) {
            Some(GraphHit::Node(id)) => {
                redesmyn_logging::tracing::trace!(node_id = %id, "graph selection changed");
                if event.modifiers.secondary() {
                    self.toggle_node(id, cx);
                } else {
                    self.select_node(id, cx);
                }
            }
            Some(GraphHit::Edge(id)) => {
                redesmyn_logging::tracing::trace!(edge_id = %id, "graph selection changed");
                self.select_edge(id, cx);
            }
            None => {
                let had_selection = !self.scene.selection().selected_nodes.is_empty()
                    || self.scene.selection().selected_edge.is_some();
                self.scene.clear_hover();
                self.pan_drag = Some(PanDrag {
                    start_mouse: event.position,
                    start_origin_world: self.camera.origin_world(),
                    pending_clear_selection: had_selection,
                    did_pan: false,
                });
            }
        }
    }

    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(mut drag) = self.pan_drag {
            if !event.dragging() {
                self.pan_drag = None;
                cx.notify();
                return;
            }

            let delta = event.position - drag.start_mouse;

            if drag.pending_clear_selection && !drag.did_pan {
                let threshold = px(3.0);
                if delta.x.abs() < threshold && delta.y.abs() < threshold {
                    return;
                }
                drag.pending_clear_selection = false;
                drag.did_pan = true;
            }

            let mut camera = self.camera;
            camera.set_origin_world(drag.start_origin_world);
            camera.pan_by_screen_delta(delta);
            self.camera = camera;
            self.pan_drag = Some(drag);

            cx.notify();
            return;
        }

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return;
        };

        let changed = match hit_test(&self.scene, &self.camera, canvas_bounds, event.position) {
            Some(GraphHit::Node(id)) => self.scene.set_hovered_node(Some(id)),
            Some(GraphHit::Edge(id)) => self.scene.set_hovered_edge(Some(id)),
            None => self.scene.clear_hover(),
        };

        if changed {
            cx.notify();
        }
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.pan_drag.take() else {
            return;
        };

        if drag.pending_clear_selection && !drag.did_pan {
            self.clear_selection(cx);
        } else {
            cx.notify();
        }
    }

    fn on_scroll_wheel(
        &mut self,
        event: &ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return;
        };

        let delta = event.delta.pixel_delta(window.line_height());
        if delta.x == px(0.0) && delta.y == px(0.0) {
            return;
        }

        self.cancel_camera_animation();
        self.pending_pan_to_selection = None;
        if !self.did_initial_fit {
            self.fit_suppressed = true;
        }

        let local_anchor = event.position - canvas_bounds.origin;

        if event.modifiers.secondary() {
            let dy = f32::from(delta.y);
            let factor = (-dy / 300.0).exp();
            self.camera.zoom_by_factor_at(factor, local_anchor);
            cx.notify();
            return;
        }

        if should_defer_pan_to_scroll_view(&self.expanded_session_scroll, event.position, delta)
            || should_defer_pan_to_scroll_view(&self.expanded_details_scroll, event.position, delta)
        {
            return;
        }

        self.camera.pan_by_screen_delta(delta);
        cx.notify();
    }

    fn selection_bar_target_visible(&self) -> bool {
        self.scene.selection().selected_edge.is_none() && self.scene.selection().selected_nodes.len() > 1
    }

    fn update_selection_bar_target(&mut self) {
        let target_visible = self.selection_bar_target_visible();
        let target = if target_visible { 1.0 } else { 0.0 };
        if (self.selection_bar_progress - target).abs() < f32::EPSILON {
            return;
        }

        self.selection_bar_transition = Some(SelectionBarTransition {
            started_at: Instant::now(),
            from: self.selection_bar_progress,
            to: target,
            duration: Duration::from_millis(180),
        });
    }

    fn tick_selection_bar_animation(&mut self, window: &mut Window) {
        let Some(transition) = self.selection_bar_transition else {
            return;
        };

        let elapsed = transition.started_at.elapsed();
        let t = (elapsed.as_secs_f32() / transition.duration.as_secs_f32()).clamp(0.0, 1.0);
        let eased = 1.0 - (1.0 - t).powi(3);
        self.selection_bar_progress = transition.from + (transition.to - transition.from) * eased;

        if t >= 1.0 {
            self.selection_bar_progress = transition.to;
            self.selection_bar_transition = None;
        } else {
            window.request_animation_frame();
        }
    }

    fn bulk_start_targets(&self) -> Vec<crate::GraphNodeId> {
        self.scene
            .selection()
            .selected_nodes
            .iter()
            .copied()
            .filter(|id| {
                if !matches!(id, crate::GraphNodeId::Task(_)) {
                    return false;
                }
                let Some(node) = self.scene.node(*id) else {
                    return false;
                };
                matches!(
                    node.state,
                    redesmyn_protocol::client::TaskState::Todo
                        | redesmyn_protocol::client::TaskState::InProgress
                        | redesmyn_protocol::client::TaskState::Unknown
                )
            })
            .collect()
    }

    fn bulk_start_disabled_reason(&self) -> Option<&'static str> {
        if self.bulk_start.action.in_flight {
            return Some("Starting agents…");
        }

        let selected_tasks = self
            .scene
            .selection()
            .selected_nodes
            .iter()
            .filter(|id| matches!(id, crate::GraphNodeId::Task(_)))
            .count();

        if self.bulk_start_targets().is_empty() {
            if selected_tasks == 0 {
                return Some("Select at least one task node");
            }
            return Some("Only todo/in-progress tasks can start agents");
        }
        None
    }

    fn start_bulk_start(&mut self, cx: &mut Context<Self>) {
        if let Some(reason) = self.bulk_start_disabled_reason() {
            redesmyn_logging::tracing::debug!(
                reason,
                "ignoring bulk start click (disabled)"
            );
            return;
        }

        let targets = self.bulk_start_targets();
        if targets.is_empty() {
            return;
        }

        let command_id = CommandId::new();
        let selected_count = self.scene.selection().selected_nodes.len();
        let eligible_count = targets.len();

        redesmyn_logging::tracing::info!(
            command_id = %command_id,
            selected_count,
            eligible_count,
            "dispatching bulk start (stub)"
        );

        self.bulk_start.action.start();
        self.bulk_start.command_id = Some(command_id);
        self.bulk_start.progress = Some(BulkProgress {
            completed: 0,
            total: eligible_count,
        });

        cx.notify();

        self.bulk_start.task = Some(cx.spawn(
            move |weak: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    for ix in 0..eligible_count {
                        gpui::Timer::after(Duration::from_millis(220)).await;
                        let Some(entity) = weak.upgrade() else {
                            return;
                        };
                        if cx
                            .update(|cx| {
                                entity.update(cx, |this, cx| {
                                    if let Some(progress) = &mut this.bulk_start.progress {
                                        progress.completed = (ix + 1).min(progress.total);
                                    }
                                    cx.notify();
                                })
                            })
                            .is_err()
                        {
                            return;
                        }
                    }

                    gpui::Timer::after(Duration::from_millis(220)).await;
                    let Some(entity) = weak.upgrade() else {
                        return;
                    };
                    let _ = cx.update(|cx| {
                        entity.update(cx, |this, cx| {
                            this.bulk_start.task = None;
                            this.bulk_start.progress = None;
                            this.bulk_start.command_id = None;
                            this.bulk_start.action.succeed();
                            cx.notify();
                        })
                    });
                    redesmyn_logging::tracing::info!(
                        command_id = %command_id,
                        "bulk start completed (stub)"
                    );
                }
            },
        ));

        // Keep the selection while the command runs; bulk actions should not implicitly clear it.
    }

    fn clear_bulk_start_error(&mut self, cx: &mut Context<Self>) {
        self.bulk_start.action.clear_error();
        cx.notify();
    }

    fn paint_graph(
        &self,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        window: &mut Window,
        cx: &App,
    ) {
        let theme = theme_for_window(window, cx);
        let t = self.layout_animation_progress_for_paint();
        let lod = edge_lod_band(self.camera.zoom());

        for edge in self.scene.edges() {
            let Some(from) = self.scene.node(edge.id.from) else {
                continue;
            };
            let Some(to) = self.scene.node(edge.id.to) else {
                continue;
            };

            let from_bounds = self.node_bounds_in_window_for_progress(from.id, canvas_bounds, t);
            let to_bounds = self.node_bounds_in_window_for_progress(to.id, canvas_bounds, t);

            let selected = self.scene.selection().selected_edge == Some(edge.id);
            let hovered = self.scene.selection().hovered_edge == Some(edge.id);

            let edge_color = if selected {
                theme.colors.ring
            } else if hovered {
                theme.colors.border.opacity(0.95)
            } else {
                theme.colors.border.opacity(0.65)
            };

            let thickness = if selected {
                px(3.0)
            } else if hovered {
                px(2.5)
            } else {
                DEFAULT_EDGE_STROKE_PX
            };

            let route = edge_route_in_window(from_bounds, to_bounds);
            for segment in route.segment_bounds(thickness).iter().flatten() {
                window.paint_quad(fill(*segment, edge_color));
            }

            if lod == EdgeLodBand::Ticks
                && let Some(commit_count) = edge.commit_count
            {
                self.paint_edge_ticks(route, commit_count, edge_color, window);
            }

            if lod >= EdgeLodBand::Labels
                && let Some(label) = edge.commit_count_label.as_ref()
            {
                self.paint_commit_count_label(route.label_center(), label, &theme, window);
            }
        }
    }
}

impl Focusable for GraphView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl GraphView {
    fn paint_edge_ticks(
        &self,
        route: EdgeRoute,
        commit_count: u32,
        color: gpui::Hsla,
        window: &mut Window,
    ) {
        if commit_count == 0 {
            return;
        }

        let [_, _, a, b] = route.points;
        if a.y != b.y {
            return;
        }

        let (left, right) = if a.x <= b.x { (a.x, b.x) } else { (b.x, a.x) };
        let length = right - left;
        if length <= px(12.0) {
            return;
        }

        let tick_thickness = px(1.0);
        let tick_height = px(6.0);
        let min_spacing = px(6.0);
        let max_by_space = (f32::from(length) / f32::from(min_spacing)).floor() as u32;
        let tick_count = commit_count.min(20).min(max_by_space);
        if tick_count == 0 {
            return;
        }

        let spacing = f32::from(length) / (tick_count as f32 + 1.0);
        for index in 1..=tick_count {
            let x = left + px(spacing * index as f32);
            let bounds = gpui::Bounds {
                origin: gpui::point(x - tick_thickness / 2.0, a.y - tick_height / 2.0),
                size: gpui::size(tick_thickness, tick_height),
            };
            window.paint_quad(fill(bounds, color.opacity(0.75)));
        }
    }

    fn paint_commit_count_label(
        &self,
        center: gpui::Point<gpui::Pixels>,
        text: &gpui::SharedString,
        theme: &redesmyn_ui::styles::UiTheme,
        window: &mut Window,
    ) {
        let font_size = px(11.0);
        let line_height = px(14.0);
        let padding_x = px(6.0);
        let padding_y = px(2.0);
        let text_color = theme.colors.foreground.opacity(0.8);

        if self.edge_label_cache.borrow().get(text).is_none() {
            let run = TextRun {
                len: text.len(),
                font: theme.typography.caption.font.clone(),
                color: text_color,
                background_color: None,
                underline: None,
                strikethrough: None,
            };

            let shaped = window.text_system().shape_line(
                text.clone(),
                font_size,
                std::slice::from_ref(&run),
                None,
            );

            self.edge_label_cache
                .borrow_mut()
                .insert(text.clone(), shaped);
        }

        let cache = self.edge_label_cache.borrow();
        let Some(shaped) = cache.get(text) else {
            return;
        };

        let bubble_size = gpui::size(
            shaped.width + padding_x * 2.0,
            line_height + padding_y * 2.0,
        );

        let bubble_bounds = gpui::Bounds {
            origin: gpui::point(
                center.x - bubble_size.width / 2.0,
                center.y - bubble_size.height / 2.0,
            ),
            size: bubble_size,
        };

        let bubble_radius = bubble_bounds.size.height / 2.0;
        let bubble = quad(
            bubble_bounds,
            bubble_radius,
            theme.colors.background.opacity(0.8),
            1.0,
            theme.colors.border.opacity(0.3),
            gpui::BorderStyle::Solid,
        );

        window.paint_quad(bubble);
        let text_origin = gpui::point(
            bubble_bounds.origin.x + padding_x,
            bubble_bounds.origin.y + padding_y,
        );
        self.paint_shaped_line(text_origin, line_height, shaped, text_color, window);
    }

    fn paint_shaped_line(
        &self,
        origin: gpui::Point<gpui::Pixels>,
        line_height: gpui::Pixels,
        shaped: &gpui::ShapedLine,
        color: gpui::Hsla,
        window: &mut Window,
    ) {
        let padding_top = (line_height - shaped.ascent - shaped.descent) / 2.0;
        let baseline_offset = gpui::point(px(0.0), padding_top + shaped.ascent);

        for run in shaped.runs.iter() {
            for glyph in run.glyphs.iter() {
                if glyph.is_emoji {
                    continue;
                }

                let glyph_origin = gpui::point(origin.x + glyph.position.x, origin.y);
                let _ = window.paint_glyph(
                    glyph_origin + baseline_offset,
                    run.font_id,
                    glyph.id,
                    shaped.font_size,
                    color,
                );
            }
        }
    }
}

impl Render for GraphView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.step_camera_animation_for_render(window);
        let started_fit = self.try_initial_fit_for_render();
        let started_pan = self.try_pan_to_selection_for_render();
        if started_fit || started_pan {
            window.request_animation_frame();
        }

        let theme = theme_for_window(window, cx);
        let label = self.selection_label();
        let zoom = self.camera.zoom();
        let controls_hint = if cfg!(target_os = "macos") {
            "Pan: two-finger scroll/drag · Zoom: ⌘ + scroll"
        } else {
            "Pan: scroll/drag · Zoom: Ctrl + scroll"
        };

        let graph = cx.entity();
        let graph_for_prepaint = graph.clone();
        let graph_for_paint = graph.clone();

        let debug_bar = div()
            .h(px(30.0))
            .px(theme.spacing.md)
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .bg(theme.colors.surface_elevated)
            .border_b_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child(label),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.md)
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(controls_hint)
                    .child(format!("Zoom: {:.2}", zoom)),
            );

        let canvas = canvas(
            move |bounds, _window, cx| {
                graph_for_prepaint.update(cx, |this, _cx| {
                    this.last_canvas_bounds = Some(bounds);
                });
            },
            move |bounds, (), window, cx| {
                let view = graph_for_paint.read(cx);
                view.paint_graph(bounds, window, cx);
            },
        )
        .size_full();

        let t = self.layout_animation_progress_for_render(window);

        let nodes_layer = if let Some(canvas_bounds) = self.last_canvas_bounds {
            let entity_id = cx.entity_id();
            let primary_selected = self.visual_selected_node();
            let selected_nodes = &self.scene.selection().selected_nodes;

            let mut layer = div().absolute().inset_0();

            for node in self.scene.nodes() {
                let bounds_in_window =
                    self.node_bounds_in_window_for_progress(node.id, canvas_bounds, t);
                let local_origin = bounds_in_window.origin - canvas_bounds.origin;
                let node_id = node.id;
                let node_key: gpui::SharedString = node_id.to_string().into();
                let is_primary_selected = primary_selected == Some(node_id);
                let is_selected = is_primary_selected || selected_nodes.contains(&node_id);
                let is_expanded = is_primary_selected;

                let bg = if is_selected {
                    theme.colors.ring.opacity(0.10)
                } else {
                    theme.colors.surface_elevated
                };
                let border = if is_selected {
                    theme.colors.ring
                } else {
                    theme.colors.border.opacity(0.8)
                };

                let node_element_id = (
                    gpui::ElementId::from(("graph_node", entity_id)),
                    node_key.clone(),
                );

                let mut card = div()
                    .id(node_element_id)
                    .absolute()
                    .top(local_origin.y)
                    .left(local_origin.x)
                    .w(bounds_in_window.size.width)
                    .h(bounds_in_window.size.height)
                    .rounded(theme.radius.md)
                    .bg(bg)
                    .border_1()
                    .border_color(border)
                    .overflow_hidden()
                    .block_mouse_except_scroll()
                    .on_mouse_down(MouseButton::Left, {
                        let focus_handle = self.focus_handle.clone();
                        let graph = graph.clone();
                        move |event, window, cx| {
                            focus_handle.focus(window);
                            graph.update(cx, |this, cx| {
                                if event.modifiers.secondary() {
                                    this.toggle_node(node_id, cx);
                                } else {
                                    this.select_node(node_id, cx);
                                }
                            });
                            cx.stop_propagation();
                        }
                    });

                if !is_expanded {
                    let task_slug = node.task_slug.clone();
                    let padding_x = px(f32::from(theme.spacing.sm) * zoom);
                    let padding_y = px(f32::from(theme.spacing.xs) * zoom);

                    card = card.child(
                        div()
                            .px(padding_x)
                            .py(padding_y)
                            .size_full()
                            .flex()
                            .items_center()
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_size(rems(0.82 * zoom))
                                    .text_color(theme.colors.foreground)
                                    .truncate()
                                    .child(task_slug),
                            ),
                    );
                } else {
                    let title = node.title.clone();
                    let task_slug = node.task_slug.clone();
                    let state = node.state;
                    let merge_readiness = node.merge_readiness;
                    let agent_status = node.agent_status;
                    let branch_name = node.branch_name.clone();

                    let close_button_id = (
                        gpui::ElementId::from(("task_card_close", entity_id)),
                        node_key.clone(),
                    );
                    let collapse = {
                        let graph = graph.clone();
                        move |_: &ClickEvent, _window: &mut Window, cx: &mut App| {
                            graph.update(cx, |this, cx| this.clear_selection(cx));
                        }
                    };

                    let action_button_id = (
                        gpui::ElementId::from(("task_card_demo_action", entity_id)),
                        node_key.clone(),
                    );
                    let start_action = {
                        let graph = graph.clone();
                        move |_: &ClickEvent, _window: &mut Window, cx: &mut App| {
                            graph.update(cx, |this, cx| this.start_demo_action(node_id, cx));
                        }
                    };

                    let left_scroll = self.expanded_session_scroll.clone();
                    let right_scroll = self.expanded_details_scroll.clone();
                    let demo_action = self.demo_action.clone();

                    card = card.child(
                        div()
                            .flex()
                            .flex_col()
                            .size_full()
                            .child(
                                div()
                                    .h(px(44.0))
                                    .px(theme.spacing.md)
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .justify_between()
                                    .bg(theme.colors.surface)
                                    .border_b_1()
                                    .border_color(theme.colors.border.opacity(0.5))
                                            .child(
                                                div()
                                                    .flex()
                                                    .flex_row()
                                                    .items_center()
                                                    .gap(theme.spacing.sm)
                                                    .child(
                                                        div()
                                                            .text_sm()
                                                            .text_color(theme.colors.foreground)
                                                            .truncate()
                                                            .child(title),
                                                    )
                                                    .child(task_status_chips(
                                                        state,
                                                        merge_readiness,
                                                        agent_status,
                                                        &theme,
                                                    ))
                                                    .child(
                                                        div()
                                                            .min_w_0()
                                                            .text_sm()
                                                            .text_color(theme.colors.foreground_muted)
                                                            .truncate()
                                                            .child(task_slug),
                                                    ),
                                            )
                                            .child(
                                                IconButton::new(close_button_id, div().child("×"))
                                            .tooltip("Collapse")
                                            .on_click(collapse),
                                    ),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_h(px(0.0))
                                    .flex()
                                    .flex_row()
                                    .bg(theme.colors.surface)
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .flex_1()
                                            .min_w_0()
                                            .border_r_1()
                                            .border_color(theme.colors.border.opacity(0.5))
                                            .child(
                                                div()
                                                    .h(px(34.0))
                                                    .px(theme.spacing.md)
                                                    .flex()
                                                    .items_center()
                                                    .text_sm()
                                                    .text_color(theme.colors.foreground)
                                                    .child("Session"),
                                            )
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_h(px(0.0))
                                                    .p(theme.spacing.md)
                                                    .child(
                                                        ScrollArea::new(
                                                            (
                                                                gpui::ElementId::from((
                                                                    "task_card_session_scroll",
                                                                    entity_id,
                                                                )),
                                                                node_key.clone(),
                                                            ),
                                                            left_scroll,
                                                        )
                                                        .child(
                                                            div()
                                                                .flex()
                                                                .flex_col()
                                                                .gap(theme.spacing.sm)
                                                                .child(
                                                                    div()
                                                                        .text_sm()
                                                                        .text_color(
                                                                            theme.colors.foreground_muted,
                                                                        )
                                                                        .child(
                                                                            "SessionView placeholder (T-64).",
                                                                        ),
                                                                )
                                                                .child(
                                                                    placeholder_message(
                                                                        "Incoming session events will render here.",
                                                                        &theme,
                                                                    ),
                                                                )
                                                                .child(
                                                                    placeholder_message(
                                                                        "Composer UI will live at the bottom.",
                                                                        &theme,
                                                                    ),
                                                                ),
                                                        ),
                                                    ),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .w(px(360.0))
                                            .min_w_0()
                                            .child(
                                                div()
                                                    .h(px(34.0))
                                                    .px(theme.spacing.md)
                                                    .flex()
                                                    .items_center()
                                                    .text_sm()
                                                    .text_color(theme.colors.foreground)
                                                    .child("Details"),
                                            )
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_h(px(0.0))
                                                    .px(theme.spacing.md)
                                                    .pb(theme.spacing.md)
                                                    .child(
                                                        ScrollArea::new(
                                                            (
                                                                gpui::ElementId::from((
                                                                    "task_card_details_scroll",
                                                                    entity_id,
                                                                )),
                                                                node_key.clone(),
                                                            ),
                                                            right_scroll,
                                                        )
                                                        .child(task_details(
                                                            entity_id,
                                                            node_id,
                                                            node_key.clone(),
                                                            state,
                                                            merge_readiness,
                                                            agent_status,
                                                            branch_name,
                                                            action_button_id,
                                                            demo_action,
                                                            start_action,
                                                            &theme,
                                                        )),
                                                    ),
                                            ),
                                    ),
                            ),
                    );
                }

                layer = layer.child(card);
            }

            layer
        } else {
            div().absolute().inset_0()
        };

        self.tick_selection_bar_animation(window);

        let selected_count = self.scene.selection().selected_nodes.len();
        let bar_progress = self.selection_bar_progress.clamp(0.0, 1.0);
        let show_selection_bar = bar_progress > 0.001;
        let selection_bar_top = theme.spacing.md - px((1.0 - bar_progress) * 56.0);

        let selection_bar = if show_selection_bar {
            let disabled_reason = self.bulk_start_disabled_reason();
            let eligible = self.bulk_start_targets().len();

            let start_label = if let Some(progress) = self.bulk_start.progress {
                format!("Start {}/{}", progress.completed, progress.total)
            } else {
                "Start agents".to_string()
            };

            Some(
                div()
                    .id(("selection_bar", cx.entity_id()))
                    .absolute()
                    .top(selection_bar_top)
                    .left(px(0.0))
                    .right(px(0.0))
                    .opacity(bar_progress)
                    .flex()
                    .justify_center()
                    .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(theme.spacing.md)
                            .px(theme.spacing.md)
                            .py(theme.spacing.sm)
                            .rounded(theme.radius.lg)
                            .bg(theme.colors.background.opacity(0.88))
                            .border_1()
                            .border_color(theme.colors.border.opacity(0.6))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(format!("{selected_count} selected")),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .gap(theme.spacing.sm)
                                    .child(
                                        TextButton::new(("bulk_start", cx.entity_id()), start_label)
                                            .kind(ButtonKind::Secondary)
                                            .disabled(disabled_reason.is_some())
                                            .when_some(disabled_reason, |this, reason| {
                                                this.disabled_reason(reason)
                                            })
                                            .tooltip(format!(
                                                "Starts agents for {eligible} of {selected_count} selected task(s)."
                                            ))
                                            .on_click({
                                                let graph = graph.clone();
                                                move |_, _, cx| {
                                                    graph.update(cx, |this, cx| {
                                                        this.start_bulk_start(cx);
                                                    });
                                                }
                                            }),
                                    )
                                    .child(
                                        TextButton::new(("bulk_clear", cx.entity_id()), "Clear")
                                            .kind(ButtonKind::Ghost)
                                            .on_click({
                                                let graph = graph.clone();
                                                move |_, _, cx| {
                                                    graph.update(cx, |this, cx| {
                                                        this.clear_selection(cx);
                                                    });
                                                }
                                            }),
                                    )
                                    .when(self.bulk_start.action.in_flight, |this| {
                                        let label = "Starting";
                                        this.child(
                                            ProgressPill::new(label).kind(ProgressPillKind::Accent),
                                        )
                                    })
                                    .when_some(self.bulk_start.action.error.clone(), |this, error| {
                                        this.child(
                                            TextButton::new(("bulk_error", cx.entity_id()), error)
                                                .kind(ButtonKind::Danger)
                                                .tooltip("Dismiss error")
                                                .on_click({
                                                    let graph = graph.clone();
                                                    move |_, _, cx| {
                                                        graph.update(cx, |this, cx| {
                                                            this.clear_bulk_start_error(cx);
                                                        });
                                                    }
                                                }),
                                        )
                                    }),
                            ),
                    ),
            )
        } else {
            None
        };

        div()
            .id(("graph_view", cx.entity_id()))
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .focusable()
            .key_context("Graph")
            .capture_key_down({
                let graph = graph.clone();
                move |event, _window, cx| {
                    if event.keystroke.key != "escape" {
                        return;
                    }

                    let did_clear = graph.update(cx, |this, cx| {
                        let had_selection = !this.scene.selection().selected_nodes.is_empty()
                            || this.scene.selection().selected_edge.is_some();
                        if !had_selection {
                            return false;
                        }

                        this.clear_selection(cx);
                        true
                    });

                    if did_clear {
                        cx.stop_propagation();
                    }
                }
            })
            .child(debug_bar)
            .child(
                div()
                    .id(("graph_canvas", cx.entity_id()))
                    .flex_1()
                    .overflow_hidden()
                    .relative()
                    .cursor(if self.pan_drag.is_some() {
                        CursorStyle::ClosedHand
                    } else if self.scene.selection().hovered_node.is_some()
                        || self.scene.selection().hovered_edge.is_some()
                    {
                        CursorStyle::PointingHand
                    } else {
                        CursorStyle::Arrow
                    })
                    .focusable()
                    .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
                    .on_mouse_move(cx.listener(Self::on_mouse_move))
                    .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
                    .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
                    .on_scroll_wheel(cx.listener(Self::on_scroll_wheel))
                    .child(canvas)
                    .child(nodes_layer)
                    .when_some(selection_bar, |this, selection_bar| {
                        this.child(selection_bar)
                    }),
            )
            .track_focus(&self.focus_handle(cx))
    }
}

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_point(a: gpui::Point<f32>, b: gpui::Point<f32>, t: f32) -> gpui::Point<f32> {
    gpui::point(lerp_f32(a.x, b.x, t), lerp_f32(a.y, b.y, t))
}

fn lerp_world_rect(from: NodeWorldRect, to: NodeWorldRect, t: f32) -> NodeWorldRect {
    NodeWorldRect {
        origin: lerp_point(from.origin, to.origin, t),
        width: lerp_f32(from.width, to.width, t),
        height: lerp_f32(from.height, to.height, t),
    }
}

fn approx_eq_point(a: gpui::Point<f32>, b: gpui::Point<f32>) -> bool {
    (a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3
}

fn should_defer_pan_to_scroll_view(
    scroll_handle: &ScrollHandle,
    window_point: gpui::Point<gpui::Pixels>,
    delta: gpui::Point<gpui::Pixels>,
) -> bool {
    if !scroll_handle.bounds().contains(&window_point) {
        return false;
    }

    let max_y = scroll_handle.max_offset().height;
    if max_y == px(0.0) {
        return false;
    }

    let delta_y = if delta.y == px(0.0) { delta.x } else { delta.y };
    if delta_y == px(0.0) {
        return false;
    }

    let offset_y = scroll_handle.offset().y;
    let epsilon = px(1.0);

    if delta_y > px(0.0) {
        // Scrolling "up": allow pan only when already at the top edge.
        offset_y < -epsilon
    } else {
        // Scrolling "down": allow pan only when already at the bottom edge.
        offset_y > (-max_y + epsilon)
    }
}

fn status_chip(
    label: impl Into<gpui::SharedString>,
    bg: gpui::Hsla,
    fg: gpui::Hsla,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    div()
        .px(theme.spacing.sm)
        .py(theme.spacing.xs)
        .rounded(theme.radius.md)
        .bg(bg)
        .text_sm()
        .text_color(fg)
        .child(label.into())
}

fn task_state_label(state: TaskState) -> &'static str {
    match state {
        TaskState::Todo => "Todo",
        TaskState::InProgress => "In progress",
        TaskState::Blocked => "Blocked",
        TaskState::Done => "Done",
        TaskState::Unknown => "Unknown",
    }
}

fn task_state_chip(state: TaskState, theme: &redesmyn_ui::styles::UiTheme) -> impl IntoElement {
    let (bg, fg) = match state {
        TaskState::InProgress => (theme.colors.ring.opacity(0.18), theme.colors.foreground),
        TaskState::Blocked => (theme.colors.warning.opacity(0.18), theme.colors.foreground),
        TaskState::Done => (
            theme.colors.accent.opacity(0.75),
            theme.colors.foreground_muted,
        ),
        TaskState::Todo | TaskState::Unknown => {
            (theme.colors.surface_elevated, theme.colors.foreground_muted)
        }
    };
    status_chip(task_state_label(state), bg, fg, theme)
}

fn merge_readiness_label(readiness: MergeReadiness) -> &'static str {
    match readiness {
        MergeReadiness::Ready => "Merge ready",
        MergeReadiness::Blocked => "Merge blocked",
        MergeReadiness::Unknown => "Merge ?",
    }
}

fn merge_readiness_chip(
    readiness: MergeReadiness,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    let (bg, fg) = match readiness {
        MergeReadiness::Ready => (theme.colors.ring.opacity(0.18), theme.colors.foreground),
        MergeReadiness::Blocked => (theme.colors.warning.opacity(0.18), theme.colors.foreground),
        MergeReadiness::Unknown => (theme.colors.surface_elevated, theme.colors.foreground_muted),
    };
    status_chip(merge_readiness_label(readiness), bg, fg, theme)
}

fn agent_status_label(status: AgentStatus) -> &'static str {
    match status {
        AgentStatus::Running => "Agent running",
        AgentStatus::Blocked => "Agent blocked",
        AgentStatus::Stopped => "Agent stopped",
        AgentStatus::Error => "Agent error",
        AgentStatus::Unknown => "Agent ?",
    }
}

fn agent_status_chip(
    status: AgentStatus,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    let (bg, fg) = match status {
        AgentStatus::Running => (theme.colors.ring.opacity(0.18), theme.colors.foreground),
        AgentStatus::Blocked => (theme.colors.warning.opacity(0.18), theme.colors.foreground),
        AgentStatus::Error => (theme.colors.danger.opacity(0.18), theme.colors.foreground),
        AgentStatus::Stopped | AgentStatus::Unknown => {
            (theme.colors.surface_elevated, theme.colors.foreground_muted)
        }
    };
    status_chip(agent_status_label(status), bg, fg, theme)
}

fn task_status_chips(
    state: TaskState,
    merge_readiness: MergeReadiness,
    agent_status: AgentStatus,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .gap(theme.spacing.xs)
        .items_center()
        .child(task_state_chip(state, theme))
        .child(merge_readiness_chip(merge_readiness, theme))
        .child(agent_status_chip(agent_status, theme))
}

fn placeholder_message(
    text: impl Into<gpui::SharedString>,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    div()
        .p(theme.spacing.md)
        .rounded(theme.radius.md)
        .bg(theme.colors.accent.opacity(0.7))
        .text_sm()
        .text_color(theme.colors.foreground_muted)
        .child(text.into())
}

fn details_kv_row(
    label: &'static str,
    value: impl IntoElement,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .items_start()
        .gap(theme.spacing.sm)
        .child(
            div()
                .w(px(110.0))
                .text_sm()
                .text_color(theme.colors.foreground_muted)
                .child(label),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .text_sm()
                .text_color(theme.colors.foreground)
                .child(value),
        )
}

fn details_section(
    title: &'static str,
    body: impl IntoElement,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    div()
        .pt(theme.spacing.md)
        .child(
            div()
                .pb(theme.spacing.sm)
                .text_sm()
                .text_color(theme.colors.foreground)
                .child(title),
        )
        .child(body)
        .child(
            div()
                .pt(theme.spacing.md)
                .border_b_1()
                .border_color(theme.colors.border.opacity(0.5)),
        )
}

fn task_details(
    entity_id: gpui::EntityId,
    node_id: GraphNodeId,
    node_key: gpui::SharedString,
    state: TaskState,
    merge_readiness: MergeReadiness,
    agent_status: AgentStatus,
    branch_name: Option<gpui::SharedString>,
    action_button_id: impl Into<gpui::ElementId>,
    demo_action: UserActionState,
    start_demo_action: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    let mut root = div()
        .flex()
        .flex_col()
        .gap(theme.spacing.md)
        .pb(theme.spacing.md)
        .child(
            div()
                .flex()
                .flex_row()
                .gap(theme.spacing.sm)
                .items_center()
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Jump"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("·"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Overview"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Agent"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Merge"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("IDs"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Artifacts"),
                ),
        );

    if let Some(error) = demo_action.error.clone() {
        root = root.child(
            Callout::new(error)
                .kind(CalloutKind::Danger)
                .title("Action failed"),
        );
    }

    root.child(details_section(
        "Overview",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(details_kv_row(
                "Task",
                div().child(format!("{node_id}")),
                theme,
            ))
            .when_some(branch_name.clone(), |this, name| {
                this.child(details_kv_row("Branch", div().child(name), theme))
            })
            .child(details_kv_row(
                "State",
                task_state_chip(state, theme),
                theme,
            ))
            .child(details_kv_row(
                "Merge",
                merge_readiness_chip(merge_readiness, theme),
                theme,
            ))
            .child(details_kv_row("Updated", div().child("—"), theme)),
        theme,
    ))
    .child(details_section(
        "Agent",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(theme.spacing.xs)
                    .child(details_kv_row(
                        "Status",
                        agent_status_chip(agent_status, theme),
                        theme,
                    ))
                    .child(details_kv_row("Session", div().child("unbound"), theme)),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        TextButton::new(action_button_id, "Run demo action")
                            .kind(ButtonKind::Secondary)
                            .disabled(demo_action.in_flight)
                            .disabled_reason("Running…")
                            .on_click(start_demo_action),
                    )
                    .when(demo_action.in_flight, |this| {
                        this.child(ProgressPill::new("Running"))
                    }),
            ),
        theme,
    ))
    .child(details_section(
        "Merge / restack",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(details_kv_row(
                "Ready",
                merge_readiness_chip(merge_readiness, theme),
                theme,
            ))
            .child(details_kv_row("Next", div().child("—"), theme)),
        theme,
    ))
    .child(details_section(
        "Identifiers",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(details_kv_row(
                "Task id",
                div().child(format!("{node_id}")),
                theme,
            ))
            .child(details_kv_row("Run id", div().child("—"), theme))
            .child(details_kv_row("Session id", div().child("—"), theme)),
        theme,
    ))
    .child(details_section(
        "Artifacts / logs",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .child(
                TextButton::new(
                    (
                        gpui::ElementId::from(("task_card_logs", entity_id)),
                        node_key.clone(),
                    ),
                    "Open logs",
                )
                .kind(ButtonKind::Ghost)
                .disabled(true)
                .disabled_reason("Coming soon"),
            )
            .child(
                TextButton::new(
                    (
                        gpui::ElementId::from(("task_card_diffs", entity_id)),
                        node_key.clone(),
                    ),
                    "View diffs",
                )
                .kind(ButtonKind::Ghost)
                .disabled(true)
                .disabled_reason("Coming soon"),
            )
            .child(
                TextButton::new(
                    (
                        gpui::ElementId::from(("task_card_merge", entity_id)),
                        node_key,
                    ),
                    "Merge / restack",
                )
                .kind(ButtonKind::Ghost)
                .disabled(true)
                .disabled_reason("Coming soon"),
            ),
        theme,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_bar_visibility_requires_multi_selection() {
        let mut scene = GraphScene::empty_demo();
        let a = GraphNodeId::Task(redesmyn_ids::TaskId::from_bytes([1; 16]));
        let b = GraphNodeId::Task(redesmyn_ids::TaskId::from_bytes([2; 16]));
        scene.insert_demo_node(a);
        scene.insert_demo_node(b);

        let selection_bar_visible =
            |scene: &GraphScene| scene.selection().selected_edge.is_none()
                && scene.selection().selected_nodes.len() > 1;

        assert!(!selection_bar_visible(&scene));

        scene.select_node(a);
        assert!(!selection_bar_visible(&scene));

        scene.toggle_node(b);
        assert!(selection_bar_visible(&scene));

        scene.select_edge(GraphEdgeId { from: a, to: b });
        assert!(!selection_bar_visible(&scene));
    }

    #[test]
    fn bulk_start_targets_respect_task_state() {
        let mut scene = GraphScene::demo();
        let a = GraphNodeId::Task(redesmyn_ids::TaskId::from_bytes([1; 16]));
        let b = GraphNodeId::Task(redesmyn_ids::TaskId::from_bytes([2; 16]));
        let c = GraphNodeId::Task(redesmyn_ids::TaskId::from_bytes([3; 16]));
        let d = GraphNodeId::Task(redesmyn_ids::TaskId::from_bytes([4; 16]));

        scene.toggle_node(a);
        scene.toggle_node(b);
        scene.toggle_node(c);
        scene.toggle_node(d);

        let targets: Vec<_> = scene
            .selection()
            .selected_nodes
            .iter()
            .copied()
            .filter(|id| matches!(id, GraphNodeId::Task(_)))
            .filter(|id| {
                let Some(node) = scene.node(*id) else {
                    return false;
                };
                matches!(
                    node.state,
                    TaskState::Todo | TaskState::InProgress | TaskState::Unknown
                )
            })
            .collect();

        assert_eq!(targets, vec![a, c]);
    }
}
