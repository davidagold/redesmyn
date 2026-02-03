use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

use gpui::{
    App, AsyncApp, ClickEvent, Context, CursorStyle, Entity, FocusHandle, Focusable, FontWeight,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, ScrollHandle,
    ScrollWheelEvent, Subscription, Task, TextRun, Window, canvas, div, fill, prelude::*, px, quad,
    rems,
};

use redesmyn_ids::CommandId;
use redesmyn_ids::TaskId;

use redesmyn_client_api::Client;

use redesmyn_ui::components::{
    Badge, BadgeKind, ButtonKind, Callout, CalloutKind, IconButton,
    MarkdownInlineSingleLineContent, ProgressPill, ProgressPillKind, ScrollArea, TextButton,
    Tooltip,
};
use redesmyn_ui::utils::{
    BoundedCache, TransitionMap, UiActivityGuard, UserActionState, theme_for_window,
    ui_idle_tracker, ui_test_mode_animation_duration,
};

use crate::camera::{GraphCamera, GraphCameraLimits};
use crate::constants::{
    TRUNK_LABEL_LOD_ZOOM, TRUNK_MARKER_WIDTH, TRUNK_THICKNESS, TRUNK_TITLE_WIDTH,
};
use crate::culling::{DEFAULT_NODE_CULLING_OVERSCAN_PX, viewport_bounds_with_overscan};
use crate::geometry::{
    DEFAULT_EDGE_STROKE_PX, EdgeLodBand, EdgeRoute, edge_lod_band,
    edge_route_between_points_in_window, edge_route_in_window,
};
use crate::hit_test::{GraphHit, hit_test};
use crate::scene::{AgentStatus, GraphEdgeId, GraphNodeId, GraphScene, TrunkMarkKind};

use redesmyn_markdown::{MarkdownParseOptions, parse_markdown};
use redesmyn_protocol::client::{
    AgentInterfaceMode, AgentKind, AgentMessageConflictAction, MergeReadiness, RequestPayload,
    ResponseResult, RestartAgentRequest, StartAgentRequest, StopAgentRequest, TaskState,
};
use redesmyn_ui_session::{SessionView, SessionViewEvent, TaskSessionOperation};

#[derive(Debug, Clone)]
struct FpsOverlay {
    enabled: bool,
    last_frame_at: Option<Instant>,
    ema_frame_time_s: Option<f32>,
    last_label_update: Option<Instant>,
    label: gpui::SharedString,
}

impl FpsOverlay {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last_frame_at: None,
            ema_frame_time_s: None,
            last_label_update: None,
            label: "FPS: —".into(),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn on_frame(&mut self) {
        if !self.enabled {
            return;
        }

        let now = Instant::now();
        if let Some(prev) = self.last_frame_at {
            let dt = (now - prev).as_secs_f32().max(0.0);
            if dt > 0.0 {
                const ALPHA: f32 = 0.12;
                let ema = self
                    .ema_frame_time_s
                    .map(|ema| ema * (1.0 - ALPHA) + dt * ALPHA)
                    .unwrap_or(dt);
                self.ema_frame_time_s = Some(ema);
            }
        }
        self.last_frame_at = Some(now);

        let should_update_label = self
            .last_label_update
            .is_none_or(|last| now.duration_since(last) >= Duration::from_millis(250));
        if should_update_label {
            if let Some(dt) = self.ema_frame_time_s {
                let fps = (1.0 / dt).clamp(0.0, 999.0);
                self.label = format!("FPS: {fps:.0}").into();
            }
            self.last_label_update = Some(now);
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskQuickActionKind {
    Start,
    Restart,
    Stop,
}

#[derive(Debug, Default)]
struct TaskQuickActionState {
    start: UserActionState,
    start_task: Option<Task<()>>,
    restart: UserActionState,
    restart_task: Option<Task<()>>,
    stop: UserActionState,
    stop_task: Option<Task<()>>,
}

#[derive(Debug, Clone)]
struct CollapsedTitleCacheEntry {
    title: gpui::SharedString,
    wrap_width_px: i32,
    font_size_px: i32,
    line1: gpui::SharedString,
    line2: Option<gpui::SharedString>,
}

pub struct GraphView {
    focus_handle: FocusHandle,
    scene: GraphScene,
    camera: GraphCamera,
    fps_overlay: FpsOverlay,
    camera_animation: Option<CameraAnimation>,
    camera_animation_guard: Option<UiActivityGuard>,
    pan_drag: Option<PanDrag>,
    last_canvas_bounds: Option<gpui::Bounds<gpui::Pixels>>,
    task_session_view: Entity<SessionView>,
    expanded_details_scroll: ScrollHandle,
    demo_action: UserActionState,
    demo_action_task: Option<Task<()>>,
    layout_animation: Option<LayoutAnimation>,
    layout_animation_guard: Option<UiActivityGuard>,
    pending_pan_to_selection: Option<GraphNodeId>,
    task_focus_zoom_before: Option<f32>,
    pending_focus_task_card: Option<GraphNodeId>,
    pending_restore_task_focus_zoom: Option<f32>,
    did_initial_fit: bool,
    fit_suppressed: bool,
    edge_label_cache: RefCell<HashMap<gpui::SharedString, gpui::ShapedLine>>,
    selection_bar_progress: f32,
    selection_bar_transition: Option<SelectionBarTransition>,
    selection_bar_guard: Option<UiActivityGuard>,
    bulk_start: BulkCommandState,
    quick_actions: HashMap<TaskId, TaskQuickActionState>,
    quick_action_opacity: TransitionMap<TaskId>,
    collapsed_title_cache: HashMap<TaskId, CollapsedTitleCacheEntry>,
    collapsed_markdown_cache:
        BoundedCache<redesmyn_ids::SessionEventId, MarkdownInlineSingleLineContent>,
    _subscriptions: Vec<Subscription>,
}

impl GraphView {
    pub fn new_empty(task_session_view: Entity<SessionView>, cx: &mut Context<Self>) -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_view.new_empty");
        let _guard = span.enter();

        Self::new_with_scene(GraphScene::empty_demo(), task_session_view, cx)
    }

    pub fn new_demo(task_session_view: Entity<SessionView>, cx: &mut Context<Self>) -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_view.new_demo");
        let _guard = span.enter();

        Self::new_with_scene(GraphScene::demo(), task_session_view, cx)
    }

    fn new_with_scene(
        scene: GraphScene,
        task_session_view: Entity<SessionView>,
        cx: &mut Context<Self>,
    ) -> Self {
        let fps_overlay_enabled = std::env::var("REDESMYN_UI_FPS_OVERLAY")
            .ok()
            .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true"));
        let mut subscriptions = Vec::new();
        subscriptions.push(cx.subscribe(
            &task_session_view,
            |_this, _, _: &SessionViewEvent, cx: &mut Context<Self>| {
                // Session state drives card chrome (loading/empty callouts), so keep the graph view
                // in sync when the embedded SessionView updates.
                cx.notify();
            },
        ));

        let mut this = Self {
            focus_handle: cx.focus_handle(),
            scene,
            camera: GraphCamera::new(GraphCameraLimits::default()),
            fps_overlay: FpsOverlay::new(fps_overlay_enabled),
            camera_animation: None,
            camera_animation_guard: None,
            pan_drag: None,
            last_canvas_bounds: None,
            task_session_view,
            expanded_details_scroll: ScrollHandle::new(),
            demo_action: UserActionState::default(),
            demo_action_task: None,
            layout_animation: None,
            layout_animation_guard: None,
            pending_pan_to_selection: None,
            task_focus_zoom_before: None,
            pending_focus_task_card: None,
            pending_restore_task_focus_zoom: None,
            did_initial_fit: false,
            fit_suppressed: false,
            edge_label_cache: RefCell::new(HashMap::new()),
            selection_bar_progress: 0.0,
            selection_bar_transition: None,
            selection_bar_guard: None,
            bulk_start: BulkCommandState::default(),
            quick_actions: HashMap::new(),
            quick_action_opacity: TransitionMap::new(),
            collapsed_title_cache: HashMap::new(),
            collapsed_markdown_cache: BoundedCache::new(512),
            _subscriptions: subscriptions,
        };

        let selected = this.scene.selection().selected_node;
        this.sync_task_session_view(selected, cx);
        this
    }

    pub fn replace_from_epic_graph(
        &mut self,
        graph: &redesmyn_protocol::client::EpicGraph,
        cx: &mut Context<Self>,
    ) {
        let previous_selection = self.scene.selection().clone();
        let from_layout = self.snapshot_displayed_layout();

        self.scene.replace_from_epic_graph(graph);

        self.pan_drag = None;
        self.edge_label_cache.borrow_mut().clear();
        self.quick_action_opacity
            .retain(|task_id| self.scene.node(GraphNodeId::Task(*task_id)).is_some());
        self.collapsed_title_cache
            .retain(|task_id, _| self.scene.node(GraphNodeId::Task(*task_id)).is_some());
        let referenced_session_events: HashSet<redesmyn_ids::SessionEventId> = self
            .scene
            .nodes()
            .filter_map(|node| {
                let session = node.latest_session.as_ref()?;
                session
                    .message_preview
                    .as_ref()
                    .map(|_| session.session_event_id)
            })
            .collect();
        self.collapsed_markdown_cache
            .retain(|event_id, _| referenced_session_events.contains(event_id));

        let next_selection = self.scene.selection().clone();
        if previous_selection.selected_node != next_selection.selected_node {
            self.reset_expanded_card_state();
            self.sync_task_session_view(next_selection.selected_node, cx);
        }

        let did_start_layout_animation = self.start_layout_animation(
            from_layout,
            self.snapshot_scene_layout(),
            previous_selection.selected_node,
            next_selection.selected_node,
        );
        if did_start_layout_animation {
            self.layout_animation_guard =
                ui_idle_tracker(cx).map(|tracker| tracker.begin_transition());
        } else {
            self.layout_animation_guard = None;
        }

        if previous_selection.selected_node != next_selection.selected_node {
            if !self.did_initial_fit {
                self.fit_suppressed = true;
            }
            self.pending_pan_to_selection = next_selection.selected_node;
        }

        self.update_selection_bar_target(cx);
        cx.notify();
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
            self.sync_task_session_view(next_selection.selected_node, cx);
            let did_start_layout_animation = self.start_layout_animation(
                from_layout,
                self.snapshot_scene_layout(),
                previous_selection.selected_node,
                next_selection.selected_node,
            );
            if did_start_layout_animation {
                self.layout_animation_guard =
                    ui_idle_tracker(cx).map(|tracker| tracker.begin_transition());
            } else {
                self.layout_animation_guard = None;
            }

            if !self.did_initial_fit {
                self.fit_suppressed = true;
            }

            self.pending_pan_to_selection = next_selection.selected_node;
            self.pending_focus_task_card = None;
            self.pending_restore_task_focus_zoom = None;

            let was_task_selected =
                matches!(previous_selection.selected_node, Some(GraphNodeId::Task(_)));
            let is_task_selected =
                matches!(next_selection.selected_node, Some(GraphNodeId::Task(_)));

            if is_task_selected {
                if !was_task_selected {
                    self.task_focus_zoom_before
                        .get_or_insert(self.camera.zoom());
                }
                self.pending_focus_task_card = next_selection.selected_node;
                self.pending_pan_to_selection = None;
            } else if was_task_selected {
                if let Some(zoom_before) = self.task_focus_zoom_before.take() {
                    self.pending_restore_task_focus_zoom = Some(zoom_before);
                }
            }
        }

        self.update_selection_bar_target(cx);
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

    fn sync_task_session_view(
        &mut self,
        selected_node: Option<GraphNodeId>,
        cx: &mut Context<Self>,
    ) {
        let task_id = match selected_node {
            Some(GraphNodeId::Task(task_id)) => Some(task_id),
            _ => None,
        };

        self.task_session_view.update(cx, |view, cx| {
            view.bind_latest_task_session(task_id, cx);
        });
    }

    fn reset_expanded_card_state(&mut self) {
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
    ) -> bool {
        let duration = ui_test_mode_animation_duration(Duration::from_millis(180));
        if duration == Duration::from_millis(0) {
            self.layout_animation = None;
            return false;
        }

        self.layout_animation = Some(LayoutAnimation {
            start: Instant::now(),
            duration,
            from_layout,
            to_layout,
            from_selected_node,
            to_selected_node,
        });
        true
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
            self.layout_animation_guard = None;
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
        self.camera_animation_guard = None;

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
            self.camera_animation_guard = None;
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

        let duration = ui_test_mode_animation_duration(duration);
        if duration == Duration::from_millis(0) {
            self.camera.set_origin_world(to_origin_world);
            self.camera.set_zoom(to_zoom);
            return true;
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

    fn try_focus_task_card_for_render(&mut self) -> bool {
        const FILL_RATIO: f32 = 0.8;

        let Some(node_id) = self.pending_focus_task_card else {
            return false;
        };

        if self.pan_drag.is_some() {
            return false;
        }

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return false;
        };

        if self.scene.node(node_id).is_none() {
            self.pending_focus_task_card = None;
            return false;
        }

        let node_rect = NodeWorldRect::from_scene(&self.scene, node_id);
        if node_rect.width <= 0.0 || node_rect.height <= 0.0 {
            self.pending_focus_task_card = None;
            return false;
        }

        let viewport_width = f32::from(canvas_bounds.size.width);
        let viewport_height = f32::from(canvas_bounds.size.height);

        let desired_width = viewport_width * FILL_RATIO;
        let desired_height = viewport_height * FILL_RATIO;
        if desired_width <= 0.0 || desired_height <= 0.0 {
            return false;
        }

        let mut camera = self.camera;
        camera.set_zoom((desired_width / node_rect.width).min(desired_height / node_rect.height));
        let target_zoom = camera.zoom();

        let node_center_world = gpui::point(
            node_rect.origin.x + node_rect.width / 2.0,
            node_rect.origin.y + node_rect.height / 2.0,
        );
        let viewport_center_world = gpui::point(
            (viewport_width / 2.0) / target_zoom,
            (viewport_height / 2.0) / target_zoom,
        );
        let target_origin = gpui::point(
            node_center_world.x - viewport_center_world.x,
            node_center_world.y - viewport_center_world.y,
        );

        self.pending_focus_task_card = None;
        self.pending_pan_to_selection = None;

        let did_start =
            self.try_start_camera_animation(target_origin, target_zoom, Duration::from_millis(240));
        if did_start {
            redesmyn_logging::tracing::info!(
                node_id = %node_id,
                origin_world = ?target_origin,
                zoom = target_zoom,
                "graph focus task card"
            );
        }
        did_start
    }

    fn try_restore_task_focus_zoom_for_render(&mut self) -> bool {
        let Some(target_zoom) = self.pending_restore_task_focus_zoom else {
            return false;
        };

        if self.pan_drag.is_some() {
            return false;
        }

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return false;
        };

        let mut camera = self.camera;
        camera.set_zoom(target_zoom);
        let target_zoom = camera.zoom();

        let viewport_center_screen = gpui::point(
            canvas_bounds.size.width / 2.0,
            canvas_bounds.size.height / 2.0,
        );
        let viewport_center_world = self.camera.screen_to_world(viewport_center_screen);
        let viewport_center_world_for_target_zoom = gpui::point(
            f32::from(viewport_center_screen.x) / target_zoom,
            f32::from(viewport_center_screen.y) / target_zoom,
        );
        let target_origin = gpui::point(
            viewport_center_world.x - viewport_center_world_for_target_zoom.x,
            viewport_center_world.y - viewport_center_world_for_target_zoom.y,
        );

        self.pending_restore_task_focus_zoom = None;

        let did_start =
            self.try_start_camera_animation(target_origin, target_zoom, Duration::from_millis(200));
        if did_start {
            redesmyn_logging::tracing::info!(
                origin_world = ?target_origin,
                zoom = target_zoom,
                "graph restore task focus zoom"
            );
        }
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

    fn task_quick_actions_row(
        &self,
        node_key: gpui::SharedString,
        task_id: TaskId,
        agent_status: AgentStatus,
        has_session: bool,
        opacity: f32,
        interactive: bool,
        theme: &redesmyn_ui::styles::UiTheme,
        zoom: f32,
        rem_size: gpui::Pixels,
        cx: &mut Context<Self>,
    ) -> gpui::Div {
        let graph = cx.entity();
        let entity_id = cx.entity_id();

        let action_state = self.quick_actions.get(&task_id);
        let start_disabled = action_state.is_some_and(|state| state.start.in_flight);
        let restart_disabled = action_state.is_some_and(|state| state.restart.in_flight);
        let stop_disabled = action_state.is_some_and(|state| state.stop.in_flight);

        let show_stop = matches!(agent_status, AgentStatus::Running | AgentStatus::Blocked);
        let show_restart = has_session || show_stop;
        let show_start = !show_restart;

        let button = |kind: TaskQuickActionKind,
                      label: &'static str,
                      tooltip: &'static str,
                      fg: gpui::Hsla,
                      hover_bg: gpui::Hsla,
                      disabled: bool| {
            let base_id = match kind {
                TaskQuickActionKind::Start => "task_quick_action_start",
                TaskQuickActionKind::Restart => "task_quick_action_restart",
                TaskQuickActionKind::Stop => "task_quick_action_stop",
            };
            let id = (
                gpui::ElementId::from((base_id, entity_id)),
                node_key.clone(),
            );

            let mut button = div()
                .id(id)
                .flex()
                .items_center()
                .justify_center()
                .size(px(22.0 * zoom))
                .rounded(px(f32::from(theme.radius.md) * zoom))
                .text_size(quantized_zoom_text_size(rem_size, 0.70, zoom))
                .text_color(fg)
                .child(label);

            if interactive {
                button = button
                    .cursor_pointer()
                    .focusable()
                    .hover(move |this| this.bg(hover_bg))
                    .tooltip(move |_, cx| cx.new(|_| Tooltip::new(tooltip)).into());

                if disabled {
                    button = button.opacity(0.55).cursor_not_allowed();
                } else {
                    let graph = graph.clone();
                    button = button.on_click(move |event, _window, cx| {
                        if event.standard_click() {
                            graph.update(cx, |this, cx| {
                                this.trigger_task_quick_action(task_id, kind, cx);
                            });
                        }
                        cx.stop_propagation();
                    });
                }
            } else {
                button = button.cursor(CursorStyle::Arrow);
            }

            button
        };

        let mut row = div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(2.0 * zoom))
            .opacity(opacity);

        if show_start {
            row = row.child(button(
                TaskQuickActionKind::Start,
                "▶",
                "Start agent",
                theme.colors.foreground_muted,
                theme.colors.accent.opacity(0.65),
                start_disabled,
            ));
        }

        if show_restart {
            row = row.child(button(
                TaskQuickActionKind::Restart,
                "↻",
                "Restart agent",
                theme.colors.foreground_muted,
                theme.colors.accent.opacity(0.65),
                restart_disabled,
            ));
        }

        if show_stop {
            row = row.child(button(
                TaskQuickActionKind::Stop,
                "■",
                "Stop agent",
                theme.colors.danger,
                theme.colors.danger.opacity(0.10),
                stop_disabled,
            ));
        }

        row
    }

    fn quick_action_state_mut(&mut self, task_id: TaskId) -> &mut TaskQuickActionState {
        self.quick_actions.entry(task_id).or_default()
    }

    fn trigger_task_quick_action(
        &mut self,
        task_id: TaskId,
        kind: TaskQuickActionKind,
        cx: &mut Context<Self>,
    ) {
        let client = self.task_session_view.read(cx).client();

        let state = self.quick_action_state_mut(task_id);
        let (action, task_slot) = match kind {
            TaskQuickActionKind::Start => (&mut state.start, &mut state.start_task),
            TaskQuickActionKind::Restart => (&mut state.restart, &mut state.restart_task),
            TaskQuickActionKind::Stop => (&mut state.stop, &mut state.stop_task),
        };

        if action.in_flight {
            return;
        }

        let Some(client) = client else {
            action.fail("Control plane client is unavailable.");
            cx.notify();
            return;
        };

        action.start();
        cx.notify();

        let view = cx.entity();
        *task_slot = Some(
            cx.spawn(move |_: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                let client = client.clone();
                let cx = cx.clone();
                async move {
                    let result = task_quick_action_request(&client, task_id, kind).await;
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            this.on_task_quick_action_completed(task_id, kind, result, cx);
                        })
                    });
                }
            }),
        );
    }

    fn on_task_quick_action_completed(
        &mut self,
        task_id: TaskId,
        kind: TaskQuickActionKind,
        result: Result<ResponseResult, redesmyn_protocol::ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        let Some(state) = self.quick_actions.get_mut(&task_id) else {
            return;
        };

        let (action, task_slot) = match kind {
            TaskQuickActionKind::Start => (&mut state.start, &mut state.start_task),
            TaskQuickActionKind::Restart => (&mut state.restart, &mut state.restart_task),
            TaskQuickActionKind::Stop => (&mut state.stop, &mut state.stop_task),
        };
        *task_slot = None;

        let message = match result {
            Ok(ResponseResult::StartAgent(_)) if kind == TaskQuickActionKind::Start => None,
            Ok(ResponseResult::RestartAgent(_)) if kind == TaskQuickActionKind::Restart => None,
            Ok(ResponseResult::StopAgent(_)) if kind == TaskQuickActionKind::Stop => None,
            Ok(ResponseResult::Error(err)) => Some(err.message),
            Err(err) => Some(err.message),
            Ok(other) => Some(format!("Unexpected response: {other:?}")),
        };

        if let Some(message) = message {
            action.fail(message);
        } else {
            action.succeed();
            action.clear_error();
        }

        cx.notify();
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
            redesmyn_logging::tracing::debug!(
                position = ?event.position,
                raw_delta = ?event.delta,
                touch_phase = ?event.touch_phase,
                modifiers = ?event.modifiers,
                "graph scroll wheel pixel delta is (0, 0)"
            );
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

        if self
            .task_session_view
            .read(cx)
            .should_defer_pan_to_scroll_view(event.position, delta)
            || should_defer_pan_to_scroll_view(&self.expanded_details_scroll, event.position, delta)
        {
            return;
        }

        self.camera.pan_by_screen_delta(delta);
        cx.notify();
    }

    fn selection_bar_target_visible(&self) -> bool {
        self.scene.selection().selected_edge.is_none()
            && self.scene.selection().selected_nodes.len() > 1
    }

    fn update_selection_bar_target(&mut self, cx: &App) {
        let target_visible = self.selection_bar_target_visible();
        let target = if target_visible { 1.0 } else { 0.0 };
        if (self.selection_bar_progress - target).abs() < f32::EPSILON {
            return;
        }

        let duration = ui_test_mode_animation_duration(Duration::from_millis(180));
        if duration == Duration::from_millis(0) {
            self.selection_bar_progress = target;
            self.selection_bar_transition = None;
            self.selection_bar_guard = None;
            return;
        }

        self.selection_bar_transition = Some(SelectionBarTransition {
            started_at: Instant::now(),
            from: self.selection_bar_progress,
            to: target,
            duration,
        });
        self.selection_bar_guard = ui_idle_tracker(cx).map(|tracker| tracker.begin_transition());
    }

    fn tick_selection_bar_animation(&mut self, window: &mut Window) {
        let Some(transition) = self.selection_bar_transition else {
            return;
        };

        if transition.duration == Duration::from_millis(0) {
            self.selection_bar_progress = transition.to;
            self.selection_bar_transition = None;
            self.selection_bar_guard = None;
            return;
        }

        let elapsed = transition.started_at.elapsed();
        let t = (elapsed.as_secs_f32() / transition.duration.as_secs_f32()).clamp(0.0, 1.0);
        let eased = 1.0 - (1.0 - t).powi(3);
        self.selection_bar_progress = transition.from + (transition.to - transition.from) * eased;

        if t >= 1.0 {
            self.selection_bar_progress = transition.to;
            self.selection_bar_transition = None;
            self.selection_bar_guard = None;
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
            redesmyn_logging::tracing::debug!(reason, "ignoring bulk start click (disabled)");
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

    fn trunk_bounds_in_window_for_progress(
        &self,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        t: f32,
    ) -> Option<gpui::Bounds<gpui::Pixels>> {
        if self.scene.node(GraphNodeId::Trunk).is_none() {
            return None;
        }

        Some(self.node_bounds_in_window_for_progress(GraphNodeId::Trunk, canvas_bounds, t))
    }

    fn trunk_base_anchor_in_window_for_progress(
        &self,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        t: f32,
    ) -> Option<gpui::Point<gpui::Pixels>> {
        let trunk_layout = self.scene.trunk_layout()?;
        let trunk_bounds = self.trunk_bounds_in_window_for_progress(canvas_bounds, t)?;

        let zoom = self.camera.zoom();
        let line_x = (TRUNK_TITLE_WIDTH + TRUNK_MARKER_WIDTH / 2) as f32;

        Some(gpui::point(
            trunk_bounds.left() + px(line_x * zoom),
            trunk_bounds.top() + px(trunk_layout.base_offset as f32 * zoom),
        ))
    }

    fn paint_trunk_line_for_progress(
        &self,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        t: f32,
        theme: &redesmyn_ui::styles::UiTheme,
        window: &mut Window,
    ) {
        let Some(trunk_bounds) = self.trunk_bounds_in_window_for_progress(canvas_bounds, t) else {
            return;
        };

        let zoom = self.camera.zoom();
        let thickness = px(TRUNK_THICKNESS as f32).max(px(1.0));
        let line_x = (TRUNK_TITLE_WIDTH + TRUNK_MARKER_WIDTH / 2) as f32;
        let line_x_px = trunk_bounds.left() + px(line_x * zoom);

        let line_bounds = gpui::Bounds {
            origin: gpui::point(line_x_px - thickness / 2.0, trunk_bounds.top()),
            size: gpui::size(thickness, trunk_bounds.size.height),
        };

        window.paint_quad(fill(line_bounds, theme.colors.border.opacity(0.6)));
    }

    fn paint_trunk_text_line_for_progress(
        &self,
        origin: gpui::Point<gpui::Pixels>,
        line_height: gpui::Pixels,
        text: gpui::SharedString,
        font: gpui::Font,
        font_size: gpui::Pixels,
        color: gpui::Hsla,
        window: &mut Window,
    ) {
        if text.is_empty() {
            return;
        }

        let run = TextRun {
            len: text.len(),
            font,
            color,
            background_color: None,
            underline: None,
            strikethrough: None,
        };

        // Note: This shapes per paint for each visible label. If this becomes hot with many trunk
        // marks on screen, cache shaped lines by (font_size, text) similarly to edge labels.
        let shaped =
            window
                .text_system()
                .shape_line(text, font_size, std::slice::from_ref(&run), None);
        self.paint_shaped_line(origin, line_height, &shaped, color, window);
    }

    fn paint_trunk_marks_for_progress(
        &self,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        t: f32,
        theme: &redesmyn_ui::styles::UiTheme,
        window: &mut Window,
    ) {
        let Some(trunk_layout) = self.scene.trunk_layout() else {
            return;
        };
        let Some(trunk_bounds) = self.trunk_bounds_in_window_for_progress(canvas_bounds, t) else {
            return;
        };

        let zoom = self.camera.zoom();
        let show_labels = zoom >= TRUNK_LABEL_LOD_ZOOM;

        let marker_size = px((10.0 * zoom).clamp(4.0, 14.0));
        let marker_radius = marker_size / 2.0;

        let line_x = (TRUNK_TITLE_WIDTH + TRUNK_MARKER_WIDTH / 2) as f32;
        let line_x_px = trunk_bounds.left() + px(line_x * zoom);

        let title_x = trunk_bounds.left();
        let sha_x =
            trunk_bounds.left() + px((TRUNK_TITLE_WIDTH + TRUNK_MARKER_WIDTH + 10) as f32 * zoom);
        let ellipsis_x = sha_x;

        let font_size = px(11.0 * zoom);
        let line_height = px(14.0 * zoom);

        let mark_count = trunk_layout.marks.len();
        let commit_sampling_step = if mark_count > 250 && zoom < TRUNK_LABEL_LOD_ZOOM {
            (mark_count / 150).max(1)
        } else {
            1
        };

        for (index, mark) in trunk_layout.marks.iter().enumerate() {
            let row_top = trunk_bounds.top()
                + px(
                    (trunk_layout.commit_padding + index as i32 * trunk_layout.commit_spacing)
                        as f32
                        * zoom,
                );
            let row_height = px(trunk_layout.row_height as f32 * zoom);
            let row_bottom = row_top + row_height;

            if row_bottom < canvas_bounds.top() || row_top > canvas_bounds.bottom() {
                continue;
            }

            match mark.kind {
                TrunkMarkKind::Connector => continue,
                TrunkMarkKind::Ellipsis => {
                    if show_labels {
                        self.paint_trunk_text_line_for_progress(
                            gpui::point(ellipsis_x, row_top),
                            line_height,
                            gpui::SharedString::new_static("···"),
                            theme.typography.caption.font.clone(),
                            font_size,
                            theme.colors.foreground_muted.opacity(0.65),
                            window,
                        );
                    }
                    continue;
                }
                TrunkMarkKind::Commit => {
                    if commit_sampling_step > 1
                        && index % commit_sampling_step != 0
                        && mark.title.is_some()
                    {
                        continue;
                    }
                }
                TrunkMarkKind::Base => {}
            }

            let row_center_y = row_top + row_height / 2.0;
            let marker_bounds = gpui::Bounds {
                origin: gpui::point(
                    line_x_px - marker_size / 2.0,
                    row_center_y - marker_size / 2.0,
                ),
                size: gpui::size(marker_size, marker_size),
            };

            let is_base = mark.kind == TrunkMarkKind::Base;
            let border_color = if is_base {
                theme.colors.foreground.opacity(0.75)
            } else {
                theme.colors.foreground.opacity(0.6)
            };

            window.paint_quad(quad(
                marker_bounds,
                marker_radius,
                theme.colors.background,
                1.0,
                border_color,
                gpui::BorderStyle::Solid,
            ));

            if !show_labels {
                continue;
            }

            let title = mark
                .title
                .as_ref()
                .map(|title| truncate_shared_string(title, 34))
                .unwrap_or_else(|| gpui::SharedString::new_static("—"));

            if !title.is_empty() {
                self.paint_trunk_text_line_for_progress(
                    gpui::point(title_x, row_top),
                    line_height,
                    title,
                    theme.typography.caption.font.clone(),
                    font_size,
                    if is_base {
                        theme.colors.foreground.opacity(0.78)
                    } else {
                        theme.colors.foreground_muted.opacity(0.78)
                    },
                    window,
                );
            }

            let sha = mark
                .sha
                .as_ref()
                .map(|sha| truncate_shared_string(sha, 7))
                .unwrap_or_default();

            if !sha.is_empty() {
                self.paint_trunk_text_line_for_progress(
                    gpui::point(sha_x, row_top),
                    line_height,
                    sha,
                    theme.typography.mono.font.clone(),
                    font_size,
                    if is_base {
                        theme.colors.foreground.opacity(0.8)
                    } else {
                        theme.colors.foreground_muted.opacity(0.85)
                    },
                    window,
                );
            }
        }
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

        self.paint_trunk_line_for_progress(canvas_bounds, t, &theme, window);

        for edge in self.scene.edges() {
            if edge.id.from == GraphNodeId::Trunk {
                let Some(from_anchor) =
                    self.trunk_base_anchor_in_window_for_progress(canvas_bounds, t)
                else {
                    continue;
                };
                let Some(to) = self.scene.node(edge.id.to) else {
                    continue;
                };

                let to_bounds = self.node_bounds_in_window_for_progress(to.id, canvas_bounds, t);
                let to_anchor = gpui::point(
                    to_bounds.left(),
                    to_bounds.top() + to_bounds.size.height / 2.0,
                );

                let route = edge_route_between_points_in_window(from_anchor, to_anchor);
                let color = theme.colors.border.opacity(0.45);
                for segment in route
                    .segment_bounds(DEFAULT_EDGE_STROKE_PX)
                    .iter()
                    .flatten()
                {
                    window.paint_quad(fill(*segment, color));
                }
                continue;
            }

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

        self.paint_trunk_marks_for_progress(canvas_bounds, t, &theme, window);
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
        self.fps_overlay.on_frame();
        self.step_camera_animation_for_render(window);
        let started_restore = self.try_restore_task_focus_zoom_for_render();
        let started_focus = if started_restore {
            false
        } else {
            self.try_focus_task_card_for_render()
        };
        let started_fit = if started_restore || started_focus {
            false
        } else {
            self.try_initial_fit_for_render()
        };
        let started_pan = if started_restore || started_focus {
            false
        } else {
            self.try_pan_to_selection_for_render()
        };
        if (started_restore || started_focus || started_fit || started_pan)
            && self.camera_animation.is_some()
        {
            if self.camera_animation_guard.is_none() {
                self.camera_animation_guard =
                    ui_idle_tracker(cx).map(|tracker| tracker.begin_transition());
            }
            window.request_animation_frame();
        }

        let theme = theme_for_window(window, cx);
        let rem_size = window.rem_size();
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
            let selection_snapshot = self.scene.selection().clone();
            let selected_nodes = selection_snapshot.selected_nodes;
            let hovered_node = selection_snapshot.hovered_node;
            let quick_actions_fade_duration = ui_test_mode_animation_duration(theme.animation.fast);
            let task_session_state = self.task_session_view.read(cx).task_binding_state();
            // Overscan avoids popping nodes in/out right at the viewport edge while panning.
            let viewport_bounds =
                viewport_bounds_with_overscan(canvas_bounds, px(DEFAULT_NODE_CULLING_OVERSCAN_PX));

            let mut layer = div().absolute().inset_0();

            for node in self.scene.nodes() {
                if node.id == GraphNodeId::Trunk {
                    continue;
                }
                let bounds_in_window =
                    self.node_bounds_in_window_for_progress(node.id, canvas_bounds, t);
                if !bounds_in_window.intersects(&viewport_bounds) {
                    continue;
                }
                let local_origin = bounds_in_window.origin - canvas_bounds.origin;
                let node_id = node.id;
                let node_key: gpui::SharedString = node_id.to_string().into();
                let is_primary_selected = primary_selected == Some(node_id);
                let is_multi_selected = selected_nodes.contains(&node_id);

                let is_animating_expand = self
                    .layout_animation
                    .as_ref()
                    .is_some_and(|animation| animation.to_selected_node == Some(node_id));
                let is_animating_collapse =
                    self.layout_animation.as_ref().is_some_and(|animation| {
                        animation.from_selected_node == Some(node_id)
                            && animation.to_selected_node != Some(node_id)
                    });

                let expandedness = if is_animating_expand {
                    t
                } else if is_animating_collapse {
                    1.0 - t
                } else if is_primary_selected {
                    1.0
                } else {
                    0.0
                };

                let expandedness = expandedness.clamp(0.0, 1.0);
                let selection_t = if is_animating_expand || is_animating_collapse {
                    expandedness
                } else if is_primary_selected || is_multi_selected {
                    1.0
                } else {
                    0.0
                };

                let expanded_opacity =
                    expandedness.powf(if is_animating_collapse { 4.0 } else { 1.5 });
                let collapsed_opacity = (1.0 - expandedness).powf(2.0);
                let bg = lerp_hsla(
                    theme.colors.surface_elevated,
                    theme.colors.ring,
                    0.10 * selection_t,
                );
                let base_border =
                    collapsed_task_border_color(node.state, node.merge_readiness, &theme)
                        .opacity(0.8);
                let border = lerp_hsla(base_border, theme.colors.ring, selection_t);

                let node_element_id = (
                    gpui::ElementId::from(("graph_node", entity_id)),
                    node_key.clone(),
                );

                let mut card = div()
                    .id(node_element_id.clone())
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

                if collapsed_opacity > 0.01 {
                    let title = node.title.clone();
                    let branch_slug = node.branch_slug.clone();
                    let state = node.state;
                    let latest_session = node.latest_session.clone();
                    let task_id = match node_id {
                        GraphNodeId::Task(task_id) => task_id,
                        GraphNodeId::Trunk => unreachable!("trunk nodes are skipped above"),
                    };
                    let padding_x = px(f32::from(theme.spacing.md) * zoom);
                    let padding_top = px(f32::from(theme.spacing.sm) * zoom);
                    let padding_bottom = px(f32::from(theme.spacing.md) * zoom);
                    let is_hovered = hovered_node == Some(node_id);
                    let show_quick_actions = is_hovered || is_primary_selected;
                    let quick_actions_opacity = self.quick_action_opacity.opacity_for_render(
                        task_id,
                        show_quick_actions,
                        quick_actions_fade_duration,
                        window,
                    );
                    let should_render_quick_actions = quick_actions_opacity > 0.01;

                    // Compute a stable (pixel-snapped) content width for text measurement and
                    // rendering so we don't flicker between neighboring wrap/truncation states
                    // while zooming.
                    const COLLAPSED_TEXT_WIDTH_SLACK_PX: f32 = 10.0;
                    let collapsed_text_width = px((f32::from(bounds_in_window.size.width)
                        - 2.0 * f32::from(padding_x)
                        - COLLAPSED_TEXT_WIDTH_SLACK_PX)
                        .max(0.0)
                        .floor());
                    let title_text_size = quantized_zoom_text_size(rem_size, 0.78, zoom);
                    let (title_line1, title_line2) = collapsed_title_lines(
                        &mut self.collapsed_title_cache,
                        task_id,
                        title.clone(),
                        collapsed_text_width,
                        theme.typography.body.font.clone(),
                        title_text_size,
                        theme.colors.foreground,
                        window,
                    );

                    let branch_slug_text_size = quantized_zoom_text_size(rem_size, 0.60, zoom);
                    let preview_text_size = quantized_zoom_text_size(rem_size, 0.66, zoom);
                    let quick_actions_slot_width = px(46.0 * zoom);
                    let quick_actions_slot_height = px(22.0 * zoom);

                    let preview_content = latest_session.as_ref().and_then(|session| {
                        let preview = session.message_preview.as_ref()?;
                        if let Some(content) =
                            self.collapsed_markdown_cache.get(&session.session_event_id)
                        {
                            return Some(content.clone());
                        }

                        let doc = parse_markdown(preview.as_str(), MarkdownParseOptions::default());
                        let content = MarkdownInlineSingleLineContent::from_doc(&doc);
                        self.collapsed_markdown_cache
                            .insert(session.session_event_id, content.clone());
                        Some(content)
                    });

                    let mut collapsed_content = div()
                        .absolute()
                        .inset_0()
                        .px(padding_x)
                        .pt(padding_top)
                        .pb(padding_bottom)
                        .size_full()
                        .flex()
                        .flex_col()
                        .gap(px(6.0 * zoom))
                        .opacity(collapsed_opacity)
                        .child(
                            div()
                                .flex()
                                .flex_row()
                                .items_center()
                                .justify_between()
                                .gap(px(8.0 * zoom))
                                .h(px(24.0 * zoom))
                                .child(
                                    div()
                                        .min_w_0()
                                        .flex_1()
                                        .text_size(branch_slug_text_size)
                                        .text_color(theme.colors.foreground_muted)
                                        .truncate()
                                        .child(branch_slug),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .flex_row()
                                        .items_center()
                                        .gap(px(4.0 * zoom))
                                        .child(
                                            div()
                                                .relative()
                                                .w(quick_actions_slot_width)
                                                .h(quick_actions_slot_height)
                                                .child(
                                                    div()
                                                        .absolute()
                                                        .inset_0()
                                                        .opacity(quick_actions_opacity)
                                                        .flex()
                                                        .flex_row()
                                                        .items_center()
                                                        .justify_end()
                                                        .when(
                                                            should_render_quick_actions,
                                                            |this| {
                                                                this.child(
                                                                    self.task_quick_actions_row(
                                                                        node_key.clone(),
                                                                        task_id,
                                                                        node.agent_status,
                                                                        latest_session
                                                                            .as_ref()
                                                                            .is_some(),
                                                                        quick_actions_opacity,
                                                                        show_quick_actions,
                                                                        &theme,
                                                                        zoom,
                                                                        rem_size,
                                                                        cx,
                                                                    ),
                                                                )
                                                            },
                                                        ),
                                                ),
                                        )
                                        .child(agent_status_dot(
                                            state,
                                            node.agent_status,
                                            latest_session.as_ref().is_some(),
                                            &theme,
                                            zoom,
                                        )),
                                ),
                        )
                        .child(
                            div()
                                .min_w_0()
                                .w_full()
                                .flex()
                                .flex_col()
                                .gap(px(0.0))
                                .child(
                                    div()
                                        .min_w_0()
                                        .w(collapsed_text_width)
                                        .font(theme.typography.body.font.clone())
                                        .text_size(title_text_size)
                                        .text_color(theme.colors.foreground)
                                        .truncate()
                                        .child(title_line1),
                                )
                                .when_some(title_line2, |this, line2| {
                                    this.child(
                                        div()
                                            .min_w_0()
                                            .w(collapsed_text_width)
                                            .font(theme.typography.body.font.clone())
                                            .text_size(title_text_size)
                                            .text_color(theme.colors.foreground)
                                            .truncate()
                                            .child(line2),
                                    )
                                }),
                        );

                    if let Some(content) = preview_content {
                        collapsed_content = collapsed_content.child(
                            div()
                                .min_w_0()
                                .w(collapsed_text_width)
                                .truncate()
                                .text_size(preview_text_size)
                                .text_color(theme.colors.foreground_muted)
                                .child(content.plain_text()),
                        );
                    }
                    card = card.child(collapsed_content);
                }

                if expanded_opacity > 0.01 {
                    let title = node.title.clone();
                    let task_slug = node.task_slug.clone();
                    let show_task_slug = !title.as_ref().contains(task_slug.as_ref());
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
                    let start_agent_action = {
                        let task_session_view = self.task_session_view.clone();
                        move |_: &ClickEvent, _window: &mut Window, cx: &mut App| {
                            let GraphNodeId::Task(task_id) = node_id else {
                                return;
                            };
                            task_session_view
                                .update(cx, |view, cx| view.start_agent_for_task(task_id, cx));
                        }
                    };

                    let right_scroll = self.expanded_details_scroll.clone();
                    let demo_action = self.demo_action.clone();

                    card = card.child(
                        div()
                            .flex()
                            .flex_col()
                            .size_full()
                            .absolute()
                            .inset_0()
                            .opacity(expanded_opacity)
                            .child(
                                div()
                                    .px(theme.spacing.md)
                                    .py(theme.spacing.sm)
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
                                            .flex_col()
                                            .min_w_0()
                                            .gap(px(2.0))
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .text_color(theme.colors.foreground)
                                                    .truncate()
                                                    .child(title),
                                            )
                                            .when(show_task_slug, |this| {
                                                this.child(
                                                    div()
                                                        .text_xs()
                                                        .text_color(theme.colors.foreground_muted)
                                                        .truncate()
                                                        .child(task_slug),
                                                )
                                            }),
                                    )
                                    .child(
                                        div()
                                            .flex()
                                            .flex_row()
                                            .items_center()
                                            .gap(theme.spacing.sm)
                                            .child(task_status_chips(
                                                state,
                                                merge_readiness,
                                                agent_status,
                                                task_session_state.session_id.is_some(),
                                                &theme,
                                            ))
                                            .child(
                                                IconButton::new(close_button_id, div().child("×"))
                                                    .tooltip("Collapse")
                                                    .on_click(collapse),
                                            ),
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
                                                    .flex_row()
                                                    .items_center()
                                                    .text_sm()
                                                    .text_color(theme.colors.foreground)
                                                    .justify_between()
                                                    .child("Session")
                                                    .when(is_primary_selected, |this| {
                                                        let label = if task_session_state.in_flight
                                                        {
                                                            "Loading…".to_string()
                                                        } else if let Some(session_id) =
                                                            task_session_state.session_id
                                                        {
                                                            session_id.to_string()
                                                        } else {
                                                            "No session".to_string()
                                                        };

                                                        this.child(
                                                            div()
                                                                .min_w_0()
                                                                .text_xs()
                                                                .text_color(
                                                                    theme.colors.foreground_muted,
                                                                )
                                                                .truncate()
                                                                .child(label),
                                                        )
                                                    }),
                                            )
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_h(px(0.0))
                                                    .p(theme.spacing.md)
                                                    .flex()
                                                    .flex_col()
                                                    .gap(theme.spacing.sm)
                                                    .when(is_primary_selected, |this| {
                                                        let mut this = this;

                                                        if task_session_state.in_flight {
                                                            let label = if task_session_state.operation
                                                                == Some(TaskSessionOperation::StartAgent)
                                                            {
                                                                "Starting agent…"
                                                            } else {
                                                                "Loading latest session…"
                                                            };
                                                            this = this.child(
                                                                div()
                                                                    .text_sm()
                                                                    .text_color(
                                                                        theme.colors.foreground_muted,
                                                                    )
                                                                    .child(label),
                                                            );
                                                        } else if let Some(err) = task_session_state.error.clone() {
                                                            let refresh_button_id = (
                                                                gpui::ElementId::from((
                                                                    "task_card_session_refresh",
                                                                    entity_id,
                                                                )),
                                                                node_key.clone(),
                                                            );
                                                            let task_session_view =
                                                                self.task_session_view.clone();
                                                            let is_start_agent_error = task_session_state.operation
                                                                == Some(TaskSessionOperation::StartAgent);
                                                            let action_label =
                                                                if is_start_agent_error { "Start agent" } else { "Retry" };

                                                            this = this.child(
                                                                Callout::new(err)
                                                                    .kind(CalloutKind::Danger)
                                                                    .title("Session")
                                                                    .action(
                                                                        TextButton::new(
                                                                            refresh_button_id,
                                                                            action_label,
                                                                        )
                                                                        .kind(ButtonKind::Secondary)
                                                                        .on_click(move |_, _, cx| {
                                                                            if is_start_agent_error {
                                                                                let GraphNodeId::Task(task_id) = node_id else {
                                                                                    return;
                                                                                };
                                                                                task_session_view.update(
                                                                                    cx,
                                                                                    |view, cx| {
                                                                                        view.start_agent_for_task(task_id, cx)
                                                                                    },
                                                                                );
                                                                            } else {
                                                                                task_session_view.update(
                                                                                    cx,
                                                                                    |view, cx| {
                                                                                        view.refresh_latest_task_session(cx)
                                                                                    },
                                                                                );
                                                                            }
                                                                        }),
                                                                    ),
                                                            );
                                                        } else if task_session_state.session_id.is_none() {
                                                            let refresh_button_id = (
                                                                gpui::ElementId::from((
                                                                    "task_card_session_refresh",
                                                                    entity_id,
                                                                )),
                                                                node_key.clone(),
                                                            );
                                                            let start_button_id = (
                                                                gpui::ElementId::from((
                                                                    "task_card_session_start_agent",
                                                                    entity_id,
                                                                )),
                                                                node_key.clone(),
                                                            );

                                                            let task_session_view =
                                                                self.task_session_view.clone();

                                                            this = this.child(
                                                                Callout::new("No session yet.")
                                                                    .kind(CalloutKind::Info)
                                                                    .title("Session")
                                                                    .action(
                                                                        div()
                                                                            .flex()
                                                                            .flex_row()
                                                                            .gap(theme.spacing.sm)
                                                                            .child(
                                                                                TextButton::new(
                                                                                    start_button_id,
                                                                                    "Start agent",
                                                                                )
                                                                                .kind(
                                                                                    ButtonKind::Secondary,
                                                                                )
                                                                                .on_click(
                                                                                    start_agent_action,
                                                                                ),
                                                                            )
                                                                            .child(
                                                                                TextButton::new(
                                                                                    refresh_button_id,
                                                                                    "Refresh",
                                                                                )
                                                                                .kind(ButtonKind::Ghost)
                                                                                .on_click(move |_, _, cx| {
                                                                                    task_session_view.update(
                                                                                        cx,
                                                                                        |view, cx| {
                                                                                            view.refresh_latest_task_session(cx)
                                                                                        },
                                                                                    );
                                                                                }),
                                                                            ),
                                                                    ),
                                                            );
                                                        }

                                                        this.child(
                                                            div()
                                                                .flex_1()
                                                                .min_h(px(0.0))
                                                                .child(
                                                                    self.task_session_view.clone(),
                                                                ),
                                                        )
                                                    }),
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
															.flex_row()
															.items_center()
															.gap(theme.spacing.sm)
															.text_sm()
															.text_color(theme.colors.foreground)
															.child("Details")
															.child({
																let active_section =
																	right_scroll.top_item().min(4);
																div()
																	.flex_1()
																	.min_w_0()
																	.flex()
																	.flex_row()
																	.items_center()
																	.justify_end()
																	.gap(theme.spacing.xs)
																	.child(details_nav_link(
																		(
																			gpui::ElementId::from((
																				"task_details_nav_overview",
																				entity_id,
																			)),
																			node_key.clone(),
																		),
																		"Overview",
																		0,
																		active_section == 0,
																		right_scroll.clone(),
																		&theme,
																	))
																	.child(details_nav_separator(&theme))
																	.child(details_nav_link(
																		(
																			gpui::ElementId::from((
																				"task_details_nav_agent",
																				entity_id,
																			)),
																			node_key.clone(),
																		),
																		"Agent",
																		1,
																		active_section == 1,
																		right_scroll.clone(),
																		&theme,
																	))
																	.child(details_nav_separator(&theme))
																	.child(details_nav_link(
																		(
																			gpui::ElementId::from((
																				"task_details_nav_merge",
																				entity_id,
																			)),
																			node_key.clone(),
																		),
																		"Merge",
																		2,
																		active_section == 2,
																		right_scroll.clone(),
																		&theme,
																	))
																	.child(details_nav_separator(&theme))
																	.child(details_nav_link(
																		(
																			gpui::ElementId::from((
																				"task_details_nav_ids",
																				entity_id,
																			)),
																			node_key.clone(),
																		),
																		"IDs",
																		3,
																		active_section == 3,
																		right_scroll.clone(),
																		&theme,
																	))
																	.child(details_nav_separator(&theme))
																	.child(details_nav_link(
																		(
																			gpui::ElementId::from((
																				"task_details_nav_artifacts",
																				entity_id,
																			)),
																			node_key.clone(),
																		),
																		"Artifacts",
																		4,
																		active_section == 4,
																		right_scroll.clone(),
																		&theme,
																	))
															}),
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
															.child(task_details_overview_section(
																branch_name.clone(),
																&theme,
															))
														.child(task_details_agent_section(
															action_button_id,
															demo_action,
															start_action,
															&theme,
														))
														.child(task_details_merge_section(&theme))
														.child(task_details_identifiers_section(
															node_id, &theme,
														))
														.child(task_details_artifacts_section(
															entity_id,
															node_key.clone(),
															&theme,
														))
															.child(div().h(theme.spacing.md)),
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

        let fps_overlay = if self.fps_overlay.enabled() {
            Some(
                div()
                    .absolute()
                    .right(theme.spacing.sm)
                    .bottom(theme.spacing.sm)
                    .px(theme.spacing.sm)
                    .py(theme.spacing.xs)
                    .rounded(theme.radius.md)
                    .bg(theme.colors.background.opacity(0.72))
                    .border_1()
                    .border_color(theme.colors.border.opacity(0.5))
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child(self.fps_overlay.label.clone()),
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
                    })
                    .when_some(fps_overlay, |this, overlay| this.child(overlay)),
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

fn quantized_zoom_text_size(rem_size: gpui::Pixels, base_rems: f32, zoom: f32) -> gpui::Pixels {
    // Quantize continuous zoom-driven font sizes to keep GPUI's glyph caches bounded, while still
    // feeling smooth during zoom.
    const STEP_PX: f32 = 0.1;
    let base_px = rems(base_rems).to_pixels(rem_size);
    let scaled = f32::from(base_px) * zoom.max(0.0);
    let quantized = (scaled / STEP_PX).round() * STEP_PX;
    px(quantized.max(1.0))
}

fn lerp_rgba(a: gpui::Rgba, b: gpui::Rgba, t: f32) -> gpui::Rgba {
    gpui::Rgba {
        r: lerp_f32(a.r, b.r, t),
        g: lerp_f32(a.g, b.g, t),
        b: lerp_f32(a.b, b.b, t),
        a: lerp_f32(a.a, b.a, t),
    }
}

fn lerp_hsla(a: gpui::Hsla, b: gpui::Hsla, t: f32) -> gpui::Hsla {
    lerp_rgba(a.into(), b.into(), t.clamp(0.0, 1.0)).into()
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

fn truncate_shared_string(value: &gpui::SharedString, max_chars: usize) -> gpui::SharedString {
    let value_str = value.as_ref();
    for (index, (offset, _)) in value_str.char_indices().enumerate() {
        if index == max_chars {
            return gpui::SharedString::new(value_str[..offset].to_string());
        }
    }
    value.clone()
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

async fn task_quick_action_request(
    client: &Client,
    task_id: TaskId,
    kind: TaskQuickActionKind,
) -> Result<ResponseResult, redesmyn_protocol::ErrorEnvelope> {
    let payload = match kind {
        TaskQuickActionKind::Start => RequestPayload::StartAgent(StartAgentRequest {
            task_id,
            agent_kind: AgentKind::Codex,
            interface_mode: AgentInterfaceMode::AppServer,
            initial_prompt: None,
            on_conflict: AgentMessageConflictAction::Fail,
        }),
        TaskQuickActionKind::Restart => RequestPayload::RestartAgent(RestartAgentRequest {
            task_id,
            agent_kind: AgentKind::Codex,
            interface_mode: AgentInterfaceMode::AppServer,
            initial_prompt: None,
        }),
        TaskQuickActionKind::Stop => RequestPayload::StopAgent(StopAgentRequest { task_id }),
    };

    client.request(payload).await
}

fn collapsed_task_border_color(
    state: TaskState,
    merge_readiness: MergeReadiness,
    theme: &redesmyn_ui::styles::UiTheme,
) -> gpui::Hsla {
    if matches!(state, TaskState::Blocked) || matches!(merge_readiness, MergeReadiness::Blocked) {
        return theme.colors.warning;
    }

    if matches!(state, TaskState::Done) {
        return theme.colors.completed;
    }

    if matches!(merge_readiness, MergeReadiness::Ready) {
        return theme.colors.success;
    }

    theme.colors.border
}

fn agent_status_dot(
    task_state: TaskState,
    agent_status: AgentStatus,
    continuable: bool,
    theme: &redesmyn_ui::styles::UiTheme,
    zoom: f32,
) -> gpui::Div {
    let size = px(12.0 * zoom);
    let radius = px(999.0);
    let transparent = theme.colors.surface.opacity(0.0);

    let (border, bg) = match task_state {
        TaskState::Blocked => (theme.colors.warning, theme.colors.warning),
        TaskState::Done => {
            let muted = theme.colors.foreground_muted.opacity(0.40);
            (muted, muted)
        }
        _ => match agent_status {
            AgentStatus::Unknown => (theme.colors.foreground_muted.opacity(0.60), transparent),
            AgentStatus::Running => (theme.colors.success, theme.colors.success),
            AgentStatus::Blocked => (theme.colors.warning, theme.colors.warning),
            AgentStatus::Error => (theme.colors.danger, theme.colors.danger),
            AgentStatus::Stopped if continuable => (theme.colors.info, transparent),
            AgentStatus::Stopped => (theme.colors.foreground_muted.opacity(0.60), transparent),
        },
    };

    div()
        .size(size)
        .rounded(radius)
        .border_1()
        .border_color(border)
        .bg(bg)
}

fn status_badge(label: impl Into<gpui::SharedString>, kind: BadgeKind) -> impl IntoElement {
    Badge::new(label).kind(kind).leading_dot(true)
}

fn collapsed_title_lines(
    cache: &mut HashMap<TaskId, CollapsedTitleCacheEntry>,
    task_id: TaskId,
    title: gpui::SharedString,
    wrap_width: gpui::Pixels,
    font: gpui::Font,
    font_size: gpui::Pixels,
    color: gpui::Hsla,
    window: &Window,
) -> (gpui::SharedString, Option<gpui::SharedString>) {
    let wrap_width_px = f32::from(wrap_width).round() as i32;
    let font_size_px = f32::from(font_size).round() as i32;

    if let Some(entry) = cache.get(&task_id)
        && entry.title == title
        && entry.wrap_width_px == wrap_width_px
        && entry.font_size_px == font_size_px
    {
        return (entry.line1.clone(), entry.line2.clone());
    }

    let sanitized = title.as_ref().replace(['\r', '\n'], " ").trim().to_string();

    let (line1, line2) =
        wrap_two_lines_wordwise(&sanitized, wrap_width, &font, font_size, color, window);

    let line1 = gpui::SharedString::new(line1);
    let line2 = line2.map(gpui::SharedString::new);

    cache.insert(
        task_id,
        CollapsedTitleCacheEntry {
            title,
            wrap_width_px,
            font_size_px,
            line1: line1.clone(),
            line2: line2.clone(),
        },
    );

    (line1, line2)
}

fn wrap_two_lines_wordwise(
    text: &str,
    max_width: gpui::Pixels,
    font: &gpui::Font,
    font_size: gpui::Pixels,
    color: gpui::Hsla,
    window: &Window,
) -> (String, Option<String>) {
    if text.is_empty() {
        return (String::new(), None);
    }

    let full_width = measure_text_width(text, font, font_size, color, window);
    if full_width <= max_width {
        return (text.to_string(), None);
    }

    // Leave a little headroom so the first line doesn't pick up an ellipsis due to pixel snapping.
    const LINE1_FUDGE_PX: f32 = 4.0;
    let max_width_line1 = px((f32::from(max_width) - LINE1_FUDGE_PX).max(0.0));

    let line1_end = fit_prefix_end(text, max_width_line1, font, font_size, color, window);
    let line1 = text[..line1_end].trim_end().to_string();

    let remainder = text[line1_end..].trim_start();
    if remainder.is_empty() {
        return (line1, None);
    }

    (line1, Some(remainder.to_string()))
}

fn fit_prefix_end(
    text: &str,
    max_width: gpui::Pixels,
    font: &gpui::Font,
    font_size: gpui::Pixels,
    color: gpui::Hsla,
    window: &Window,
) -> usize {
    let mut candidates: Vec<usize> = Vec::new();
    let mut in_word = false;

    for (ix, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if in_word {
                candidates.push(ix);
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }

    if in_word {
        candidates.push(text.len());
    }

    candidates.sort_unstable();
    candidates.dedup();

    let mut best = 0usize;
    let mut lo = 0usize;
    let mut hi = candidates.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let end = candidates[mid];
        let candidate = text[..end].trim_end();
        let width = measure_text_width(candidate, font, font_size, color, window);
        if width <= max_width {
            best = end;
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if best > 0 {
        return best;
    }

    // Fall back to char-level fitting when even the first "word" does not fit.
    let mut char_ends: Vec<usize> = text
        .char_indices()
        .map(|(ix, ch)| ix + ch.len_utf8())
        .collect();
    char_ends.sort_unstable();
    char_ends.dedup();

    let mut lo = 0usize;
    let mut hi = char_ends.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let end = char_ends[mid];
        let candidate = text[..end].trim_end();
        let width = measure_text_width(candidate, font, font_size, color, window);
        if width <= max_width {
            best = end;
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    best.max(text.chars().next().map(|ch| ch.len_utf8()).unwrap_or(0))
}

fn measure_text_width(
    text: &str,
    font: &gpui::Font,
    font_size: gpui::Pixels,
    color: gpui::Hsla,
    window: &Window,
) -> gpui::Pixels {
    if text.is_empty() {
        return px(0.0);
    }

    let run = TextRun {
        len: text.len(),
        font: font.clone(),
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    };

    let shaped = window.text_system().shape_line(
        gpui::SharedString::new(text.to_string()),
        font_size,
        std::slice::from_ref(&run),
        None,
    );

    shaped.width
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

fn task_state_badge_kind(state: TaskState) -> BadgeKind {
    match state {
        TaskState::InProgress => BadgeKind::Info,
        TaskState::Blocked => BadgeKind::Warning,
        TaskState::Done => BadgeKind::Completed,
        TaskState::Todo | TaskState::Unknown => BadgeKind::Neutral,
    }
}

fn task_state_badge(state: TaskState) -> impl IntoElement {
    status_badge(task_state_label(state), task_state_badge_kind(state))
}

fn merge_readiness_label(readiness: MergeReadiness) -> &'static str {
    match readiness {
        MergeReadiness::Ready => "Merge ready",
        MergeReadiness::Blocked => "Merge blocked",
        MergeReadiness::Unknown => "Merge ?",
    }
}

fn merge_readiness_badge_kind(readiness: MergeReadiness) -> BadgeKind {
    match readiness {
        MergeReadiness::Ready => BadgeKind::Success,
        MergeReadiness::Blocked => BadgeKind::Warning,
        MergeReadiness::Unknown => BadgeKind::Neutral,
    }
}

fn merge_readiness_badge(readiness: MergeReadiness) -> impl IntoElement {
    status_badge(
        merge_readiness_label(readiness),
        merge_readiness_badge_kind(readiness),
    )
}

fn agent_status_value_label(status: AgentStatus) -> &'static str {
    match status {
        AgentStatus::Running => "Running",
        AgentStatus::Blocked => "Blocked",
        AgentStatus::Stopped => "Stopped",
        AgentStatus::Error => "Error",
        AgentStatus::Unknown => "Unknown",
    }
}

fn task_status_chips(
    state: TaskState,
    merge_readiness: MergeReadiness,
    agent_status: AgentStatus,
    agent_continuable: bool,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    div()
        .flex()
        .flex_row()
        .gap(theme.spacing.xs)
        .items_center()
        .child(task_state_badge(state))
        .child(merge_readiness_badge(merge_readiness))
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(theme.spacing.xs)
                .child(agent_status_dot(
                    state,
                    agent_status,
                    agent_continuable,
                    theme,
                    1.0,
                ))
                .child(
                    div()
                        .text_xs()
                        .text_color(theme.colors.foreground_muted)
                        .child(agent_status_value_label(agent_status)),
                ),
        )
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
                .w(px(96.0))
                .text_xs()
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
        .pt(theme.spacing.sm)
        .child(
            div()
                .pb(theme.spacing.xs)
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child(title),
        )
        .child(body)
        .child(
            div()
                .pt(theme.spacing.sm)
                .border_b_1()
                .border_color(theme.colors.border.opacity(0.35)),
        )
}

fn details_nav_separator(theme: &redesmyn_ui::styles::UiTheme) -> impl IntoElement {
    div()
        .text_xs()
        .text_color(theme.colors.foreground_muted.opacity(0.6))
        .child("·")
}

fn details_nav_link(
    id: impl Into<gpui::ElementId>,
    label: &'static str,
    target_item_ix: usize,
    active: bool,
    scroll: ScrollHandle,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    let hover_bg = theme.colors.accent.opacity(0.6);
    let active_bg = theme.colors.accent.opacity(0.75);
    let fg = if active {
        theme.colors.foreground
    } else {
        theme.colors.foreground_muted
    };

    div()
        .id(id)
        .flex()
        .items_center()
        .px(theme.spacing.xs)
        .py(px(2.0))
        .rounded(px(999.0))
        .text_xs()
        .text_color(fg)
        .when(active, |this| {
            this.bg(active_bg).font_weight(FontWeight::SEMIBOLD)
        })
        .cursor_pointer()
        .hover(move |this| this.bg(hover_bg))
        .on_click(move |event, _window, cx| {
            if event.standard_click() {
                scroll.scroll_to_top_of_item(target_item_ix);
            }
            cx.stop_propagation();
        })
        .child(label)
}

fn task_details_overview_section(
    branch_name: Option<gpui::SharedString>,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    details_section(
        "Overview",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .when_some(branch_name, |this, name| {
                this.child(details_kv_row("Branch", div().child(name), theme))
            })
            .child(details_kv_row("Updated", div().child("—"), theme)),
        theme,
    )
}

fn task_details_agent_section(
    action_button_id: impl Into<gpui::ElementId>,
    demo_action: UserActionState,
    start_demo_action: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    let mut body = div().flex().flex_col().gap(theme.spacing.sm);
    if let Some(error) = demo_action.error.clone() {
        body = body.child(
            Callout::new(error)
                .kind(CalloutKind::Danger)
                .title("Action failed"),
        );
    }

    body = body
        .child(
            div()
                .flex()
                .flex_col()
                .gap(theme.spacing.xs)
                .child(details_kv_row("Session", div().child("unbound"), theme)),
        )
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .gap(theme.spacing.sm)
                .child(
                    TextButton::new(action_button_id, "Run demo")
                        .kind(ButtonKind::Secondary)
                        .small()
                        .disabled(demo_action.in_flight)
                        .disabled_reason("Running…")
                        .on_click(start_demo_action),
                )
                .when(demo_action.in_flight, |this| {
                    this.child(ProgressPill::new("Running"))
                }),
        );

    details_section("Agent", body, theme)
}

fn task_details_merge_section(theme: &redesmyn_ui::styles::UiTheme) -> impl IntoElement {
    details_section(
        "Merge / restack",
        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(details_kv_row("Next", div().child("—"), theme)),
        theme,
    )
}

fn task_details_identifiers_section(
    node_id: GraphNodeId,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    details_section(
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
    )
}

fn task_details_artifacts_section(
    entity_id: gpui::EntityId,
    node_key: gpui::SharedString,
    theme: &redesmyn_ui::styles::UiTheme,
) -> impl IntoElement {
    details_section(
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
                .small()
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
                .small()
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
                .small()
                .disabled(true)
                .disabled_reason("Coming soon"),
            ),
        theme,
    )
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

        let selection_bar_visible = |scene: &GraphScene| {
            scene.selection().selected_edge.is_none() && scene.selection().selected_nodes.len() > 1
        };

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
