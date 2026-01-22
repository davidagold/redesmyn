use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use gpui::{
    App, AsyncApp, ClickEvent, Context, CursorStyle, FocusHandle, Focusable, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, ScrollHandle, ScrollWheelEvent, Task,
    Window, canvas, div, fill, prelude::*, px,
};

use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ScrollArea, TextButton,
};
use redesmyn_ui::utils::UserActionState;
use redesmyn_ui::utils::theme_for_window;

use crate::camera::{GraphCamera, GraphCameraLimits};
use crate::geometry::{DEFAULT_EDGE_THICKNESS_PX, edge_segments_in_window};
use crate::hit_test::{GraphHit, hit_test};
use crate::scene::{GraphEdgeId, GraphNodeId, GraphScene};

#[derive(Debug, Clone, Copy)]
struct PanDrag {
    start_mouse: gpui::Point<gpui::Pixels>,
    start_origin_world: gpui::Point<f32>,
}

#[derive(Debug, Clone, Copy)]
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

pub struct GraphView {
    focus_handle: FocusHandle,
    scene: GraphScene,
    camera: GraphCamera,
    pan_drag: Option<PanDrag>,
    last_canvas_bounds: Option<gpui::Bounds<gpui::Pixels>>,
    expanded_session_scroll: ScrollHandle,
    expanded_details_scroll: ScrollHandle,
    demo_action: UserActionState,
    demo_action_task: Option<Task<()>>,
    layout_animation: Option<LayoutAnimation>,
}

impl GraphView {
    pub fn new_demo(cx: &mut Context<Self>) -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_view.new_demo");
        let _guard = span.enter();

        Self {
            focus_handle: cx.focus_handle(),
            scene: GraphScene::demo(),
            camera: GraphCamera::new(GraphCameraLimits::default()),
            pan_drag: None,
            last_canvas_bounds: None,
            expanded_session_scroll: ScrollHandle::new(),
            expanded_details_scroll: ScrollHandle::new(),
            demo_action: UserActionState::default(),
            demo_action_task: None,
            layout_animation: None,
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
        if let Some(node) = self.visual_selected_node() {
            return format!("Selected node: {node}");
        }
        if let Some(edge) = self.scene.selection().selected_edge {
            return format!("Selected edge: {edge}");
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
        }

        cx.notify();
    }

    fn select_node(&mut self, node_id: GraphNodeId, cx: &mut Context<Self>) {
        self.update_selection(|scene| scene.select_node(node_id), cx);
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

        let Some(canvas_bounds) = self.last_canvas_bounds else {
            return;
        };

        match hit_test(&self.scene, &self.camera, canvas_bounds, event.position) {
            Some(GraphHit::Node(id)) => {
                redesmyn_logging::tracing::trace!(node_id = %id, "graph selection changed");
                self.select_node(id, cx);
            }
            Some(GraphHit::Edge(id)) => {
                redesmyn_logging::tracing::trace!(edge_id = %id, "graph selection changed");
                self.select_edge(id, cx);
            }
            None => {
                self.pan_drag = Some(PanDrag {
                    start_mouse: event.position,
                    start_origin_world: self.camera.origin_world(),
                });
                self.clear_selection(cx);
            }
        }
    }

    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(drag) = self.pan_drag else {
            return;
        };

        if !event.dragging() {
            self.pan_drag = None;
            return;
        }

        let delta = event.position - drag.start_mouse;
        let mut camera = self.camera;
        camera.set_origin_world(drag.start_origin_world);
        camera.pan_by_screen_delta(delta);
        self.camera = camera;

        cx.notify();
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if self.pan_drag.take().is_some() {
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

    fn paint_graph(
        &self,
        canvas_bounds: gpui::Bounds<gpui::Pixels>,
        window: &mut Window,
        cx: &App,
    ) {
        let theme = theme_for_window(window, cx);
        let t = self.layout_animation_progress_for_paint();

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
            let edge_color = if selected {
                theme.colors.ring
            } else {
                theme.colors.border.opacity(0.9)
            };

            let segments =
                edge_segments_in_window(from_bounds, to_bounds, DEFAULT_EDGE_THICKNESS_PX);
            for segment in segments.segments.iter().flatten() {
                window.paint_quad(fill(*segment, edge_color));
            }
        }
    }
}

impl Focusable for GraphView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for GraphView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
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
            let selected = self.visual_selected_node();

            let mut layer = div().absolute().inset_0();

            for node in self.scene.nodes() {
                let bounds_in_window =
                    self.node_bounds_in_window_for_progress(node.id, canvas_bounds, t);
                let local_origin = bounds_in_window.origin - canvas_bounds.origin;
                let node_id = node.id;
                let node_key: gpui::SharedString = node_id.to_string().into();
                let is_selected = selected == Some(node_id);
                let is_expanded = is_selected;

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
                    .occlude()
                    .on_mouse_down(MouseButton::Left, {
                        let graph = graph.clone();
                        move |_, _, cx| {
                            graph.update(cx, |this, cx| this.select_node(node_id, cx));
                            cx.stop_propagation();
                        }
                    });

                if !is_expanded {
                    let title = node.title.clone();
                    card = card.child(
                        div()
                            .p(theme.spacing.sm)
                            .flex()
                            .flex_col()
                            .gap(theme.spacing.xs)
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .justify_between()
                                    .gap(theme.spacing.sm)
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w_0()
                                            .text_sm()
                                            .text_color(theme.colors.foreground)
                                            .truncate()
                                            .child(title),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .flex_row()
                                    .gap(theme.spacing.xs)
                                    .items_center()
                                    .child(status_chip(
                                        "idle",
                                        theme.colors.accent,
                                        theme.colors.foreground,
                                        &theme,
                                    ))
                                    .child(status_chip(
                                        "merge: ?",
                                        theme.colors.surface,
                                        theme.colors.foreground_muted,
                                        &theme,
                                    )),
                            ),
                    );
                } else {
                    let title = node.title.clone();
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
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .text_color(theme.colors.foreground_muted)
                                                    .child(format!("{node_id}")),
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
                        let had_selection = this.scene.selection().selected_node.is_some()
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
                    .flex_1()
                    .overflow_hidden()
                    .relative()
                    .cursor(if self.pan_drag.is_some() {
                        CursorStyle::ClosedHand
                    } else {
                        CursorStyle::Arrow
                    })
                    .on_mouse_down(MouseButton::Left, cx.listener(Self::on_mouse_down))
                    .on_mouse_move(cx.listener(Self::on_mouse_move))
                    .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
                    .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
                    .on_scroll_wheel(cx.listener(Self::on_scroll_wheel))
                    .child(canvas)
                    .child(nodes_layer),
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
            .child(details_kv_row(
                "State",
                status_chip(
                    "unknown",
                    theme.colors.accent,
                    theme.colors.foreground,
                    theme,
                ),
                theme,
            ))
            .child(details_kv_row(
                "Merge",
                status_chip(
                    "unknown",
                    theme.colors.surface,
                    theme.colors.foreground_muted,
                    theme,
                ),
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
                    .child(details_kv_row("Status", div().child("idle"), theme))
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
            .child(details_kv_row("Ready", div().child("—"), theme))
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
