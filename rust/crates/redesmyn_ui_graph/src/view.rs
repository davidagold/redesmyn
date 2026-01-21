use gpui::{
    canvas, div, fill, prelude::*, px, quad, App, Context, CursorStyle, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, Render, ScrollWheelEvent, Window,
};

use redesmyn_ui::utils::theme_for_window;

use crate::camera::{GraphCamera, GraphCameraLimits};
use crate::geometry::{edge_segments_in_window, node_bounds_in_window, DEFAULT_EDGE_THICKNESS_PX};
use crate::hit_test::{hit_test, GraphHit};
use crate::scene::GraphScene;

#[derive(Debug, Clone, Copy)]
struct PanDrag {
    start_mouse: gpui::Point<gpui::Pixels>,
    start_origin_world: gpui::Point<f32>,
}

pub struct GraphView {
    scene: GraphScene,
    camera: GraphCamera,
    pan_drag: Option<PanDrag>,
    last_canvas_bounds: Option<gpui::Bounds<gpui::Pixels>>,
}

impl GraphView {
    pub fn new_demo(_: &mut Context<Self>) -> Self {
        let span = redesmyn_logging::redesmyn_info_span!("ui.graph_view.new_demo");
        let _guard = span.enter();

        Self {
            scene: GraphScene::demo(),
            camera: GraphCamera::new(GraphCameraLimits::default()),
            pan_drag: None,
            last_canvas_bounds: None,
        }
    }

    fn selection_label(&self) -> String {
        if let Some(node) = self.scene.selection().selected_node {
            return format!("Selected node: {node}");
        }
        if let Some(edge) = self.scene.selection().selected_edge {
            return format!("Selected edge: {edge}");
        }
        "Selected: <none>".to_string()
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
                redesmyn_logging::tracing::debug!(node_id = %id, "graph selection changed");
                self.scene.select_node(id);
            }
            Some(GraphHit::Edge(id)) => {
                redesmyn_logging::tracing::debug!(edge_id = %id, "graph selection changed");
                self.scene.select_edge(id);
            }
            None => {
                self.scene.clear_selection();
                self.pan_drag = Some(PanDrag {
                    start_mouse: event.position,
                    start_origin_world: self.camera.origin_world(),
                });
            }
        }

        cx.notify();
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

        for edge in self.scene.edges() {
            let Some(from) = self.scene.node(edge.id.from) else {
                continue;
            };
            let Some(to) = self.scene.node(edge.id.to) else {
                continue;
            };

            let from_bounds = node_bounds_in_window(&self.scene, &self.camera, from, canvas_bounds);
            let to_bounds = node_bounds_in_window(&self.scene, &self.camera, to, canvas_bounds);

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

        for node in self.scene.nodes() {
            let bounds = node_bounds_in_window(&self.scene, &self.camera, node, canvas_bounds);
            let selected = self.scene.selection().selected_node == Some(node.id);

            let background = if selected {
                theme.colors.ring.opacity(0.14)
            } else {
                theme.colors.surface_elevated
            };

            let border_color = if selected {
                theme.colors.ring
            } else {
                theme.colors.border.opacity(0.9)
            };

            let node_quad = quad(
                bounds,
                theme.radius.md,
                background,
                1.0,
                border_color,
                gpui::BorderStyle::Solid,
            );

            window.paint_quad(node_quad);
        }
    }
}

impl Render for GraphView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let label = self.selection_label();
        let zoom = self.camera.zoom();

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
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
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

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .child(debug_bar)
            .child(
                div()
                    .flex_1()
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
                    .child(canvas),
            )
    }
}
