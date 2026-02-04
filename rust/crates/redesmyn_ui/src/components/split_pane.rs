use serde::{Deserialize, Serialize};

use gpui::{
    ClickEvent, Context, CursorStyle, DragMoveEvent, EventEmitter, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, Point, Render, Window, div, prelude::*, px,
};

use crate::utils::theme_for_window;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitPaneAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Debug)]
pub enum SplitPaneEvent {
    StateChanged(SplitPaneState),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SplitPaneState {
    pub primary_size_px: f32,
    pub collapsed: bool,
}

impl Default for SplitPaneState {
    fn default() -> Self {
        Self {
            primary_size_px: 360.0,
            collapsed: false,
        }
    }
}

pub struct SplitPane {
    axis: SplitPaneAxis,
    state: SplitPaneState,
    min_primary_px: f32,
    dragging: Option<DragState>,
    primary: gpui::AnyView,
    secondary: gpui::AnyView,
}

#[derive(Debug, Clone, Copy)]
struct DragState {
    origin: Point<Pixels>,
    start_primary_px: f32,
}

#[derive(Debug, Clone, Copy)]
struct SplitPaneResizeDrag;

struct SplitPaneResizeDragGhost;

impl SplitPaneResizeDragGhost {
    fn new(_: &mut Context<Self>) -> Self {
        Self
    }
}

impl Render for SplitPaneResizeDragGhost {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().size(px(0.0)).opacity(0.0)
    }
}

impl EventEmitter<SplitPaneEvent> for SplitPane {}

impl SplitPane {
    pub fn new(
        axis: SplitPaneAxis,
        state: SplitPaneState,
        primary: gpui::AnyView,
        secondary: gpui::AnyView,
    ) -> Self {
        Self {
            axis,
            state,
            min_primary_px: 240.0,
            dragging: None,
            primary,
            secondary,
        }
    }

    pub fn min_primary_px(mut self, min_primary_px: f32) -> Self {
        self.min_primary_px = min_primary_px;
        self
    }

    pub fn state(&self) -> SplitPaneState {
        self.state
    }

    pub fn toggle_collapsed(&mut self, cx: &mut Context<Self>) {
        self.state.collapsed = !self.state.collapsed;
        cx.emit(SplitPaneEvent::StateChanged(self.state));
        cx.notify();
    }

    fn on_divider_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.state.collapsed {
            return;
        }
        self.dragging = Some(DragState {
            origin: event.position,
            start_primary_px: self.state.primary_size_px,
        });
        cx.notify();
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let dragging = self.dragging.take();
        if let Some(dragging) = dragging {
            if dragging.start_primary_px != self.state.primary_size_px {
                cx.emit(SplitPaneEvent::StateChanged(self.state));
            }
            cx.notify();
        }
    }

    fn on_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.update_primary_size(event.position, window, cx);
    }

    fn on_drag_move(
        &mut self,
        event: &DragMoveEvent<SplitPaneResizeDrag>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.update_primary_size(event.event.position, window, cx);
    }

    fn update_primary_size(
        &mut self,
        cursor_position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(dragging) = self.dragging else {
            return;
        };

        let max_primary_px = max_primary_px_for(self.axis, window);
        let delta = match self.axis {
            SplitPaneAxis::Horizontal => f32::from(cursor_position.x - dragging.origin.x),
            SplitPaneAxis::Vertical => f32::from(cursor_position.y - dragging.origin.y),
        };

        let primary_size_px = (dragging.start_primary_px + delta)
            .clamp(self.min_primary_px, max_primary_px)
            .max(0.0);

        if (primary_size_px - self.state.primary_size_px).abs() <= f32::EPSILON {
            return;
        }

        self.state.primary_size_px = primary_size_px;
        cx.notify();
    }

    fn on_divider_click(
        &mut self,
        event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.click_count() >= 2 {
            self.toggle_collapsed(cx);
        }
    }
}

impl Render for SplitPane {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let axis = self.axis;

        let resize_cursor = match axis {
            SplitPaneAxis::Horizontal => CursorStyle::ResizeLeftRight,
            SplitPaneAxis::Vertical => CursorStyle::ResizeUpDown,
        };

        let divider_line_color = theme.colors.ring.opacity(0.25);

        let divider_line_thickness = px(1.0);
        let divider_hit_inset = px(-3.0);

        let divider_hit = div()
            .id(("split_pane_divider_hit", cx.entity_id()))
            .absolute()
            .when(axis == SplitPaneAxis::Horizontal, |this| {
                this.top(px(0.0))
                    .bottom(px(0.0))
                    .left(divider_hit_inset)
                    .right(divider_hit_inset)
            })
            .when(axis == SplitPaneAxis::Vertical, |this| {
                this.left(px(0.0))
                    .right(px(0.0))
                    .top(divider_hit_inset)
                    .bottom(divider_hit_inset)
            })
            .cursor(resize_cursor)
            .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
            .focusable()
            .when(!self.state.collapsed, |this| {
                this.on_mouse_down(MouseButton::Left, cx.listener(Self::on_divider_mouse_down))
                    .on_drag(SplitPaneResizeDrag, |_, _, _, cx| {
                        cx.new(SplitPaneResizeDragGhost::new)
                    })
            })
            .on_click(cx.listener(Self::on_divider_click));

        let divider = div()
            .id(("split_pane_divider", cx.entity_id()))
            .relative()
            .when(axis == SplitPaneAxis::Horizontal, |this| {
                this.w(divider_line_thickness).h_full()
            })
            .when(axis == SplitPaneAxis::Vertical, |this| {
                this.h(divider_line_thickness).w_full()
            })
            .child(divider_hit)
            .child(
                div()
                    .when(axis == SplitPaneAxis::Horizontal, |this| {
                        this.w(divider_line_thickness).h_full()
                    })
                    .when(axis == SplitPaneAxis::Vertical, |this| {
                        this.h(divider_line_thickness).w_full()
                    })
                    .bg(divider_line_color),
            );

        let primary_size = if self.state.collapsed {
            px(0.0)
        } else {
            px(self.state.primary_size_px)
        };

        let primary = div()
            .when(axis == SplitPaneAxis::Horizontal, |this| {
                this.w(primary_size)
            })
            .when(axis == SplitPaneAxis::Vertical, |this| this.h(primary_size))
            .overflow_hidden()
            .child(self.primary.clone());

        let secondary = div()
            .flex_1()
            .overflow_hidden()
            .child(self.secondary.clone());

        div()
            .id(("split_pane_root", cx.entity_id()))
            .flex()
            .size_full()
            .when(axis == SplitPaneAxis::Horizontal, |this| this.flex_row())
            .when(axis == SplitPaneAxis::Vertical, |this| this.flex_col())
            .on_drag_move(cx.listener(Self::on_drag_move))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .child(primary)
            .child(divider)
            .child(secondary)
    }
}

fn max_primary_px_for(axis: SplitPaneAxis, window: &Window) -> f32 {
    let bounds = window.bounds();
    match axis {
        SplitPaneAxis::Horizontal => (f32::from(bounds.size.width) * 0.8).max(0.0),
        SplitPaneAxis::Vertical => (f32::from(bounds.size.height) * 0.8).max(0.0),
    }
}
