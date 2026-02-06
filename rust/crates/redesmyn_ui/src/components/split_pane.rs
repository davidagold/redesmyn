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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitPaneResizeMode {
    /// Resize panes live while dragging the divider.
    Live,
    /// Keep pane layout fixed while dragging the divider, and only apply the new size on mouse up.
    ///
    /// This can dramatically improve performance when pane contents are expensive to reflow
    /// (e.g. markdown/chat history) by avoiding per-pixel re-layout during the drag gesture.
    Deferred,
}

impl Default for SplitPaneResizeMode {
    fn default() -> Self {
        Self::Live
    }
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
    resize_mode: SplitPaneResizeMode,
    dragging: Option<DragState>,
    drag_preview_primary_px: Option<f32>,
    drag_render_pending: bool,
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
            resize_mode: SplitPaneResizeMode::default(),
            dragging: None,
            drag_preview_primary_px: None,
            drag_render_pending: false,
            primary,
            secondary,
        }
    }

    pub fn min_primary_px(mut self, min_primary_px: f32) -> Self {
        self.min_primary_px = min_primary_px;
        self
    }

    pub fn resize_mode(mut self, resize_mode: SplitPaneResizeMode) -> Self {
        self.resize_mode = resize_mode;
        self
    }

    pub fn state(&self) -> SplitPaneState {
        self.state
    }

    pub fn toggle_collapsed(&mut self, cx: &mut Context<Self>) {
        self.dragging = None;
        self.drag_preview_primary_px = None;
        self.drag_render_pending = false;
        self.state.collapsed = !self.state.collapsed;
        cx.emit(SplitPaneEvent::StateChanged(self.state));
        cx.notify();
    }

    fn request_drag_render(&mut self, cx: &mut Context<Self>) {
        if self.drag_render_pending {
            return;
        }
        self.drag_render_pending = true;
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
        self.drag_render_pending = false;
        if self.resize_mode == SplitPaneResizeMode::Deferred {
            self.drag_preview_primary_px = Some(self.state.primary_size_px);
        }
        cx.notify();
    }

    fn on_mouse_up(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let dragging = self.dragging.take();
        if let Some(dragging) = dragging {
            self.drag_render_pending = false;
            if self.resize_mode == SplitPaneResizeMode::Deferred {
                if let Some(preview) = self.drag_preview_primary_px.take() {
                    self.state.primary_size_px = preview;
                }
            } else {
                self.drag_preview_primary_px = None;
            }
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

        let current_primary_px = match self.resize_mode {
            SplitPaneResizeMode::Live => self.state.primary_size_px,
            SplitPaneResizeMode::Deferred => self
                .drag_preview_primary_px
                .unwrap_or(self.state.primary_size_px),
        };

        if (primary_size_px - current_primary_px).abs() <= f32::EPSILON {
            return;
        }

        match self.resize_mode {
            SplitPaneResizeMode::Live => {
                self.state.primary_size_px = primary_size_px;
            }
            SplitPaneResizeMode::Deferred => {
                self.drag_preview_primary_px = Some(primary_size_px);
            }
        }
        self.request_drag_render(cx);
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
        self.drag_render_pending = false;
        let theme = theme_for_window(window, cx);
        let axis = self.axis;

        let resize_cursor = match axis {
            SplitPaneAxis::Horizontal => CursorStyle::ResizeLeftRight,
            SplitPaneAxis::Vertical => CursorStyle::ResizeUpDown,
        };

        let divider_line_color = theme.colors.ring.opacity(0.25);

        let divider_line_thickness = px(1.0);
        let effective_divider_thickness = if self.state.collapsed {
            px(0.0)
        } else {
            divider_line_thickness
        };
        let divider_hit_inset = px(-3.0);

        let show_drag_ghost = self.resize_mode == SplitPaneResizeMode::Deferred
            && self.dragging.is_some()
            && !self.state.collapsed;

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
                this.w(effective_divider_thickness).h_full()
            })
            .when(axis == SplitPaneAxis::Vertical, |this| {
                this.h(effective_divider_thickness).w_full()
            })
            .child(divider_hit)
            .child(
                div()
                    .when(axis == SplitPaneAxis::Horizontal, |this| {
                        this.w(effective_divider_thickness).h_full()
                    })
                    .when(axis == SplitPaneAxis::Vertical, |this| {
                        this.h(effective_divider_thickness).w_full()
                    })
                    .bg(divider_line_color)
                    .opacity(if show_drag_ghost { 0.0 } else { 1.0 }),
            );

        let drag_ghost = if show_drag_ghost {
            let preview_px = self
                .drag_preview_primary_px
                .unwrap_or(self.state.primary_size_px);
            let preview = px(preview_px);
            let ghost_color = theme.colors.ring.opacity(0.35);

            Some(
                div()
                    .id(("split_pane_divider_ghost", cx.entity_id()))
                    .absolute()
                    .when(axis == SplitPaneAxis::Horizontal, |this| {
                        this.top(px(0.0))
                            .bottom(px(0.0))
                            .left(preview)
                            .w(divider_line_thickness)
                    })
                    .when(axis == SplitPaneAxis::Vertical, |this| {
                        this.left(px(0.0))
                            .right(px(0.0))
                            .top(preview)
                            .h(divider_line_thickness)
                    })
                    .bg(ghost_color),
            )
        } else {
            None
        };

        let primary_size = if self.state.collapsed {
            px(0.0)
        } else {
            px(self.state.primary_size_px)
        };

        let divider_offset = primary_size;

        let primary = div()
            .absolute()
            .overflow_hidden()
            .when(axis == SplitPaneAxis::Horizontal, |this| {
                this.left(px(0.0))
                    .top(px(0.0))
                    .bottom(px(0.0))
                    .w(primary_size)
            })
            .when(axis == SplitPaneAxis::Vertical, |this| {
                this.left(px(0.0))
                    .right(px(0.0))
                    .top(px(0.0))
                    .h(primary_size)
            })
            .child(self.primary.clone());

        let secondary = div()
            .absolute()
            .overflow_hidden()
            .when(axis == SplitPaneAxis::Horizontal, |this| {
                this.left(divider_offset + effective_divider_thickness)
                    .right(px(0.0))
                    .top(px(0.0))
                    .bottom(px(0.0))
            })
            .when(axis == SplitPaneAxis::Vertical, |this| {
                this.left(px(0.0))
                    .right(px(0.0))
                    .top(divider_offset + effective_divider_thickness)
                    .bottom(px(0.0))
            })
            .child(self.secondary.clone());

        let divider = divider
            .absolute()
            .when(axis == SplitPaneAxis::Horizontal, |this| {
                this.left(divider_offset).top(px(0.0)).bottom(px(0.0))
            })
            .when(axis == SplitPaneAxis::Vertical, |this| {
                this.top(divider_offset).left(px(0.0)).right(px(0.0))
            });

        let mut root = div()
            .id(("split_pane_root", cx.entity_id()))
            .relative()
            .size_full()
            .on_drag_move(cx.listener(Self::on_drag_move))
            .on_mouse_move(cx.listener(Self::on_mouse_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_mouse_up))
            .child(secondary)
            .when(!self.state.collapsed, |this| this.child(divider))
            .when(!self.state.collapsed, |this| this.child(primary));

        if let Some(ghost) = drag_ghost {
            root = root.child(ghost);
        }

        root
    }
}

fn max_primary_px_for(axis: SplitPaneAxis, window: &Window) -> f32 {
    let bounds = window.bounds();
    match axis {
        SplitPaneAxis::Horizontal => (f32::from(bounds.size.width) * 0.8).max(0.0),
        SplitPaneAxis::Vertical => (f32::from(bounds.size.height) * 0.8).max(0.0),
    }
}
