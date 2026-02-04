use std::time::{Duration, Instant};

use gpui::{
    AnyElement, App, Bounds, Corners, DispatchPhase, Element, ElementId, EntityId, GlobalElementId,
    Hitbox, HitboxBehavior, InspectorElementId, LayoutId, ListState, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, Pixels, Point, ScrollHandle, Size, Window, fill, px, size,
};

use crate::utils::{TransitionMap, theme_for_window};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollbarAxis {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone)]
pub enum ScrollbarTarget {
    ScrollHandle(ScrollHandle),
    ListState(ListState),
}

impl ScrollbarTarget {
    fn bounds(&self) -> Bounds<Pixels> {
        match self {
            Self::ScrollHandle(handle) => handle.bounds(),
            Self::ListState(state) => state.viewport_bounds(),
        }
    }

    fn offset(&self) -> Point<Pixels> {
        match self {
            Self::ScrollHandle(handle) => handle.offset(),
            Self::ListState(state) => state.scroll_px_offset_for_scrollbar(),
        }
    }

    fn max_offset(&self) -> Size<Pixels> {
        match self {
            Self::ScrollHandle(handle) => handle.max_offset(),
            Self::ListState(state) => state.max_offset_for_scrollbar(),
        }
    }

    fn set_offset_from_scrollbar(&self, point: Point<Pixels>) {
        match self {
            Self::ScrollHandle(handle) => handle.set_offset(point),
            Self::ListState(state) => state.set_offset_from_scrollbar(point),
        }
    }

    fn scrollbar_drag_started(&self) {
        if let Self::ListState(state) = self {
            state.scrollbar_drag_started();
        }
    }

    fn scrollbar_drag_ended(&self) {
        if let Self::ListState(state) = self {
            state.scrollbar_drag_ended();
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ScrollbarStyle {
    pub axis: ScrollbarAxis,
    pub thickness: Pixels,
    /// Inset from the far edge of the scroll viewport (right for vertical scrollbars, bottom for
    /// horizontal). Negative values are allowed and will place the thumb "into the gutter"
    /// outside the viewport (useful when the scrollable is inside a padded container).
    pub inset: Pixels,
    pub min_thumb_length: Pixels,
    pub idle_delay: Duration,
    pub fade_duration: Duration,
}

impl Default for ScrollbarStyle {
    fn default() -> Self {
        Self {
            axis: ScrollbarAxis::Vertical,
            thickness: px(6.0),
            inset: px(0.0),
            min_thumb_length: px(24.0),
            idle_delay: Duration::from_millis(700),
            fade_duration: Duration::from_millis(140),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ThumbGeometry {
    offset: Pixels,
    length: Pixels,
}

#[derive(Debug, Clone, Copy)]
struct ScrollbarGeometry {
    viewport: Bounds<Pixels>,
    max_offset: Pixels,
    scroll_pos: Pixels,
    thumb: ThumbGeometry,
    thickness: Pixels,
    thumb_bounds: Bounds<Pixels>,
    gutter_hitbox_bounds: Bounds<Pixels>,
}

fn clamp_scroll_axis(offset: Pixels, max_offset: Pixels) -> Pixels {
    // Offsets are negative when scrolled down/right (content translated up/left). Map to a positive
    // scroll position in [0, max_offset].
    (-offset).max(px(0.0)).min(max_offset)
}

fn compute_thumb(
    viewport_len: Pixels,
    max_offset: Pixels,
    scroll_pos: Pixels,
    min_thumb_len: Pixels,
) -> Option<ThumbGeometry> {
    if viewport_len <= px(1.0) || max_offset <= px(0.5) {
        return None;
    }

    let content_len = viewport_len + max_offset;
    if content_len <= px(1.0) {
        return None;
    }

    let mut thumb_len = viewport_len * (viewport_len / content_len);
    thumb_len = thumb_len.max(min_thumb_len).min(viewport_len);

    let track_len = (viewport_len - thumb_len).max(px(0.0));
    let t = (scroll_pos / max_offset).clamp(0.0, 1.0);
    let offset = track_len * t;

    Some(ThumbGeometry {
        offset,
        length: thumb_len,
    })
}

fn compute_geometry(target: &ScrollbarTarget, style: ScrollbarStyle) -> Option<ScrollbarGeometry> {
    let viewport = target.bounds();
    if viewport.size.width <= px(1.0) || viewport.size.height <= px(1.0) {
        return None;
    }

    let (max_offset, raw_offset) = match style.axis {
        ScrollbarAxis::Vertical => (target.max_offset().height, target.offset().y),
        ScrollbarAxis::Horizontal => (target.max_offset().width, target.offset().x),
    };

    let viewport_len = match style.axis {
        ScrollbarAxis::Vertical => viewport.size.height,
        ScrollbarAxis::Horizontal => viewport.size.width,
    };

    let scroll_pos = clamp_scroll_axis(raw_offset, max_offset);
    let thumb = compute_thumb(viewport_len, max_offset, scroll_pos, style.min_thumb_length)?;

    let thickness = style.thickness.min(match style.axis {
        ScrollbarAxis::Vertical => viewport.size.width,
        ScrollbarAxis::Horizontal => viewport.size.height,
    });

    let gutter_hitbox_thickness = (thickness + px(10.0)).max(px(18.0));
    let inset = style.inset;

    let (thumb_bounds, gutter_hitbox_bounds) = match style.axis {
        ScrollbarAxis::Vertical => {
            let right = viewport.origin.x + viewport.size.width - inset;
            let thumb_x = right - thickness;
            let thumb_bounds = Bounds::new(
                gpui::point(thumb_x, viewport.origin.y + thumb.offset),
                size(thickness, thumb.length),
            );
            let gutter_hitbox_bounds = Bounds::new(
                gpui::point(right - gutter_hitbox_thickness, viewport.origin.y),
                size(gutter_hitbox_thickness, viewport.size.height),
            );
            (thumb_bounds, gutter_hitbox_bounds)
        }
        ScrollbarAxis::Horizontal => {
            let bottom = viewport.origin.y + viewport.size.height - inset;
            let thumb_y = bottom - thickness;
            let thumb_bounds = Bounds::new(
                gpui::point(viewport.origin.x + thumb.offset, thumb_y),
                size(thumb.length, thickness),
            );
            let gutter_hitbox_bounds = Bounds::new(
                gpui::point(viewport.origin.x, bottom - gutter_hitbox_thickness),
                size(viewport.size.width, gutter_hitbox_thickness),
            );
            (thumb_bounds, gutter_hitbox_bounds)
        }
    };

    Some(ScrollbarGeometry {
        viewport,
        max_offset,
        scroll_pos,
        thumb,
        thickness,
        thumb_bounds,
        gutter_hitbox_bounds,
    })
}

#[derive(Debug)]
struct StyledScrollbarState {
    last_scroll_pos: Pixels,
    last_interaction_at: Option<Instant>,
    opacity: TransitionMap<()>,
}

impl Default for StyledScrollbarState {
    fn default() -> Self {
        Self {
            last_scroll_pos: px(0.0),
            last_interaction_at: None,
            opacity: TransitionMap::new(),
        }
    }
}

/// Paints a minimal, overlay scrollbar thumb for a scrollable element.
///
/// This wrapper inserts a narrow hitbox near the scroll thumb so we can reveal the scrollbar on
/// hover and support dragging.
pub struct StyledScrollbar {
    id: ElementId,
    target: ScrollbarTarget,
    style: ScrollbarStyle,
    thumb_color: Option<gpui::Hsla>,
    child: AnyElement,
}

impl StyledScrollbar {
    pub fn for_scroll_handle(
        id: impl Into<ElementId>,
        handle: ScrollHandle,
        child: impl gpui::IntoElement,
    ) -> Self {
        Self::new(id, ScrollbarTarget::ScrollHandle(handle), child)
    }

    pub fn for_list_state(
        id: impl Into<ElementId>,
        state: ListState,
        child: impl gpui::IntoElement,
    ) -> Self {
        Self::new(id, ScrollbarTarget::ListState(state), child)
    }

    pub fn new(
        id: impl Into<ElementId>,
        target: ScrollbarTarget,
        child: impl gpui::IntoElement,
    ) -> Self {
        Self {
            id: id.into(),
            target,
            style: ScrollbarStyle::default(),
            thumb_color: None,
            child: child.into_any_element(),
        }
    }

    pub fn style(mut self, style: ScrollbarStyle) -> Self {
        self.style = style;
        self
    }

    pub fn axis(mut self, axis: ScrollbarAxis) -> Self {
        self.style.axis = axis;
        self
    }

    pub fn thumb_color(mut self, color: gpui::Hsla) -> Self {
        self.thumb_color = Some(color);
        self
    }
}

impl gpui::IntoElement for StyledScrollbar {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for StyledScrollbar {
    type RequestLayoutState = ();
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        (self.child.request_layout(window, cx), ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Self::PrepaintState {
        self.child.prepaint(window, cx);

        compute_geometry(&self.target, self.style)
            .map(|geometry| window.insert_hitbox(geometry.gutter_hitbox_bounds, HitboxBehavior::Normal))
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _layout_state: &mut Self::RequestLayoutState,
        prepaint_state: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.paint(window, cx);

        let target = self.target.clone();
        let style = self.style;
        let Some(geometry) = compute_geometry(&target, style) else {
            return;
        };

        let hovered = prepaint_state
            .as_ref()
            .is_some_and(|hitbox| hitbox.is_hovered(window));

        let current_view = window.current_view();
        let drag_key = ScrollbarDragKey {
            view_id: current_view,
            element_id: self.id.clone(),
        };
        let is_dragging = {
            let global = cx.default_global::<ScrollbarDragGlobal>();
            global
                .active
                .as_ref()
                .is_some_and(|drag| drag.key == drag_key)
        };

        window.with_element_state::<StyledScrollbarState, _>(
            global_id.expect("StyledScrollbar should always have an id"),
            |state, window| {
                let mut state = state.unwrap_or_default();

                let scroll_changed = (geometry.scroll_pos - state.last_scroll_pos).abs() > px(0.5);
                if scroll_changed {
                    state.last_scroll_pos = geometry.scroll_pos;
                    state.last_interaction_at = Some(Instant::now());
                }

                let now = Instant::now();
                let within_idle_window = state
                    .last_interaction_at
                    .is_some_and(|at| now.duration_since(at) <= style.idle_delay);
                let visible = hovered || is_dragging || within_idle_window;

                // Keep ticking while we’re within the idle window so the fade-out can start promptly.
                if within_idle_window && !scroll_changed {
                    window.request_animation_frame();
                }

                let opacity = state
                    .opacity
                    .opacity_for_render((), visible, style.fade_duration, window);

                if opacity > 1e-3 {
                    let theme = theme_for_window(window, cx);
                    let base_thumb = self.thumb_color.unwrap_or_else(|| {
                        let factor = match theme.mode {
                            crate::styles::ThemeMode::Dark => 0.55,
                            crate::styles::ThemeMode::Light => 0.35,
                        };
                        theme.colors.foreground_muted.opacity(factor)
                    });

                    let thumb_bounds = geometry.thumb_bounds;

                    let radius = (geometry.thickness * 0.5).max(px(0.0));
                    let corners: Corners<Pixels> = Corners {
                        top_left: radius,
                        top_right: radius,
                        bottom_left: radius,
                        bottom_right: radius,
                    };

                    window.paint_quad(
                        fill(thumb_bounds, base_thumb.opacity(opacity)).corner_radii(corners),
                    );
                }

                ((), state)
            },
        );

        let Some(hitbox) = prepaint_state.clone() else {
            return;
        };

        let hitbox_for_mouse_move = hitbox.clone();
        let hitbox_for_mouse_down = hitbox.clone();

        let target_for_events = target.clone();
        let style_for_events = style;
        let drag_key_for_events = drag_key.clone();
        let was_hovered = hovered;

        window.on_mouse_event(move |event: &MouseMoveEvent, phase, window, cx| {
            if phase == DispatchPhase::Capture {
                let hovered = hitbox_for_mouse_move.is_hovered(window);
                if hovered != was_hovered {
                    cx.notify(current_view);
                }
            }

            if phase != DispatchPhase::Bubble {
                return;
            }

            let drag_offset = {
                let global = cx.default_global::<ScrollbarDragGlobal>();
                global
                    .active
                    .as_ref()
                    .filter(|drag| drag.key == drag_key_for_events)
                    .map(|drag| drag.drag_offset_in_thumb)
            };

            let Some(drag_offset_in_thumb) = drag_offset else {
                return;
            };

            let Some(geometry) = compute_geometry(&target_for_events, style_for_events) else {
                return;
            };

            let pointer = event.position;
            let (track_origin, pointer_pos) = match style_for_events.axis {
                ScrollbarAxis::Vertical => (geometry.viewport.origin.y, pointer.y),
                ScrollbarAxis::Horizontal => (geometry.viewport.origin.x, pointer.x),
            };

            let thumb_len = geometry.thumb.length;
            let track_len = (match style_for_events.axis {
                ScrollbarAxis::Vertical => geometry.viewport.size.height,
                ScrollbarAxis::Horizontal => geometry.viewport.size.width,
            } - thumb_len)
                .max(px(0.0));
            if track_len <= px(0.5) {
                return;
            }

            let thumb_top = (pointer_pos - drag_offset_in_thumb - track_origin).clamp(px(0.0), track_len);
            let t = (thumb_top / track_len).clamp(0.0, 1.0);
            let scroll_pos = geometry.max_offset * t;
            let raw_offset = -scroll_pos;

            match style_for_events.axis {
                ScrollbarAxis::Vertical => {
                    let offset = target_for_events.offset();
                    target_for_events.set_offset_from_scrollbar(gpui::point(offset.x, raw_offset));
                }
                ScrollbarAxis::Horizontal => {
                    let offset = target_for_events.offset();
                    target_for_events.set_offset_from_scrollbar(gpui::point(raw_offset, offset.y));
                }
            }

            cx.notify(current_view);
            cx.stop_propagation();
            window.prevent_default();
        });

        let target_for_mouse_down = target.clone();
        let style_for_mouse_down = style;
        let drag_key_for_mouse_down = drag_key.clone();
        window.on_mouse_event(move |event: &MouseDownEvent, phase, window, cx| {
            if phase != DispatchPhase::Bubble
                || event.button != MouseButton::Left
                || !hitbox_for_mouse_down.is_hovered(window)
            {
                return;
            }

            let Some(geometry) = compute_geometry(&target_for_mouse_down, style_for_mouse_down) else {
                return;
            };

            target_for_mouse_down.scrollbar_drag_started();

            let in_thumb = geometry.thumb_bounds.contains(&event.position);
            let drag_offset_in_thumb = match style_for_mouse_down.axis {
                ScrollbarAxis::Vertical => {
                    if in_thumb {
                        event.position.y - geometry.thumb_bounds.origin.y
                    } else {
                        geometry.thumb.length * 0.5
                    }
                }
                ScrollbarAxis::Horizontal => {
                    if in_thumb {
                        event.position.x - geometry.thumb_bounds.origin.x
                    } else {
                        geometry.thumb.length * 0.5
                    }
                }
            };

            {
                let global = cx.default_global::<ScrollbarDragGlobal>();
                global.active = Some(ActiveScrollbarDrag {
                    key: drag_key_for_mouse_down.clone(),
                    drag_offset_in_thumb,
                });
            }

            if !in_thumb {
                // Clicking on the track should immediately jump to that scroll position.
                let pointer = event.position;
                let (track_origin, pointer_pos) = match style_for_mouse_down.axis {
                    ScrollbarAxis::Vertical => (geometry.viewport.origin.y, pointer.y),
                    ScrollbarAxis::Horizontal => (geometry.viewport.origin.x, pointer.x),
                };

                let thumb_len = geometry.thumb.length;
                let track_len = (match style_for_mouse_down.axis {
                    ScrollbarAxis::Vertical => geometry.viewport.size.height,
                    ScrollbarAxis::Horizontal => geometry.viewport.size.width,
                } - thumb_len)
                    .max(px(0.0));

                if track_len > px(0.5) {
                    let thumb_top = (pointer_pos - drag_offset_in_thumb - track_origin)
                        .clamp(px(0.0), track_len);
                    let t = (thumb_top / track_len).clamp(0.0, 1.0);
                    let scroll_pos = geometry.max_offset * t;
                    let raw_offset = -scroll_pos;

                    match style_for_mouse_down.axis {
                        ScrollbarAxis::Vertical => {
                            let offset = target_for_mouse_down.offset();
                            target_for_mouse_down
                                .set_offset_from_scrollbar(gpui::point(offset.x, raw_offset));
                        }
                        ScrollbarAxis::Horizontal => {
                            let offset = target_for_mouse_down.offset();
                            target_for_mouse_down
                                .set_offset_from_scrollbar(gpui::point(raw_offset, offset.y));
                        }
                    }
                }

                cx.notify(current_view);
            }

            cx.stop_propagation();
            window.prevent_default();
        });

        let target_for_mouse_up = target.clone();
        let drag_key_for_mouse_up = drag_key.clone();
        window.on_mouse_event(move |event: &MouseUpEvent, phase, _window, cx| {
            if phase != DispatchPhase::Bubble || event.button != MouseButton::Left {
                return;
            }

            let should_end = {
                let global = cx.default_global::<ScrollbarDragGlobal>();
                global
                    .active
                    .as_ref()
                    .is_some_and(|drag| drag.key == drag_key_for_mouse_up)
            };

            if !should_end {
                return;
            }

            target_for_mouse_up.scrollbar_drag_ended();
            {
                let global = cx.default_global::<ScrollbarDragGlobal>();
                global.active = None;
            }
            cx.notify(current_view);
            cx.stop_propagation();
        });
    }
}

#[derive(Default)]
struct ScrollbarDragGlobal {
    active: Option<ActiveScrollbarDrag>,
}

impl gpui::Global for ScrollbarDragGlobal {}

#[derive(Debug)]
struct ActiveScrollbarDrag {
    key: ScrollbarDragKey,
    drag_offset_in_thumb: Pixels,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScrollbarDragKey {
    view_id: EntityId,
    element_id: ElementId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_thumb_is_none_when_not_scrollable() {
        assert_eq!(
            compute_thumb(px(100.0), px(0.0), px(0.0), px(24.0)),
            None
        );
    }

    #[test]
    fn compute_thumb_clamps_to_min_length() {
        let viewport = px(100.0);
        let max_offset = px(900.0);
        let thumb = compute_thumb(viewport, max_offset, px(0.0), px(30.0)).unwrap();
        assert!(thumb.length >= px(30.0));
    }

    #[test]
    fn compute_thumb_moves_from_top_to_bottom() {
        let viewport = px(120.0);
        let max_offset = px(240.0);

        let top = compute_thumb(viewport, max_offset, px(0.0), px(24.0)).unwrap();
        let bottom = compute_thumb(viewport, max_offset, max_offset, px(24.0)).unwrap();

        assert_eq!(top.offset, px(0.0));
        assert!((bottom.offset - (viewport - bottom.length)).abs() < px(0.5));
    }
}
