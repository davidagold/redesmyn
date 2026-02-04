use std::time::{Duration, Instant};

use gpui::{
    AnyElement, App, Bounds, Corners, Element, ElementId, GlobalElementId, InspectorElementId,
    LayoutId, ListState, Pixels, Point, ScrollHandle, Size, Window, fill, px, size,
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
}

#[derive(Debug, Clone, Copy)]
pub struct ScrollbarStyle {
    pub axis: ScrollbarAxis,
    pub thickness: Pixels,
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
/// This wrapper is *paint-only*: it does not add overlay elements, so pointer/wheel hit-testing is
/// unaffected.
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
    type PrepaintState = ();

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
    ) {
        self.child.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _layout_state: &mut Self::RequestLayoutState,
        _prepaint_state: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.paint(window, cx);

        let target = &self.target;
        let style = self.style;
        let viewport = target.bounds();

        if viewport.size.width <= px(1.0) || viewport.size.height <= px(1.0) {
            return;
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
        let Some(thumb) = compute_thumb(viewport_len, max_offset, scroll_pos, style.min_thumb_length)
        else {
            return;
        };

        window.with_element_state::<StyledScrollbarState, _>(
            global_id.expect("StyledScrollbar should always have an id"),
            |state, window| {
                let mut state = state.unwrap_or_default();

                let scroll_changed = (scroll_pos - state.last_scroll_pos).abs() > px(0.5);
                if scroll_changed {
                    state.last_scroll_pos = scroll_pos;
                    state.last_interaction_at = Some(Instant::now());
                }

                let now = Instant::now();
                let visible = state
                    .last_interaction_at
                    .is_some_and(|at| now.duration_since(at) <= style.idle_delay);

                // Keep ticking while we’re within the idle window so the fade-out can start promptly.
                if visible && !scroll_changed {
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

                    let thickness = style.thickness.min(match style.axis {
                        ScrollbarAxis::Vertical => viewport.size.width,
                        ScrollbarAxis::Horizontal => viewport.size.height,
                    });

                    let inset = style.inset.max(px(0.0));
                    let thumb_bounds = match style.axis {
                        ScrollbarAxis::Vertical => Bounds::new(
                            gpui::point(
                                viewport.origin.x + viewport.size.width - inset - thickness,
                                viewport.origin.y + thumb.offset,
                            ),
                            size(thickness, thumb.length),
                        ),
                        ScrollbarAxis::Horizontal => Bounds::new(
                            gpui::point(
                                viewport.origin.x + thumb.offset,
                                viewport.origin.y + viewport.size.height - inset - thickness,
                            ),
                            size(thumb.length, thickness),
                        ),
                    };

                    let radius = (thickness * 0.5).max(px(0.0));
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
    }
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
