use gpui::{
    AnyElement, App, Bounds, Corners, Element, ElementId, GlobalElementId, InspectorElementId,
    LayoutId, Pixels, ScrollHandle, Window, fill, linear_color_stop, linear_gradient, point, px,
    size,
};

use crate::utils::theme_for_window;

fn smoothstep(t: f32) -> f32 {
    // 3t^2 - 2t^3
    t * t * (3.0 - 2.0 * t)
}

fn clamp_scroll_y(offset_y: Pixels, max_offset_y: Pixels) -> Pixels {
    // `ScrollHandle::offset().y` is negative when scrolled down (i.e., the content is translated
    // upwards). Map to a positive "scroll position" in [0, max_offset].
    (-offset_y).max(px(0.0)).min(max_offset_y)
}

fn fade_stop(distance: Pixels, max_offset_y: Pixels, edge_threshold: Pixels) -> f32 {
    if distance <= edge_threshold || max_offset_y <= edge_threshold {
        return 0.0;
    }

    // Increase fade thickness as you move away from an edge. We bias early so the fade becomes
    // noticeable quickly, while still varying smoothly over large scroll ranges.
    let progress = (distance / max_offset_y).clamp(0.0, 1.0);
    smoothstep(progress.sqrt())
}

fn fade_stops(offset_y: Pixels, max_offset_y: Pixels, edge_threshold: Pixels) -> (f32, f32) {
    let scroll_y = clamp_scroll_y(offset_y, max_offset_y);
    let distance_to_top = scroll_y;
    let distance_to_bottom = (max_offset_y - scroll_y).max(px(0.0));
    (
        fade_stop(distance_to_top, max_offset_y, edge_threshold),
        fade_stop(distance_to_bottom, max_offset_y, edge_threshold),
    )
}

/// Paints top/bottom fades over a scrollable element.
///
/// This wrapper is *paint-only*: it does not add overlay elements, so pointer/wheel hit-testing is
/// unaffected.
///
/// Implementation detail: we vary the gradient stop (rather than alpha) so that the content-edge
/// "seam" is always covered by an opaque background, while the fade thickness ramps smoothly.
pub struct ScrollFade {
    scroll_handle: ScrollHandle,
    child: AnyElement,
    fade_height: Pixels,
    background_color: Option<gpui::Hsla>,
    corner_radius: Pixels,
}

impl ScrollFade {
    pub fn new(scroll_handle: ScrollHandle, child: impl gpui::IntoElement) -> Self {
        Self {
            scroll_handle,
            child: child.into_any_element(),
            fade_height: px(16.0),
            background_color: None,
            corner_radius: px(0.0),
        }
    }

    pub fn fade_height(mut self, height: Pixels) -> Self {
        self.fade_height = height.max(px(0.0));
        self
    }

    pub fn bg(mut self, color: gpui::Hsla) -> Self {
        self.background_color = Some(color);
        self
    }

    pub fn corner_radius(mut self, radius: Pixels) -> Self {
        self.corner_radius = radius.max(px(0.0));
        self
    }
}

impl gpui::IntoElement for ScrollFade {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for ScrollFade {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
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
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _layout_state: &mut Self::RequestLayoutState,
        _prepaint_state: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.paint(window, cx);

        let fade_height = self.fade_height;
        if fade_height <= px(0.0) {
            return;
        }

        let viewport = self.scroll_handle.bounds();
        let max_offset_y = self.scroll_handle.max_offset().height;
        if max_offset_y <= px(0.5) || viewport.size.height <= px(1.0) {
            return;
        }

        let fade_height = fade_height.min(viewport.size.height * 0.5);
        if fade_height <= px(0.0) {
            return;
        }

        let edge_threshold = px(1.0);
        let (top_stop, bottom_stop) =
            fade_stops(self.scroll_handle.offset().y, max_offset_y, edge_threshold);

        let theme = theme_for_window(window, cx);
        let bg = self.background_color.unwrap_or(theme.colors.surface);

        let top_bounds = Bounds::new(viewport.origin, size(viewport.size.width, fade_height));
        let bottom_bounds = Bounds::new(
            point(
                viewport.origin.x,
                viewport.origin.y + viewport.size.height - fade_height,
            ),
            size(viewport.size.width, fade_height),
        );

        let radius = self.corner_radius;
        let top_corners: Corners<Pixels> = Corners {
            top_left: radius,
            top_right: radius,
            bottom_left: px(0.0),
            bottom_right: px(0.0),
        };
        let bottom_corners: Corners<Pixels> = Corners {
            top_left: px(0.0),
            top_right: px(0.0),
            bottom_left: radius,
            bottom_right: radius,
        };

        if top_stop > 1e-3 {
            let gradient = linear_gradient(
                180.0,
                linear_color_stop(bg, 0.0),
                linear_color_stop(bg.opacity(0.0), top_stop),
            )
            .color_space(gpui::ColorSpace::Oklab);
            window.paint_quad(fill(top_bounds, gradient).corner_radii(top_corners));
        }

        if bottom_stop > 1e-3 {
            let gradient = linear_gradient(
                0.0,
                linear_color_stop(bg, 0.0),
                linear_color_stop(bg.opacity(0.0), bottom_stop),
            )
            .color_space(gpui::ColorSpace::Oklab);
            window.paint_quad(fill(bottom_bounds, gradient).corner_radii(bottom_corners));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fade_stops_are_zero_at_scroll_limits() {
        let max = px(100.0);
        let edge = px(1.0);

        // Top edge: no top fade, bottom fade present.
        let (top, bottom) = fade_stops(px(0.0), max, edge);
        assert_eq!(top, 0.0);
        assert!(bottom > 0.9);

        // Bottom edge: no bottom fade, top fade present.
        let (top, bottom) = fade_stops(-max, max, edge);
        assert_eq!(bottom, 0.0);
        assert!(top > 0.9);
    }

    #[test]
    fn fade_stops_vary_smoothly_between_edges() {
        let max = px(200.0);
        let edge = px(1.0);

        let (top, bottom) = fade_stops(-px(100.0), max, edge);
        assert!(top > 0.0 && top < 1.0);
        assert!(bottom > 0.0 && bottom < 1.0);
        assert!((top - bottom).abs() < 1e-3);
    }

    #[test]
    fn fade_stops_clamp_scroll_sign() {
        let max = px(100.0);
        let edge = px(1.0);

        // Positive offsets should clamp to the top edge.
        let (top, bottom) = fade_stops(px(12.0), max, edge);
        assert_eq!(top, 0.0);
        assert!(bottom > 0.9);
    }
}
