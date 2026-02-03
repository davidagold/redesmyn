use gpui::{
    AnyElement, App, Bounds, Corners, Element, ElementId, GlobalElementId, InspectorElementId,
    LayoutId, Pixels, ScrollHandle, Window, fill, linear_color_stop, linear_gradient, point, px,
    size,
};

use crate::utils::theme_for_window;

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

        let offset_y = self.scroll_handle.offset().y.max(px(0.0)).min(max_offset_y);
        let top_alpha = (offset_y / fade_height).clamp(0.0, 1.0);
        let bottom_alpha = ((max_offset_y - offset_y) / fade_height).clamp(0.0, 1.0);

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

        if top_alpha > 1e-3 {
            let gradient = linear_gradient(
                180.0,
                linear_color_stop(bg.opacity(top_alpha), 0.0),
                linear_color_stop(bg.opacity(0.0), 1.0),
            )
            .color_space(gpui::ColorSpace::Oklab);
            window.paint_quad(fill(top_bounds, gradient).corner_radii(top_corners));
        }

        if bottom_alpha > 1e-3 {
            let gradient = linear_gradient(
                0.0,
                linear_color_stop(bg.opacity(bottom_alpha), 0.0),
                linear_color_stop(bg.opacity(0.0), 1.0),
            )
            .color_space(gpui::ColorSpace::Oklab);
            window.paint_quad(fill(bottom_bounds, gradient).corner_radii(bottom_corners));
        }
    }
}
