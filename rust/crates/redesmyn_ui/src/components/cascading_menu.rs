use gpui::{AnyElement, Pixels, RenderOnce, Window, div, prelude::*, px};

/// Layout metrics for a two-level cascading menu (primary + optional secondary menu).
#[derive(Debug, Clone, Copy)]
pub struct CascadingMenuMetrics {
    pub primary_width: Pixels,
    pub secondary_width: Pixels,
    /// Horizontal overlap between the primary and secondary menus.
    pub overlap: Pixels,
}

impl CascadingMenuMetrics {
    pub fn filter_defaults() -> Self {
        Self {
            primary_width: px(260.0),
            secondary_width: px(240.0),
            overlap: px(6.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadingMenuSecondarySide {
    Left,
    Right,
}

struct CascadingMenuSecondary {
    top: Pixels,
    element: AnyElement,
}

/// Simple, reusable layout wrapper for a two-level cascading menu.
///
/// This component is intentionally layout-only: callers build the contents of the primary and
/// secondary menus (including any event handlers) and this component positions them with the
/// correct overlap/alignment.
#[derive(IntoElement)]
pub struct CascadingMenu {
    primary: AnyElement,
    secondary: Option<CascadingMenuSecondary>,
    metrics: CascadingMenuMetrics,
    secondary_side: CascadingMenuSecondarySide,
}

impl CascadingMenu {
    pub fn new(primary: impl IntoElement) -> Self {
        Self {
            primary: primary.into_any_element(),
            secondary: None,
            metrics: CascadingMenuMetrics::filter_defaults(),
            secondary_side: CascadingMenuSecondarySide::Right,
        }
    }

    pub fn metrics(mut self, metrics: CascadingMenuMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn secondary_side(mut self, secondary_side: CascadingMenuSecondarySide) -> Self {
        self.secondary_side = secondary_side;
        self
    }

    pub fn secondary(mut self, top: Pixels, element: impl IntoElement) -> Self {
        self.secondary = Some(CascadingMenuSecondary {
            top,
            element: element.into_any_element(),
        });
        self
    }

    pub fn maybe_secondary<E: IntoElement>(
        mut self,
        top: Option<Pixels>,
        element: Option<E>,
    ) -> Self {
        if let (Some(top), Some(element)) = (top, element) {
            self.secondary = Some(CascadingMenuSecondary {
                top,
                element: element.into_any_element(),
            });
        }
        self
    }
}

impl RenderOnce for CascadingMenu {
    fn render(self, _window: &mut Window, _cx: &mut gpui::App) -> impl IntoElement {
        let metrics = self.metrics;
        let secondary_side = self.secondary_side;

        div()
            .relative()
            .child(div().w(metrics.primary_width).child(self.primary))
            .when_some(self.secondary, move |this, secondary| {
                let left = match secondary_side {
                    CascadingMenuSecondarySide::Right => metrics.primary_width - metrics.overlap,
                    CascadingMenuSecondarySide::Left => {
                        -(metrics.secondary_width - metrics.overlap)
                    }
                };

                this.child(
                    div()
                        .absolute()
                        .top(secondary.top)
                        .left(left)
                        .w(metrics.secondary_width)
                        .child(secondary.element),
                )
            })
    }
}
