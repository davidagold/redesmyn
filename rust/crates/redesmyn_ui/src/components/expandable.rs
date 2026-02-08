use gpui::{AnyElement, App, RenderOnce, Window, div, prelude::*};

#[derive(IntoElement)]
pub struct Expandable {
    child: AnyElement,
    opacity: f32,
    max_height: Option<gpui::Pixels>,
}

impl Expandable {
    pub fn new(child: impl IntoElement) -> Self {
        Self {
            child: child.into_any_element(),
            opacity: 1.0,
            max_height: None,
        }
    }

    pub fn opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity;
        self
    }

    pub fn max_height(mut self, max_height: gpui::Pixels) -> Self {
        self.max_height = Some(max_height);
        self
    }
}

impl RenderOnce for Expandable {
    fn render(self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let mut container = div().opacity(self.opacity);
        if let Some(max_height) = self.max_height {
            container = container.overflow_hidden().max_h(max_height);
        }
        container.child(self.child)
    }
}
