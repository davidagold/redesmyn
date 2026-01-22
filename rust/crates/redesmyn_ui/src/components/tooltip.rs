use gpui::{Context, Render, SharedString, Window, div, prelude::*, px};

use crate::utils::theme_for_window;

pub struct Tooltip {
    text: SharedString,
}

impl Tooltip {
    pub fn new(text: impl Into<SharedString>) -> Self {
        Self { text: text.into() }
    }
}

impl Render for Tooltip {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        div()
            .max_w(px(360.0))
            .px(theme.spacing.sm)
            .py(theme.spacing.xs)
            .bg(theme.colors.surface_elevated)
            .border_1()
            .border_color(theme.colors.border)
            .rounded(theme.radius.md)
            .shadow_md()
            .text_sm()
            .text_color(theme.colors.foreground)
            .child(self.text.clone())
    }
}
