use gpui::{AnyElement, App, RenderOnce, SharedString, Window, div, prelude::*};

use crate::utils::theme_for_window;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalloutKind {
    Info,
    Warning,
    Danger,
}

impl Default for CalloutKind {
    fn default() -> Self {
        Self::Info
    }
}

#[derive(IntoElement)]
pub struct Callout {
    kind: CalloutKind,
    title: Option<SharedString>,
    message: SharedString,
    action: Option<AnyElement>,
}

impl Callout {
    pub fn new(message: impl Into<SharedString>) -> Self {
        Self {
            kind: CalloutKind::default(),
            title: None,
            message: message.into(),
            action: None,
        }
    }

    pub fn kind(mut self, kind: CalloutKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn title(mut self, title: impl Into<SharedString>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn action(mut self, element: impl IntoElement) -> Self {
        self.action = Some(element.into_any_element());
        self
    }
}

impl RenderOnce for Callout {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let (border, bg) = match self.kind {
            CalloutKind::Info => (theme.colors.border, theme.colors.surface_elevated),
            CalloutKind::Warning => (theme.colors.warning, theme.colors.surface_elevated),
            CalloutKind::Danger => (theme.colors.danger, theme.colors.surface_elevated),
        };

        let mut body = div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .flex_1()
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .when_some(self.title, |this, title| this.child(title)),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(self.message),
            );

        if let Some(action) = self.action {
            body = body.child(div().pt(theme.spacing.xs).child(action));
        }

        div()
            .flex()
            .gap(theme.spacing.md)
            .px(theme.spacing.md)
            .py(theme.spacing.sm)
            .bg(bg)
            .border_l_2()
            .border_color(border)
            .rounded(theme.radius.md)
            .child(body)
    }
}
