use gpui::{AnyElement, App, RenderOnce, SharedString, Window, div, prelude::*, px};

use crate::utils::theme_for_window;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadgeKind {
    Neutral,
    Info,
    Success,
    Warning,
    Danger,
    Completed,
}

impl Default for BadgeKind {
    fn default() -> Self {
        Self::Neutral
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadgeStyle {
    Subtle,
    Solid,
}

impl Default for BadgeStyle {
    fn default() -> Self {
        Self::Subtle
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadgeSize {
    Sm,
    Md,
}

impl Default for BadgeSize {
    fn default() -> Self {
        Self::Sm
    }
}

#[derive(IntoElement)]
pub struct Badge {
    label: SharedString,
    kind: BadgeKind,
    style: BadgeStyle,
    size: BadgeSize,
    leading_dot: bool,
    leading: Option<AnyElement>,
    trailing: Option<AnyElement>,
}

impl Badge {
    pub fn new(label: impl Into<SharedString>) -> Self {
        Self {
            label: label.into(),
            kind: BadgeKind::default(),
            style: BadgeStyle::default(),
            size: BadgeSize::default(),
            leading_dot: false,
            leading: None,
            trailing: None,
        }
    }

    pub fn kind(mut self, kind: BadgeKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn style(mut self, style: BadgeStyle) -> Self {
        self.style = style;
        self
    }

    pub fn size(mut self, size: BadgeSize) -> Self {
        self.size = size;
        self
    }

    pub fn leading_dot(mut self, enabled: bool) -> Self {
        self.leading_dot = enabled;
        self
    }

    pub fn leading(mut self, element: impl IntoElement) -> Self {
        self.leading = Some(element.into_any_element());
        self
    }

    pub fn trailing(mut self, element: impl IntoElement) -> Self {
        self.trailing = Some(element.into_any_element());
        self
    }
}

impl RenderOnce for Badge {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let tone = match self.kind {
            BadgeKind::Neutral => theme.colors.foreground_muted,
            BadgeKind::Info => theme.colors.info,
            BadgeKind::Success => theme.colors.success,
            BadgeKind::Warning => theme.colors.warning,
            BadgeKind::Danger => theme.colors.danger,
            BadgeKind::Completed => theme.colors.completed,
        };

        let (bg, border, fg) = match self.style {
            BadgeStyle::Subtle => (
                theme.colors.surface_elevated,
                theme.colors.border.opacity(0.8),
                match self.kind {
                    BadgeKind::Neutral => theme.colors.foreground_muted,
                    _ => theme.colors.foreground,
                },
            ),
            BadgeStyle::Solid => (
                tone.opacity(0.18),
                tone.opacity(0.4),
                match self.kind {
                    BadgeKind::Neutral => theme.colors.foreground_muted,
                    _ => theme.colors.foreground,
                },
            ),
        };

        let (padding_x, padding_y, icon_label_gap, gap) = match self.size {
            BadgeSize::Sm => (theme.spacing.sm, px(2.0), px(6.0), theme.spacing.xs),
            BadgeSize::Md => (
                theme.spacing.sm,
                theme.spacing.xs,
                px(6.0),
                theme.spacing.xs,
            ),
        };

        let mut badge = div()
            .flex()
            .flex_row()
            .items_center()
            .gap(gap)
            .px(padding_x)
            .py(padding_y)
            .rounded(px(999.0))
            .bg(bg)
            .border_1()
            .border_color(border);

        badge = match self.size {
            BadgeSize::Sm => badge.text_xs(),
            BadgeSize::Md => badge.text_sm(),
        }
        .text_color(fg);

        let mut content = div().flex().flex_row().items_center().gap(icon_label_gap);

        let mut has_icon = false;
        let mut icon_row = div().flex().flex_row().items_center().gap(theme.spacing.xs);

        if self.leading_dot {
            has_icon = true;
            icon_row = icon_row.child(div().size(px(6.0)).rounded(px(999.0)).bg(tone));
        }

        if let Some(leading) = self.leading {
            has_icon = true;
            icon_row = icon_row.child(leading);
        }

        if has_icon {
            content = content.child(icon_row);
        }
        content = content.child(self.label);
        badge = badge.child(content);

        if let Some(trailing) = self.trailing {
            badge = badge.child(trailing);
        }

        badge
    }
}
