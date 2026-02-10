use gpui::{
    AnyElement, App, ClickEvent, ElementId, RenderOnce, SharedString, Window, div, prelude::*, px,
};

use crate::components::Tooltip;
use crate::utils::theme_for_window;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    Primary,
    Secondary,
    Ghost,
    GhostHover,
    Danger,
}

impl Default for ButtonKind {
    fn default() -> Self {
        Self::Secondary
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextButtonLayout {
    Default,
    MenuItem,
}

impl Default for TextButtonLayout {
    fn default() -> Self {
        Self::Default
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextButtonSize {
    Regular,
    /// Compact padding but regular text size.
    Compact,
    Small,
}

impl Default for TextButtonSize {
    fn default() -> Self {
        Self::Regular
    }
}

#[derive(IntoElement)]
pub struct TextButton {
    id: ElementId,
    label: SharedString,
    kind: ButtonKind,
    layout: TextButtonLayout,
    size: TextButtonSize,
    disabled: bool,
    disabled_reason: Option<SharedString>,
    tooltip: Option<SharedString>,
    on_click: Option<Box<dyn Fn(&ClickEvent, &mut Window, &mut App)>>,
    trailing: Option<AnyElement>,
}

impl TextButton {
    pub fn new(id: impl Into<ElementId>, label: impl Into<SharedString>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            kind: ButtonKind::default(),
            layout: TextButtonLayout::default(),
            size: TextButtonSize::default(),
            disabled: false,
            disabled_reason: None,
            tooltip: None,
            on_click: None,
            trailing: None,
        }
    }

    pub fn kind(mut self, kind: ButtonKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn disabled_reason(mut self, reason: impl Into<SharedString>) -> Self {
        self.disabled_reason = Some(reason.into());
        self
    }

    pub fn tooltip(mut self, text: impl Into<SharedString>) -> Self {
        self.tooltip = Some(text.into());
        self
    }

    pub fn on_click(
        mut self,
        handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_click = Some(Box::new(handler));
        self
    }

    pub fn trailing(mut self, element: impl IntoElement) -> Self {
        self.trailing = Some(element.into_any_element());
        self
    }

    pub fn menu_item(mut self) -> Self {
        self.layout = TextButtonLayout::MenuItem;
        self.size = TextButtonSize::Small;
        self
    }

    pub fn compact(mut self) -> Self {
        self.size = TextButtonSize::Compact;
        self
    }

    pub fn small(mut self) -> Self {
        self.size = TextButtonSize::Small;
        self
    }
}

impl RenderOnce for TextButton {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let (bg, fg, border) = match self.kind {
            ButtonKind::Primary => (
                Some(theme.colors.ring),
                theme.colors.background,
                Some(theme.colors.ring),
            ),
            ButtonKind::Secondary => (
                Some(theme.colors.accent),
                theme.colors.foreground,
                Some(theme.colors.border),
            ),
            ButtonKind::Ghost => (None, theme.colors.foreground, None),
            ButtonKind::GhostHover => (
                None,
                theme.colors.foreground,
                Some(theme.colors.border.opacity(0.0)),
            ),
            ButtonKind::Danger => (
                Some(theme.colors.danger),
                theme.colors.background,
                Some(theme.colors.danger),
            ),
        };

        let (padding_x, padding_y, radius, gap) = match self.size {
            TextButtonSize::Regular => (
                theme.spacing.md,
                theme.spacing.sm,
                theme.radius.md,
                theme.spacing.sm,
            ),
            TextButtonSize::Compact | TextButtonSize::Small => (
                theme.spacing.sm,
                theme.spacing.xs,
                theme.radius.sm,
                theme.spacing.xs,
            ),
        };

        let mut button = div()
            .id(self.id)
            .flex()
            .flex_row()
            .gap(gap)
            .items_center()
            .px(padding_x)
            .py(padding_y)
            .rounded(radius)
            .text_color(fg)
            .when_some(bg, |this, bg| this.bg(bg))
            .when_some(border, |this, border| this.border_1().border_color(border))
            .cursor_pointer()
            .focusable()
            .focus(|mut style| {
                style.border_color = Some(theme.colors.ring);
                style
            });

        match self.layout {
            TextButtonLayout::Default => {
                button = button.justify_center();
                if !self.disabled && self.kind == ButtonKind::GhostHover {
                    button = button
                        .hover(|this| this.bg(theme.colors.accent.opacity(0.7)))
                        .active(|this| this.bg(theme.colors.accent.opacity(0.82)));
                }
            }
            TextButtonLayout::MenuItem => {
                button = button.justify_start().w_full();
                if !self.disabled {
                    button = button.hover(|this| this.bg(theme.colors.accent));
                }
            }
        }

        button = match self.size {
            TextButtonSize::Regular | TextButtonSize::Compact => button.text_sm(),
            TextButtonSize::Small => button.text_xs(),
        };

        if let Some(tooltip) = tooltip_text(self.disabled, &self.disabled_reason, &self.tooltip) {
            button = button.tooltip(move |_, cx| cx.new(|_| Tooltip::new(tooltip.clone())).into());
        }

        if self.disabled {
            button = button
                .opacity(0.55)
                .cursor_not_allowed()
                .border_color(theme.colors.border);
        } else if let Some(on_click) = self.on_click {
            button = button.on_click(move |event, window, cx| {
                on_click(event, window, cx);
            });
        }

        button = match self.layout {
            TextButtonLayout::Default => button.child(self.label),
            TextButtonLayout::MenuItem => button.child(
                div()
                    .flex_1()
                    .min_w_0()
                    .text_left()
                    .truncate()
                    .child(self.label),
            ),
        };
        if let Some(trailing) = self.trailing {
            button = button.child(trailing);
        }

        button
    }
}

#[derive(IntoElement)]
pub struct IconButton {
    id: ElementId,
    icon: AnyElement,
    tooltip: Option<SharedString>,
    active: bool,
    disabled: bool,
    disabled_reason: Option<SharedString>,
    on_click: Option<Box<dyn Fn(&ClickEvent, &mut Window, &mut App)>>,
}

impl IconButton {
    pub fn new(id: impl Into<ElementId>, icon: impl IntoElement) -> Self {
        Self {
            id: id.into(),
            icon: icon.into_any_element(),
            tooltip: None,
            active: false,
            disabled: false,
            disabled_reason: None,
            on_click: None,
        }
    }

    pub fn tooltip(mut self, text: impl Into<SharedString>) -> Self {
        self.tooltip = Some(text.into());
        self
    }

    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn disabled_reason(mut self, reason: impl Into<SharedString>) -> Self {
        self.disabled_reason = Some(reason.into());
        self
    }

    pub fn on_click(
        mut self,
        handler: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_click = Some(Box::new(handler));
        self
    }
}

impl RenderOnce for IconButton {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let hover_bg = if self.active {
            theme.colors.accent.opacity(0.55)
        } else {
            theme.colors.accent
        };

        let mut button = div()
            .id(self.id)
            .flex()
            .items_center()
            .justify_center()
            .size(px(28.0))
            .rounded(theme.radius.md)
            .text_color(theme.colors.foreground)
            .cursor_pointer()
            .focusable()
            .when(self.active, |this| {
                this.bg(theme.colors.accent.opacity(0.35))
            })
            .hover(move |this| this.bg(hover_bg))
            .focus(|mut style| {
                style.border_color = Some(theme.colors.ring);
                style
            })
            .child(self.icon);

        if let Some(tooltip) = tooltip_text(self.disabled, &self.disabled_reason, &self.tooltip) {
            button = button.tooltip(move |_, cx| cx.new(|_| Tooltip::new(tooltip.clone())).into());
        }

        if self.disabled {
            button = button.opacity(0.55).cursor_not_allowed();
        } else if let Some(on_click) = self.on_click {
            button = button.on_click(move |event, window, cx| {
                on_click(event, window, cx);
            });
        }

        button
    }
}

fn tooltip_text(
    disabled: bool,
    disabled_reason: &Option<SharedString>,
    tooltip: &Option<SharedString>,
) -> Option<SharedString> {
    if disabled {
        disabled_reason.clone().or_else(|| tooltip.clone())
    } else {
        tooltip.clone()
    }
}
