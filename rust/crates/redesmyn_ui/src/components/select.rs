use std::rc::Rc;

use gpui::{
    App, ClickEvent, ElementId, Pixels, RenderOnce, SharedString, Window, div, prelude::*, px,
};

use crate::components::{ButtonKind, TextButton, Tooltip};
use crate::utils::theme_for_window;

#[derive(Debug, Clone)]
pub struct SelectOption<T> {
    id: ElementId,
    label: SharedString,
    tooltip: Option<SharedString>,
    disabled: bool,
    disabled_reason: Option<SharedString>,
    value: T,
}

impl<T> SelectOption<T> {
    pub fn new(id: impl Into<ElementId>, label: impl Into<SharedString>, value: T) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            tooltip: None,
            disabled: false,
            disabled_reason: None,
            value,
        }
    }

    pub fn tooltip(mut self, text: impl Into<SharedString>) -> Self {
        self.tooltip = Some(text.into());
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
}

type OpenChangeHandler = Rc<dyn Fn(bool, &mut Window, &mut App)>;
type SelectHandler<T> = Rc<dyn Fn(T, &mut Window, &mut App)>;

#[derive(IntoElement)]
pub struct Select<T: Clone + PartialEq + 'static> {
    id: ElementId,
    options: Vec<SelectOption<T>>,
    value: Option<T>,
    open: bool,
    disabled: bool,
    disabled_reason: Option<SharedString>,
    tooltip: Option<SharedString>,
    placeholder: SharedString,
    menu_width: Pixels,
    menu_offset: Pixels,
    on_open_change: Option<OpenChangeHandler>,
    on_select: Option<SelectHandler<T>>,
}

impl<T: Clone + PartialEq + 'static> Select<T> {
    pub fn new(id: impl Into<ElementId>) -> Self {
        Self {
            id: id.into(),
            options: Vec::new(),
            value: None,
            open: false,
            disabled: false,
            disabled_reason: None,
            tooltip: None,
            placeholder: "Select".into(),
            menu_width: px(220.0),
            menu_offset: px(34.0),
            on_open_change: None,
            on_select: None,
        }
    }

    pub fn option(mut self, option: SelectOption<T>) -> Self {
        self.options.push(option);
        self
    }

    pub fn value(mut self, value: T) -> Self {
        self.value = Some(value);
        self
    }

    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
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

    pub fn placeholder(mut self, placeholder: impl Into<SharedString>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    pub fn menu_width(mut self, width: Pixels) -> Self {
        self.menu_width = width;
        self
    }

    pub fn menu_offset(mut self, offset: Pixels) -> Self {
        self.menu_offset = offset;
        self
    }

    pub fn on_open_change(
        mut self,
        handler: impl Fn(bool, &mut Window, &mut App) + 'static,
    ) -> Self {
        self.on_open_change = Some(Rc::new(handler));
        self
    }

    pub fn on_select(mut self, handler: impl Fn(T, &mut Window, &mut App) + 'static) -> Self {
        self.on_select = Some(Rc::new(handler));
        self
    }
}

impl<T: Clone + PartialEq + 'static> RenderOnce for Select<T> {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let open = self.open && !self.disabled;
        let chevron = if open { "▴" } else { "▾" };

        let selected_label = self
            .value
            .as_ref()
            .and_then(|value| {
                self.options
                    .iter()
                    .find(|option| option.value == *value)
                    .map(|option| option.label.clone())
            })
            .unwrap_or_else(|| self.placeholder.clone());

        let on_open_change = self.on_open_change.clone();

        let mut trigger = div()
            .id(self.id.clone())
            .flex()
            .flex_row()
            .items_center()
            .gap(theme.spacing.xs)
            .px(theme.spacing.sm)
            .py(theme.spacing.xs)
            .rounded(theme.radius.md)
            .bg(theme.colors.accent)
            .text_size(theme.typography.caption.size)
            .text_color(theme.colors.foreground)
            .cursor_pointer()
            .focusable()
            .child(selected_label)
            .child(
                div()
                    .flex_shrink_0()
                    .text_color(theme.colors.foreground_muted)
                    .child(chevron),
            )
            .focus(|mut style| {
                style.border_color = Some(theme.colors.ring);
                style
            });

        if let Some(tooltip) = tooltip_text(self.disabled, &self.disabled_reason, &self.tooltip) {
            trigger =
                trigger.tooltip(move |_, cx| cx.new(|_| Tooltip::new(tooltip.clone())).into());
        }

        if self.disabled {
            trigger = trigger.opacity(0.55).cursor_not_allowed();
        } else if let Some(on_open_change) = on_open_change.clone() {
            trigger = trigger.on_click(move |event: &ClickEvent, window, cx| {
                if event.standard_click() {
                    on_open_change(!open, window, cx);
                }
                cx.stop_propagation();
            });
        }

        let on_select = self.on_select.clone();
        let menu_width = self.menu_width;
        let menu_offset = self.menu_offset;
        let value = self.value.clone();

        let mut menu = div()
            .absolute()
            .left(px(0.0))
            .bottom(menu_offset)
            .w(menu_width)
            .rounded(theme.radius.md)
            .shadow_md()
            .occlude()
            .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
            .child(
                div()
                    .w_full()
                    .p(theme.spacing.xs)
                    .rounded(theme.radius.md)
                    .bg(theme.colors.surface)
                    .border_1()
                    .border_color(theme.colors.border.opacity(0.5))
                    .overflow_hidden()
                    .flex()
                    .flex_col()
                    .gap(theme.spacing.xs)
                    .children(self.options.into_iter().map(|option| {
                        let selected = value.as_ref().is_some_and(|value| option.value == *value);
                        let kind = if selected {
                            ButtonKind::Secondary
                        } else {
                            ButtonKind::Ghost
                        };

                        let on_open_change = on_open_change.clone();
                        let on_select = on_select.clone();
                        let option_value = option.value;

                        let mut item = TextButton::new(option.id, option.label)
                            .menu_item()
                            .kind(kind)
                            .disabled(option.disabled)
                            .on_click(move |event, window, cx| {
                                if event.standard_click() {
                                    if let Some(on_select) = on_select.clone() {
                                        on_select(option_value.clone(), window, cx);
                                    }
                                    if let Some(on_open_change) = on_open_change.clone() {
                                        on_open_change(false, window, cx);
                                    }
                                }
                                cx.stop_propagation();
                            });

                        if option.disabled {
                            if let Some(reason) = option.disabled_reason {
                                item = item.disabled_reason(reason);
                            }
                        }

                        if let Some(tooltip) = option.tooltip {
                            item = item.tooltip(tooltip);
                        }

                        item
                    })),
            );

        if open {
            if let Some(on_open_change) = on_open_change.clone() {
                menu = menu.on_mouse_down_out(move |_, window, cx| {
                    on_open_change(false, window, cx);
                });
            }
        }

        let mut container = div()
            .relative()
            .child(trigger)
            .when(open, |this| this.child(menu));

        if open {
            if let Some(on_open_change) = self.on_open_change {
                container = container.capture_key_down(move |event, window, cx| {
                    if event.keystroke.key != "escape" {
                        return;
                    }

                    on_open_change(false, window, cx);
                    cx.stop_propagation();
                });
            }
        }

        container
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
