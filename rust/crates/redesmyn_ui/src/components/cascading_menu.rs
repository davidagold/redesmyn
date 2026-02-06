use gpui::{AnyElement, Div, Global, Pixels, RenderOnce, Window, div, prelude::*, px};

use crate::styles::UiTheme;
use super::overlay_surface::{OverlaySurfaceKind, overlay_surface};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadingMenuId {
    TaskFilters,
    SessionSettings,
    SessionModel,
    SessionReasoning,
}

#[derive(Debug, Default)]
pub struct CascadingMenuState {
    open_menu: Option<CascadingMenuId>,
}

impl Global for CascadingMenuState {}

impl CascadingMenuState {
    pub fn open_menu(&self) -> Option<CascadingMenuId> {
        self.open_menu
    }

    pub fn set_open_menu(&mut self, menu: Option<CascadingMenuId>) {
        self.open_menu = menu;
    }
}

pub fn set_open_cascading_menu(menu: Option<CascadingMenuId>, cx: &mut impl BorrowAppContext) {
    cx.update_default_global::<CascadingMenuState, _>(|state, _cx| {
        state.set_open_menu(menu);
    });
}

/// Visual styling defaults shared across cascading menus.
#[derive(Debug, Clone, Copy)]
pub struct CascadingMenuRowStyle {
    pub height: Pixels,
    pub padding_x: Pixels,
    pub gap: Pixels,
    pub radius: Pixels,
    /// Background alpha used for hover/selection highlights.
    pub hover_bg_alpha: f32,
}

impl CascadingMenuRowStyle {
    #[must_use]
    pub fn compact(theme: &UiTheme) -> Self {
        Self {
            height: px(34.0),
            padding_x: theme.spacing.sm,
            gap: theme.spacing.xs,
            radius: theme.radius.sm,
            hover_bg_alpha: 0.08,
        }
    }
}

/// Visual styling defaults for cascading menu surfaces.
#[derive(Debug, Clone, Copy)]
pub struct CascadingMenuSurfaceStyle {
    pub padding_x: Pixels,
    pub padding_y: Pixels,
}

impl CascadingMenuSurfaceStyle {
    #[must_use]
    pub fn compact(theme: &UiTheme) -> Self {
        Self {
            padding_x: theme.spacing.xs,
            padding_y: theme.spacing.xs,
        }
    }
}

/// Returns a standardized menu row container that matches the filter menu's hover/selection
/// treatment (higher-contrast than `accent` on dark themes).
#[must_use]
pub fn cascading_menu_row(
    theme: &UiTheme,
    style: CascadingMenuRowStyle,
    keyboard_selected: bool,
    hover_opacity: f32,
) -> Div {
    let hover_opacity = hover_opacity.clamp(0.0, 1.0);
    let selected_bg = theme.colors.foreground.opacity(style.hover_bg_alpha);

    div()
        .flex()
        .flex_row()
        .w_full()
        .items_center()
        .gap(style.gap)
        .px(style.padding_x)
        .h(style.height)
        .rounded(style.radius)
        .text_xs()
        // Default the row's foreground to a muted color so icons can simply inherit it.
        // When highlighted, bump the row color for contrast against the hover background.
        .text_color(theme.colors.foreground_muted)
        .when(keyboard_selected, move |this| this.bg(selected_bg))
        .when(keyboard_selected, move |this| this.text_color(theme.colors.foreground))
        .when(hover_opacity > 1e-3, move |this| {
            this.bg(
                theme
                    .colors
                    .foreground
                    .opacity(style.hover_bg_alpha * hover_opacity),
            )
            .text_color(theme.colors.foreground)
        })
}

/// Standard checkbox indicator used for multi-select menu options.
#[must_use]
pub fn cascading_menu_checkbox_indicator(theme: &UiTheme, selected: bool, highlighted: bool) -> Div {
    let border = if highlighted && !selected {
        theme.colors.foreground_muted.opacity(0.65)
    } else {
        theme.colors.border.opacity(0.45)
    };

    div()
        .size(px(16.0))
        .flex()
        .items_center()
        .justify_center()
        .rounded(theme.radius.sm)
        .border_1()
        .border_color(border)
        .when(selected, |this| {
            this.border_color(theme.colors.ring.opacity(0.7))
                .text_color(theme.colors.ring)
        })
        .text_xs()
        .text_color(theme.colors.foreground_muted)
        .child(if selected { "✓" } else { "" })
}

/// Standard radio indicator used for single-select menu options.
#[must_use]
pub fn cascading_menu_radio_indicator(theme: &UiTheme, selected: bool, highlighted: bool) -> Div {
    let border = if highlighted && !selected {
        theme.colors.foreground_muted.opacity(0.65)
    } else {
        theme.colors.border.opacity(0.45)
    };

    div()
        .flex_shrink_0()
        .size(px(12.0))
        .rounded_full()
        .border_1()
        .border_color(border)
        .when(selected, |this| this.bg(theme.colors.ring).border_color(theme.colors.ring))
}

/// Standardized truncating value label for primary menu rows.
#[must_use]
pub fn cascading_menu_row_value(theme: &UiTheme, value: impl IntoElement) -> Div {
    div()
        .flex_1()
        .min_w_0()
        .text_color(theme.colors.foreground_muted)
        .truncate()
        .child(value)
}

/// Standard select-style row with an icon/radio slot and a truncating label.
///
/// Use this for option menus that should share keyboard-highlight and hover behavior.
#[must_use]
pub fn cascading_select_menu_item(
    theme: &UiTheme,
    style: CascadingMenuRowStyle,
    keyboard_selected: bool,
    hover_opacity: f32,
    icon: impl IntoElement,
    label: impl IntoElement,
) -> Div {
    cascading_menu_row(theme, style, keyboard_selected, hover_opacity)
        .child(icon)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_row()
                .items_center()
                .text_color(theme.colors.foreground)
                .truncate()
                .child(label),
        )
}

/// Standardized surface wrapper for cascading menus.
#[must_use]
pub fn cascading_menu_surface(theme: &UiTheme, style: CascadingMenuSurfaceStyle) -> Div {
    overlay_surface(theme, OverlaySurfaceKind::Menu, theme.radius.lg)
        .pt(style.padding_y)
        .pb(style.padding_y)
        .px(style.padding_x)
        .shadow_md()
        .occlude()
}

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
