use gpui::{
    AnyElement, App, ClickEvent, Context, Entity, SharedString, Window, div, prelude::*, px,
};

use redesmyn_ui::components::{
    Callout, CalloutKind, IconButton, ProgressPill, ProgressPillKind, TextInput,
};
use redesmyn_ui::utils::theme_for_window;

use super::RootView;

pub(super) struct SharedPaletteOverlay {
    pub key_context: &'static str,
    pub title: SharedString,
    pub subtitle: SharedString,
    pub progress_label: Option<SharedString>,
    pub progress_kind: ProgressPillKind,
    pub error_title: SharedString,
    pub error: Option<SharedString>,
    pub input: Entity<TextInput>,
    pub list: AnyElement,
    pub close_button_id: &'static str,
    pub close_tooltip: &'static str,
    pub backdrop_id: &'static str,
}

pub(super) fn render_shared_palette_overlay<CloseFn>(
    root: Entity<RootView>,
    model: SharedPaletteOverlay,
    window: &mut Window,
    cx: &mut Context<RootView>,
    on_close: CloseFn,
) -> impl IntoElement
where
    CloseFn: Fn(&mut RootView, &mut Context<RootView>) + Clone + 'static,
{
    let theme = theme_for_window(window, cx);

    let left_header = div()
        .flex()
        .flex_col()
        .gap(theme.spacing.xs)
        .child(
            div()
                .text_sm()
                .text_color(theme.colors.foreground)
                .child(model.title),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child(model.subtitle),
        );

    let mut right_header = div().flex().flex_row().items_center().gap(theme.spacing.sm);
    if let Some(progress_label) = model.progress_label {
        right_header = right_header.child(
            ProgressPill::new(progress_label)
                .kind(model.progress_kind)
                .into_any_element(),
        );
    }

    let palette_header = div()
        .flex()
        .flex_row()
        .items_center()
        .justify_between()
        .pb(theme.spacing.sm)
        .child(left_header)
        .child(right_header);

    let mut palette_body = div().flex().flex_col().gap(theme.spacing.sm);
    if let Some(error) = model.error {
        palette_body = palette_body.child(
            Callout::new(error)
                .kind(CalloutKind::Danger)
                .title(model.error_title),
        );
    }
    palette_body = palette_body.child(model.input).child(model.list);

    let close_button = IconButton::new((model.close_button_id, cx.entity_id()), div().child("×"))
        .tooltip(model.close_tooltip)
        .on_click({
            let root = root.clone();
            let on_close = on_close.clone();
            move |_: &ClickEvent, window, cx: &mut App| {
                let focus = root.read(cx).focus_handle.clone();
                root.update(cx, |this, cx| on_close(this, cx));
                window.focus(&focus);
            }
        });

    let palette_card = div()
        .key_context(model.key_context)
        .w(px(760.0))
        .max_w(px(920.0))
        .px(theme.spacing.lg)
        .py(theme.spacing.lg)
        .bg(theme.colors.surface)
        .border_1()
        .border_color(theme.colors.ring)
        .rounded(theme.radius.xl)
        .shadow_lg()
        .relative()
        .child(
            div()
                .absolute()
                .top(theme.spacing.sm)
                .right(theme.spacing.sm)
                .child(close_button),
        )
        .child(palette_header)
        .child(palette_body);

    div()
        .size_full()
        .absolute()
        .top_0()
        .left_0()
        .on_scroll_wheel(|_, _, cx| cx.stop_propagation())
        .child(
            div()
                .size_full()
                .bg(theme.colors.background.opacity(0.0))
                .absolute()
                .top_0()
                .left_0()
                .id((model.backdrop_id, cx.entity_id()))
                .cursor_pointer()
                .on_scroll_wheel(|_, _, cx| cx.stop_propagation())
                .on_click({
                    let root = root.clone();
                    move |_: &ClickEvent, window, cx: &mut App| {
                        let focus = root.read(cx).focus_handle.clone();
                        root.update(cx, |this, cx| on_close(this, cx));
                        window.focus(&focus);
                    }
                }),
        )
        .child(
            div()
                .size_full()
                .absolute()
                .top_0()
                .left_0()
                .flex()
                .flex_col()
                .items_center()
                .pt(px(104.0))
                .child(palette_card),
        )
}
