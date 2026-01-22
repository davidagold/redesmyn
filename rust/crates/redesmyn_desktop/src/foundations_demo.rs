use std::time::Duration;

use gpui::{
    App, AsyncApp, Context, Entity, FocusHandle, Focusable, Render, ScrollHandle, SharedString,
    Subscription, Task, Window, div, prelude::*, px,
};

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ProgressPillKind, ScrollArea,
    SplitPane, SplitPaneAxis, SplitPaneState, TextArea, TextButton, TextInput, TextInputEvent,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{UserActionState, theme_for_window};

pub struct FoundationsDemo {
    scroll_handle: ScrollHandle,
    split_pane: Entity<SplitPane>,
    text_input: Entity<TextInput>,
    text_area: Entity<TextArea>,
    last_submitted: Option<SharedString>,
    theme_error: Option<SharedString>,
    demo_action: UserActionState,
    demo_action_attempts: u32,
    demo_action_task: Option<Task<()>>,
    _subscriptions: Vec<Subscription>,
}

impl FoundationsDemo {
    pub fn new(cx: &mut Context<Self>) -> Self {
        let scroll_handle = ScrollHandle::new();
        let text_input = cx.new(|cx| TextInput::new(cx).placeholder("Draft request…"));
        let text_area = cx.new(|cx| TextArea::new(cx).placeholder("Notes…"));

        let split_primary = cx.new(|_| SplitPaneDemoPane {
            title: "Primary pane".into(),
            hint: "Drag the divider to resize. Double-click to collapse.".into(),
        });
        let split_secondary = cx.new(|_| SplitPaneDemoPane {
            title: "Secondary pane".into(),
            hint: "This is a placeholder surface for future panes.".into(),
        });
        let split_pane = cx.new(|_| {
            SplitPane::new(
                SplitPaneAxis::Horizontal,
                SplitPaneState::default(),
                split_primary.into(),
                split_secondary.into(),
            )
            .min_primary_px(240.0)
        });

        let mut subscriptions = Vec::new();
        subscriptions.push(cx.observe_global::<UiContext>(|_, cx| cx.notify()));

        subscriptions.push(cx.subscribe(&text_input, |this, _, event, cx| {
            if let TextInputEvent::Submitted(text) = event {
                this.last_submitted = Some(text.clone());
                cx.notify();
            }
        }));

        Self {
            scroll_handle,
            split_pane,
            text_input,
            text_area,
            last_submitted: None,
            theme_error: None,
            demo_action: UserActionState::default(),
            demo_action_attempts: 0,
            demo_action_task: None,
            _subscriptions: subscriptions,
        }
    }

    pub fn text_input_focus_handle(&self, cx: &App) -> FocusHandle {
        self.text_input.focus_handle(cx)
    }

    fn set_theme_preference(&mut self, preference: ThemePreference, cx: &mut Context<Self>) {
        let span = redesmyn_logging::redesmyn_info_span!(
            "ui_theme_preference_set",
            preference = ?preference
        );
        let _guard = span.enter();

        match cx
            .global_mut::<UiContext>()
            .set_theme_preference(preference)
        {
            Ok(()) => {
                self.theme_error = None;
                redesmyn_logging::tracing::info!("saved theme preference");
            }
            Err(error) => {
                redesmyn_logging::tracing::error!(error = %error, "failed to save theme preference");
                self.theme_error = Some(error.to_string().into());
            }
        }

        cx.notify();
    }

    fn clear_theme_error(&mut self, cx: &mut Context<Self>) {
        self.theme_error = None;
        cx.notify();
    }

    fn run_demo_action(&mut self, cx: &mut Context<Self>) {
        if self.demo_action.in_flight {
            return;
        }

        self.demo_action.start();
        self.demo_action_attempts += 1;
        let attempt = self.demo_action_attempts;
        let draft_len = self.text_input.read(cx).text().len();

        redesmyn_logging::tracing::info!(attempt, draft_len, "starting demo action (simulated)");

        cx.notify();

        self.demo_action_task = Some(cx.spawn(
            move |weak: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    gpui::Timer::after(Duration::from_millis(900)).await;
                    let Some(entity) = weak.upgrade() else {
                        return;
                    };

                    let success = attempt % 2 == 0;
                    if cx
                        .update(|cx| {
                            entity.update(cx, |this, cx| {
                                this.demo_action_task = None;
                                if success {
                                    this.demo_action.succeed();
                                } else {
                                    this.demo_action
                                        .fail(format!("Simulated error (attempt {attempt})."));
                                }
                                cx.notify();
                            })
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            },
        ));
    }

    fn clear_demo_error(&mut self, cx: &mut Context<Self>) {
        self.demo_action.clear_error();
        cx.notify();
    }
}

impl Render for FoundationsDemo {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let demo = cx.entity();

        let preference = cx
            .try_global::<UiContext>()
            .map(|ui| ui.theme_preference())
            .unwrap_or(ThemePreference::System);

        let theme_button = |id, label: &'static str, pref| {
            let kind = if preference == pref {
                ButtonKind::Primary
            } else {
                ButtonKind::Secondary
            };

            TextButton::new((id, cx.entity_id()), label)
                .kind(kind)
                .on_click({
                    let demo = demo.clone();
                    move |_, _, cx| {
                        demo.update(cx, |this, cx| this.set_theme_preference(pref, cx));
                    }
                })
        };

        let header = div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .px(theme.spacing.lg)
            .py(theme.spacing.md)
            .bg(theme.colors.surface)
            .border_b_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .flex()
                    .gap(theme.spacing.sm)
                    .items_center()
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground)
                            .child("UI Foundations"),
                    )
                    .child(
                        IconButton::new(("header_info", cx.entity_id()), div().child("i"))
                            .tooltip("A small survey surface for `redesmyn_ui`."),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.xs)
                    .child(theme_button("theme_light", "Light", ThemePreference::Light))
                    .child(theme_button("theme_dark", "Dark", ThemePreference::Dark))
                    .child(theme_button(
                        "theme_system",
                        "System",
                        ThemePreference::System,
                    )),
            );

        let mut content = div()
            .flex()
            .flex_col()
            .gap(theme.spacing.lg)
            .px(theme.spacing.lg)
            .py(theme.spacing.lg);

        if let Some(error) = self.theme_error.clone() {
            content = content.child(
                Callout::new(error)
                    .kind(CalloutKind::Danger)
                    .title("Theme save failed")
                    .action(
                        TextButton::new(("theme_error_dismiss", cx.entity_id()), "Dismiss")
                            .kind(ButtonKind::Ghost)
                            .on_click({
                                let demo = demo.clone();
                                move |_, _, cx| {
                                    demo.update(cx, |this, cx| this.clear_theme_error(cx));
                                }
                            }),
                    ),
            );
        }

        content = content.child(
            div()
                .flex()
                .flex_col()
                .gap(theme.spacing.sm)
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .child("Buttons"),
                )
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .flex_wrap()
                        .gap(theme.spacing.sm)
                        .child(
                            TextButton::new(("btn_primary", cx.entity_id()), "Primary")
                                .kind(ButtonKind::Primary)
                                .tooltip("Primary action"),
                        )
                        .child(TextButton::new(
                            ("btn_secondary", cx.entity_id()),
                            "Secondary",
                        ))
                        .child(
                            TextButton::new(("btn_ghost", cx.entity_id()), "Ghost")
                                .kind(ButtonKind::Ghost),
                        )
                        .child(
                            TextButton::new(("btn_danger", cx.entity_id()), "Danger")
                                .kind(ButtonKind::Danger),
                        )
                        .child(
                            TextButton::new(("btn_disabled", cx.entity_id()), "Disabled")
                                .disabled(true)
                                .disabled_reason("Requires a connected daemon"),
                        ),
                ),
        );

        content = content.child(
            div()
                .flex()
                .flex_col()
                .gap(theme.spacing.sm)
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .child("Text Input"),
                )
                .child(self.text_input.clone())
                .when_some(self.last_submitted.clone(), |this, submitted| {
                    this.child(
                        Callout::new(submitted)
                            .kind(CalloutKind::Info)
                            .title("Last submitted"),
                    )
                }),
        );

        let mut action_row = div()
            .flex()
            .flex_row()
            .items_center()
            .gap(theme.spacing.sm)
            .child(
                TextButton::new(("demo_action", cx.entity_id()), "Run demo action")
                    .kind(ButtonKind::Primary)
                    .disabled(self.demo_action.in_flight)
                    .disabled_reason("Already running")
                    .tooltip("Demonstrates the `UserActionState` no-silent-actions pattern.")
                    .on_click({
                        let demo = demo.clone();
                        move |_, _, cx| {
                            demo.update(cx, |this, cx| this.run_demo_action(cx));
                        }
                    }),
            );
        if self.demo_action.in_flight {
            action_row =
                action_row.child(ProgressPill::new("Running").kind(ProgressPillKind::Accent));
        }

        content = content.child(
            div()
                .flex()
                .flex_col()
                .gap(theme.spacing.sm)
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .child("No Silent Actions"),
                )
                .child(action_row)
                .when_some(self.demo_action.error.clone(), |this, error| {
                    let demo = demo.clone();
                    this.child(
                        Callout::new(error)
                            .kind(CalloutKind::Danger)
                            .title("Action failed")
                            .action(
                                TextButton::new(("demo_action_dismiss", cx.entity_id()), "Dismiss")
                                    .kind(ButtonKind::Ghost)
                                    .on_click({
                                        let demo = demo.clone();
                                        move |_, _, cx| {
                                            demo.update(cx, |this, cx| this.clear_demo_error(cx));
                                        }
                                    }),
                            ),
                    )
                }),
        );

        content = content.child(
            div()
                .flex()
                .flex_col()
                .gap(theme.spacing.sm)
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .child("Text Area"),
                )
                .child(self.text_area.clone()),
        );

        content = content.child(
            div()
                .flex()
                .flex_col()
                .gap(theme.spacing.sm)
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .child("SplitPane"),
                )
                .child(
                    div()
                        .h(px(220.0))
                        .bg(theme.colors.surface_elevated)
                        .rounded(theme.radius.md)
                        .overflow_hidden()
                        .child(self.split_pane.clone()),
                ),
        );

        let body = ScrollArea::new(
            ("foundations_scroll", cx.entity_id()),
            self.scroll_handle.clone(),
        )
        .scrollbar_width(px(10.0))
        .child(content);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .child(header)
            .child(div().flex_1().child(body))
    }
}

struct SplitPaneDemoPane {
    title: SharedString,
    hint: SharedString,
}

impl Render for SplitPaneDemoPane {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .px(theme.spacing.md)
            .py(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child(self.title.clone()),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(self.hint.clone()),
            )
    }
}
