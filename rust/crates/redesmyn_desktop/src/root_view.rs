mod command_palette_overlay;

use std::sync::Arc;

use gpui::{
    App, AsyncApp, Context, Entity, FocusHandle, Focusable, Render, ScrollHandle, SharedString,
    Subscription, Task, Window, div, prelude::*, px,
};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;
use redesmyn_ui_session::SessionView;

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ScrollArea, SplitPane,
    SplitPaneAxis, SplitPaneEvent, SplitPaneState, TextButton,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{UserActionState, theme_for_window};
use redesmyn_ui_graph::GraphView;

use crate::command_palette::{
    CloseCommandPalette, SelectNextCommand, SelectPreviousCommand, ToggleCommandPalette,
};
use crate::control_plane_client::ControlPlaneClient;

use self::command_palette_overlay::CommandPaletteOverlay;

#[derive(Debug)]
pub struct DesktopModel {
    config: Arc<redesmyn_config::RustConfig>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
    session_control_plane_client: Option<ClientInProcEndpoint>,
    chrome_control_plane_client: Option<ControlPlaneClient>,
}

impl DesktopModel {
    #[must_use]
    pub fn new(
        config: Arc<redesmyn_config::RustConfig>,
        daemon_host_id: Option<redesmyn_ids::HostId>,
        tokio_handle: tokio::runtime::Handle,
        session_control_plane_client: Option<ClientInProcEndpoint>,
        chrome_control_plane_client: Option<ClientInProcEndpoint>,
    ) -> Self {
        Self {
            config,
            daemon_host_id,
            session_control_plane_client,
            chrome_control_plane_client: chrome_control_plane_client
                .map(|conn| ControlPlaneClient::new(tokio_handle, conn)),
        }
    }

    pub fn take_control_plane_client(&mut self) -> Option<ClientInProcEndpoint> {
        self.session_control_plane_client.take()
    }
}

pub struct RootView {
    model: Entity<DesktopModel>,
    split_pane: Entity<SplitPane>,
    workspace_pane: Entity<WorkspacePaneHost>,
    focus_handle: FocusHandle,
    command_palette: CommandPaletteOverlay,
    chrome: ChromeState,
    _subscriptions: Vec<Subscription>,
}

impl RootView {
    #[must_use]
    pub fn new(model: Entity<DesktopModel>, cx: &mut Context<Self>) -> Self {
        let initial_split_state = cx
            .try_global::<UiContext>()
            .map(|ui| ui.main_split_pane_state())
            .unwrap_or_else(SplitPaneState::default);

        let session_pane = cx.new(|cx| EpicSessionPaneHost::new(model.clone(), cx));
        let workspace_pane =
            cx.new(|cx| WorkspacePaneHost::new(model.clone(), initial_split_state.collapsed, cx));

        let split_pane_state = initial_split_state;
        let split_primary = session_pane.clone();
        let split_secondary = workspace_pane.clone();

        let split_pane = cx.new(|_| {
            SplitPane::new(
                SplitPaneAxis::Horizontal,
                split_pane_state,
                split_primary.into(),
                split_secondary.into(),
            )
            .min_primary_px(280.0)
        });

        let focus_handle = cx.focus_handle();
        let command_palette = CommandPaletteOverlay::new(
            focus_handle.clone(),
            split_pane.clone(),
            workspace_pane.clone(),
            cx,
        );

        let palette_input = command_palette.input_entity();

        let mut subscriptions = Vec::new();
        subscriptions.push(cx.observe_global::<UiContext>(|_, cx| cx.notify()));

        subscriptions.push(cx.subscribe(&split_pane, |this, _, event, cx| match event {
            SplitPaneEvent::StateChanged(state) => {
                this.workspace_pane.update(cx, |pane, cx| {
                    pane.set_sessions_collapsed(state.collapsed, cx)
                });
                this.persist_split_pane_state(*state, cx);
            }
        }));

        subscriptions.push(cx.subscribe(&palette_input, |this, _, event, cx| {
            this.command_palette
                .handle_text_input_event(event.clone(), cx);
        }));

        Self {
            model,
            split_pane,
            workspace_pane,
            focus_handle,
            command_palette,
            chrome: ChromeState::new(),
            _subscriptions: subscriptions,
        }
    }

    fn persist_split_pane_state(&mut self, state: SplitPaneState, cx: &mut Context<Self>) {
        if cx.try_global::<UiContext>().is_none() {
            return;
        }

        let result = cx
            .global_mut::<UiContext>()
            .set_main_split_pane_state(state);
        match result {
            Ok(()) => self
                .workspace_pane
                .update(cx, |pane, cx| pane.set_ui_settings_error(None, cx)),
            Err(error) => {
                redesmyn_logging::tracing::error!(
                    error = %error,
                    "failed to persist main split pane state"
                );
                self.workspace_pane.update(cx, |pane, cx| {
                    pane.set_ui_settings_error(Some(error.to_string().into()), cx);
                });
            }
        }
    }

    fn toggle_command_palette(
        &mut self,
        _: &ToggleCommandPalette,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.toggle(window, cx);
    }

    fn close_command_palette(
        &mut self,
        _: &CloseCommandPalette,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.handle_close_action(window, cx);
    }

    fn select_previous_command(
        &mut self,
        _: &SelectPreviousCommand,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.select_previous(cx);
    }

    fn select_next_command(
        &mut self,
        _: &SelectNextCommand,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.select_next(cx);
    }

    fn toggle_sessions_pane(&mut self, cx: &mut Context<Self>) {
        self.split_pane
            .update(cx, |pane, cx| pane.toggle_collapsed(cx));
    }

    fn toggle_panel(&mut self, panel: ChromePanel, cx: &mut Context<Self>) {
        self.chrome.panel = match self.chrome.panel {
            Some(open) if open == panel => None,
            _ => Some(panel),
        };
        cx.notify();
    }

    fn close_panel(&mut self, cx: &mut Context<Self>) {
        if self.chrome.panel.take().is_some() {
            cx.notify();
        }
    }

    fn select_epic(&mut self, slug: String, cx: &mut Context<Self>) {
        let span =
            redesmyn_logging::redesmyn_info_span!("ui.chrome.select_epic", epic_slug = %slug);
        let _guard = span.enter();

        if self.chrome.selected_epic_slug.as_deref() == Some(slug.as_str()) {
            self.close_panel(cx);
            return;
        }

        self.chrome.selected_epic_slug = Some(slug);
        self.chrome.panel = None;
        cx.notify();
    }

    fn clear_refresh_error(&mut self, cx: &mut Context<Self>) {
        self.chrome.refresh.clear_error();
        cx.notify();
    }

    fn clear_theme_error(&mut self, cx: &mut Context<Self>) {
        self.chrome.theme_error = None;
        cx.notify();
    }

    fn set_theme_preference(&mut self, preference: ThemePreference, cx: &mut Context<Self>) {
        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.chrome.theme_preference_set",
            preference = ?preference
        );
        let _guard = span.enter();

        if cx.try_global::<UiContext>().is_none() {
            self.chrome.theme_error = Some("UI context unavailable; theme not saved.".into());
            cx.notify();
            return;
        }

        match cx
            .global_mut::<UiContext>()
            .set_theme_preference(preference)
        {
            Ok(()) => {
                self.chrome.theme_error = None;
                redesmyn_logging::tracing::info!("saved theme preference");
            }
            Err(error) => {
                redesmyn_logging::tracing::error!(
                    error = %error,
                    "failed to save theme preference"
                );
                self.chrome.theme_error = Some(error.to_string().into());
            }
        }

        cx.notify();
    }

    fn start_refresh(&mut self, reason: RefreshReason, cx: &mut Context<Self>) {
        if self.chrome.refresh.in_flight {
            return;
        }

        let span = redesmyn_logging::redesmyn_info_span!("ui.chrome.refresh", reason = ?reason);
        let _guard = span.enter();

        self.chrome.refresh.start();

        let client = self.model.read(cx).chrome_control_plane_client.clone();
        let tokio = client.as_ref().map(|client| client.tokio().clone());

        cx.notify();

        self.chrome.refresh_task = Some(cx.spawn(
            move |weak: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    let Some(entity) = weak.upgrade() else {
                        return;
                    };

                    let Some(client) = client else {
                        if cx
                            .update(|cx| {
                                entity.update(cx, |this, cx| {
                                    this.chrome.refresh_task = None;
                                    this.chrome
                                        .refresh
                                        .fail("Control plane client unavailable (not embedded).");
                                    cx.notify();
                                })
                            })
                            .is_err()
                        {
                            return;
                        }
                        return;
                    };

                    let tokio = tokio.expect("tokio handle is present when client is present");
                    let task = tokio.spawn(async move {
                        let status = client.status().await;
                        let epics = client.list_epics().await;
                        (status, epics)
                    });

                    let (status, epics) = match task.await {
                        Ok(result) => result,
                        Err(error) => {
                            let _ = cx.update(|cx| {
                                entity.update(cx, |this, cx| {
                                    this.chrome.refresh_task = None;
                                    this.chrome.refresh.fail(format!("Refresh failed: {error}"));
                                    cx.notify();
                                })
                            });
                            return;
                        }
                    };

                    let _ = cx.update(|cx| {
                        entity.update(cx, |this, cx| {
                            this.chrome.refresh_task = None;

                            let mut errors = Vec::new();

                            match status {
                                Ok(status) => {
                                    this.chrome.control_plane_status = Some(status);
                                }
                                Err(error) => {
                                    redesmyn_logging::tracing::error!(
                                        error = %error,
                                        "control plane status request failed"
                                    );
                                    errors.push(format!("Status: {error}"));
                                }
                            }

                            match epics {
                                Ok(mut epics) => {
                                    epics.sort_by(|a, b| a.slug.cmp(&b.slug));
                                    this.chrome.epics = epics;

                                    if let Some(selected) = this.chrome.selected_epic_slug.as_deref()
                                    {
                                        let still_present = this
                                            .chrome
                                            .epics
                                            .iter()
                                            .any(|epic| epic.slug == selected);
                                        if !still_present {
                                            redesmyn_logging::tracing::warn!(
                                                epic_slug = %selected,
                                                "selected epic no longer present; clearing selection"
                                            );
                                            this.chrome.selected_epic_slug = None;
                                        }
                                    }
                                }
                                Err(error) => {
                                    redesmyn_logging::tracing::error!(
                                        error = %error,
                                        "control plane list epics request failed"
                                    );
                                    errors.push(format!("Epics: {error}"));
                                }
                            }

                            if errors.is_empty() {
                                this.chrome.refresh.succeed();
                            } else {
                                this.chrome.refresh.fail(errors.join(" · "));
                            }

                            cx.notify();
                        })
                    });
                }
            },
        ));
    }
}

impl Focusable for RootView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for RootView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        if !self.chrome.did_startup_refresh {
            self.chrome.did_startup_refresh = true;
            self.start_refresh(RefreshReason::Startup, cx);
        }

        let model = self.model.read(cx);
        let split_state = self.split_pane.read(cx).state();
        let toggle_icon = if split_state.collapsed { "⟩" } else { "⟨" };
        let toggle_tooltip = if split_state.collapsed {
            "Show sessions pane"
        } else {
            "Hide sessions pane"
        };

        let root = cx.entity();

        let selected_epic = self
            .chrome
            .selected_epic_slug
            .as_deref()
            .and_then(|slug| self.chrome.epics.iter().find(|epic| epic.slug == slug))
            .cloned();
        let selected_epic_name = selected_epic.as_ref().map(|epic| epic.name.clone().into());

        let epic_button_label: SharedString = match selected_epic.as_ref() {
            Some(epic) => epic.slug.clone().into(),
            None if self.chrome.epics.is_empty() => "No epics".into(),
            None => "Select epic".into(),
        };

        let epic_button_disabled = self.chrome.epics.is_empty();
        let epic_button_disabled_reason = if self.chrome.refresh.in_flight {
            "Loading epics…"
        } else {
            "No epics available"
        };

        let preference = cx
            .try_global::<UiContext>()
            .map(|ui| ui.theme_preference())
            .unwrap_or(ThemePreference::System);

        let theme_pref_button = |id, label: &'static str, pref| {
            let kind = if preference == pref {
                ButtonKind::Primary
            } else {
                ButtonKind::Secondary
            };

            TextButton::new((id, cx.entity_id()), label)
                .kind(kind)
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.set_theme_preference(pref, cx));
                    }
                })
        };

        let status_dot_color = if self.chrome.control_plane_status.is_some() {
            theme.colors.ring
        } else if self.chrome.refresh.in_flight {
            theme.colors.border.opacity(0.7)
        } else {
            theme.colors.danger
        };

        let control_plane_label = if self.chrome.control_plane_status.is_some() {
            "Control plane: running"
        } else if self.chrome.refresh.in_flight {
            "Control plane: checking…"
        } else if model.chrome_control_plane_client.is_some()
            || model.config.desktop.embed_control_plane
        {
            "Control plane: unavailable"
        } else {
            "Control plane: external"
        };

        let daemon_label: SharedString = if model.config.desktop.embed_daemon {
            match model.daemon_host_id {
                Some(id) => format!("Daemon: embedded · host {id}").into(),
                None => "Daemon: embedded".into(),
            }
        } else {
            "Daemon: external (start: rn daemon run)".into()
        };

        let refresh_button = IconButton::new(("chrome_refresh", cx.entity_id()), div().child("↻"))
            .tooltip("Refresh")
            .disabled(self.chrome.refresh.in_flight)
            .disabled_reason("Refreshing…")
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| this.start_refresh(RefreshReason::Manual, cx));
                }
            });

        let settings_button =
            IconButton::new(("chrome_settings", cx.entity_id()), div().child("⚙"))
                .tooltip("Settings")
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.toggle_panel(ChromePanel::Settings, cx));
                    }
                });

        let theme_toggle_button =
            IconButton::new(("chrome_theme_toggle", cx.entity_id()), div().child("◐"))
                .tooltip("Toggle theme")
                .on_click({
                    let root = root.clone();
                    move |_, window, cx| {
                        let mode = theme_for_window(window, cx).mode;
                        let next = match mode {
                            redesmyn_ui::styles::ThemeMode::Dark => ThemePreference::Light,
                            redesmyn_ui::styles::ThemeMode::Light => ThemePreference::Dark,
                        };
                        root.update(cx, |this, cx| this.set_theme_preference(next, cx));
                    }
                });

        let epic_button =
            TextButton::new(("chrome_epic_selector", cx.entity_id()), epic_button_label)
                .kind(ButtonKind::Ghost)
                .disabled(epic_button_disabled)
                .disabled_reason(epic_button_disabled_reason)
                .trailing(div().child("▾"))
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.toggle_panel(ChromePanel::EpicMenu, cx));
                    }
                });

        let header = div()
            .h(px(44.0))
            .px(theme.spacing.md)
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .gap(theme.spacing.lg)
            .bg(theme.colors.surface_elevated)
            .border_b_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .child(
                        IconButton::new(
                            ("chrome_toggle_sessions", cx.entity_id()),
                            div().child(toggle_icon),
                        )
                        .tooltip(toggle_tooltip)
                        .on_click({
                            let root = root.clone();
                            move |_, _, cx| {
                                root.update(cx, |this, cx| this.toggle_sessions_pane(cx));
                            }
                        }),
                    )
                    .child(epic_button)
                    .when_some(selected_epic_name, |this, name: SharedString| {
                        this.child(
                            div()
                                .text_sm()
                                .text_color(theme.colors.foreground_muted)
                                .child(name),
                        )
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(theme.spacing.xs)
                            .child(div().size(px(8.0)).rounded(px(999.0)).bg(status_dot_color))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(control_plane_label),
                            ),
                    )
                    .child(
                        div()
                            .w(px(1.0))
                            .h(px(18.0))
                            .bg(theme.colors.border.opacity(0.6)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child(daemon_label),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .when(self.chrome.refresh.in_flight, |this| {
                        this.child(ProgressPill::new("Refreshing"))
                    })
                    .child(refresh_button)
                    .child(theme_toggle_button)
                    .child(settings_button),
            );

        let mut chrome_extras = div().flex().flex_col().gap(theme.spacing.sm);
        let epic_menu_top = px(44.0) + theme.spacing.xs;
        let epic_menu_left = theme.spacing.md + px(28.0) + theme.spacing.sm;

        if let Some(error) = self.chrome.refresh.error.clone() {
            chrome_extras = chrome_extras.child(
                div().px(theme.spacing.md).child(
                    Callout::new(error)
                        .kind(CalloutKind::Danger)
                        .title("Refresh failed")
                        .action(
                            TextButton::new(("refresh_error_dismiss", cx.entity_id()), "Dismiss")
                                .kind(ButtonKind::Ghost)
                                .on_click({
                                    let root = root.clone();
                                    move |_, _, cx| {
                                        root.update(cx, |this, cx| this.clear_refresh_error(cx));
                                    }
                                }),
                        ),
                ),
            );
        }

        if let Some(error) = self.chrome.theme_error.clone() {
            chrome_extras = chrome_extras.child(
                div().px(theme.spacing.md).child(
                    Callout::new(error)
                        .kind(CalloutKind::Warning)
                        .title("Theme save failed")
                        .action(
                            TextButton::new(("theme_error_dismiss", cx.entity_id()), "Dismiss")
                                .kind(ButtonKind::Ghost)
                                .on_click({
                                    let root = root.clone();
                                    move |_, _, cx| {
                                        root.update(cx, |this, cx| this.clear_theme_error(cx));
                                    }
                                }),
                        ),
                ),
            );
        }

        let epic_menu_overlay = if matches!(self.chrome.panel, Some(ChromePanel::EpicMenu)) {
            let mut menu_body = div().flex().flex_col().gap(theme.spacing.sm).child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Select an epic"),
            );

            if self.chrome.epics.is_empty() {
                menu_body = menu_body.child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("No epics available. Ensure the control plane can discover epics."),
                );
            } else {
                let scroll = self.chrome.epic_scroll.clone();
                let root_for_items = root.clone();
                let selected_slug = self.chrome.selected_epic_slug.clone();
                let in_flight = self.chrome.refresh.in_flight;
                let entity_id = cx.entity_id();

                let list = ScrollArea::new(("chrome_epic_scroll", cx.entity_id()), scroll)
                    .scrollbar_width(px(8.0))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .w_full()
                            .gap(theme.spacing.xs)
                            .children(self.chrome.epics.iter().cloned().map(|epic| {
                                let epic_slug = epic.slug.clone();
                                let epic_name = epic.name.clone();
                                let selected = selected_slug.as_deref() == Some(epic_slug.as_str());
                                let kind = if selected {
                                    ButtonKind::Secondary
                                } else {
                                    ButtonKind::Ghost
                                };
                                let item_id = (
                                    gpui::ElementId::from(("chrome_epic_item", entity_id)),
                                    epic_slug.clone(),
                                );
                                TextButton::new(item_id, epic_slug.clone())
                                    .menu_item()
                                    .kind(kind)
                                    .disabled(in_flight)
                                    .disabled_reason("Refreshing…")
                                    .tooltip(epic_name)
                                    .on_click({
                                        let root = root_for_items.clone();
                                        let slug = epic_slug.clone();
                                        move |_, _, cx| {
                                            root.update(cx, |this, cx| {
                                                this.select_epic(slug.clone(), cx)
                                            });
                                        }
                                    })
                            })),
                    );

                menu_body =
                    menu_body.child(div().h(px(220.0)).w_full().overflow_hidden().child(list));
            }

            Some(
                div()
                    .absolute()
                    .inset_0()
                    .child(div().absolute().inset_0().occlude().on_mouse_down(
                        gpui::MouseButton::Left,
                        {
                            let root = root.clone();
                            move |_, _, cx| {
                                root.update(cx, |this, cx| this.close_panel(cx));
                                cx.stop_propagation();
                            }
                        },
                    ))
                    .child(
                        div()
                            .absolute()
                            .top(epic_menu_top)
                            .left(epic_menu_left)
                            .child(
                                div()
                                    .w(px(360.0))
                                    .rounded(theme.radius.md)
                                    .shadow_md()
                                    .occlude()
                                    .child(
                                        div()
                                            .w_full()
                                            .p(theme.spacing.md)
                                            .rounded(theme.radius.md)
                                            .bg(theme.colors.surface)
                                            .border_1()
                                            .border_color(theme.colors.border.opacity(0.5))
                                            .overflow_hidden()
                                            .child(menu_body),
                                    ),
                            ),
                    ),
            )
        } else {
            None
        };

        if matches!(self.chrome.panel, Some(ChromePanel::Settings)) {
            let settings_panel = div()
                .p(theme.spacing.md)
                .rounded(theme.radius.md)
                .bg(theme.colors.surface)
                .border_1()
                .border_color(theme.colors.border.opacity(0.5))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.colors.foreground)
                                .child("Settings"),
                        )
                        .child(
                            IconButton::new(("settings_close", cx.entity_id()), div().child("×"))
                                .tooltip("Close")
                                .on_click({
                                    let root = root.clone();
                                    move |_, _, cx| {
                                        root.update(cx, |this, cx| this.close_panel(cx));
                                    }
                                }),
                        ),
                )
                .child(
                    div()
                        .pt(theme.spacing.md)
                        .flex()
                        .gap(theme.spacing.xs)
                        .child(theme_pref_button(
                            "theme_light",
                            "Light",
                            ThemePreference::Light,
                        ))
                        .child(theme_pref_button(
                            "theme_dark",
                            "Dark",
                            ThemePreference::Dark,
                        ))
                        .child(theme_pref_button(
                            "theme_system",
                            "System",
                            ThemePreference::System,
                        )),
                )
                .child(
                    div()
                        .pt(theme.spacing.md)
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("More settings coming soon."),
                );

            chrome_extras = chrome_extras.child(
                div()
                    .px(theme.spacing.md)
                    .pb(theme.spacing.md)
                    .child(settings_panel),
            );
        }

        let mut root_container = div()
            .id(("desktop_root", cx.entity_id()))
            .relative()
            .key_context("Desktop")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::toggle_command_palette))
            .on_action(cx.listener(Self::close_command_palette))
            .on_action(cx.listener(Self::select_previous_command))
            .on_action(cx.listener(Self::select_next_command))
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .capture_key_down({
                let root = root.clone();
                move |event, _, cx| {
                    if event.keystroke.key != "escape" {
                        return;
                    }

                    let did_close = root.update(cx, |this, cx| {
                        let was_open = this.chrome.panel.is_some();
                        if was_open {
                            this.close_panel(cx);
                        }
                        was_open
                    });

                    if did_close {
                        cx.stop_propagation();
                    }
                }
            })
            .child(header)
            .child(chrome_extras)
            .child(div().flex_1().min_h(px(0.0)).child(self.split_pane.clone()));

        if let Some(epic_menu_overlay) = epic_menu_overlay {
            root_container = root_container.child(epic_menu_overlay);
        }

        if self.command_palette.is_open() {
            root_container = root_container.child(self.command_palette.render(window, cx));
        }

        root_container
    }
}

struct EpicSessionPaneHost {
    session_view: Entity<SessionView>,
}

impl EpicSessionPaneHost {
    fn new(model: Entity<DesktopModel>, cx: &mut Context<Self>) -> Self {
        let client = model.update(cx, |model, _cx| model.take_control_plane_client());
        let session_view = cx.new(|cx| SessionView::new(client, cx));
        Self { session_view }
    }
}

impl Render for EpicSessionPaneHost {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.surface)
            .child(
                div()
                    .h(px(44.0))
                    .px(theme.spacing.md)
                    .flex()
                    .items_center()
                    .bg(theme.colors.surface_elevated)
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground)
                            .child("Sessions"),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h(px(0.0))
                    .px(theme.spacing.md)
                    .py(theme.spacing.md)
                    .child(self.session_view.clone()),
            )
    }
}

struct WorkspacePaneHost {
    focus_handle: FocusHandle,
    #[allow(dead_code)]
    model: Entity<DesktopModel>,
    graph_view: Entity<GraphView>,
    sessions_collapsed: bool,
    ui_settings_error: Option<SharedString>,
}

impl Focusable for WorkspacePaneHost {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl WorkspacePaneHost {
    fn new(model: Entity<DesktopModel>, sessions_collapsed: bool, cx: &mut Context<Self>) -> Self {
        Self {
            focus_handle: cx.focus_handle(),
            model,
            graph_view: cx.new(GraphView::new_demo),
            sessions_collapsed,
            ui_settings_error: None,
        }
    }

    fn set_sessions_collapsed(&mut self, collapsed: bool, cx: &mut Context<Self>) {
        if self.sessions_collapsed == collapsed {
            return;
        }
        self.sessions_collapsed = collapsed;
        cx.notify();
    }

    fn set_ui_settings_error(&mut self, error: Option<SharedString>, cx: &mut Context<Self>) {
        if self.ui_settings_error == error {
            return;
        }
        self.ui_settings_error = error;
        cx.notify();
    }

    fn refresh_graph(&mut self, cx: &mut Context<Self>) {
        self.graph_view = cx.new(GraphView::new_demo);
        cx.notify();
    }
}

impl Render for WorkspacePaneHost {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let mut body = div().flex().flex_col().size_full();

        if let Some(error) = self.ui_settings_error.clone() {
            body = body.child(
                div().px(theme.spacing.md).pt(theme.spacing.md).child(
                    Callout::new(error)
                        .kind(CalloutKind::Warning)
                        .title("Unable to save UI settings"),
                ),
            );
        }

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .child(body.child(self.graph_view.clone()))
            .track_focus(&self.focus_handle(cx))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefreshReason {
    Startup,
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChromePanel {
    EpicMenu,
    Settings,
}

struct ChromeState {
    did_startup_refresh: bool,
    panel: Option<ChromePanel>,
    epics: Vec<redesmyn_protocol::client::EpicSummary>,
    selected_epic_slug: Option<String>,
    control_plane_status: Option<redesmyn_protocol::client::StatusResponse>,
    refresh: UserActionState,
    refresh_task: Option<Task<()>>,
    theme_error: Option<SharedString>,
    epic_scroll: ScrollHandle,
}

impl ChromeState {
    fn new() -> Self {
        Self {
            did_startup_refresh: false,
            panel: None,
            epics: Vec::new(),
            selected_epic_slug: None,
            control_plane_status: None,
            refresh: UserActionState::default(),
            refresh_task: None,
            theme_error: None,
            epic_scroll: ScrollHandle::new(),
        }
    }
}
