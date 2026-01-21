mod command_palette_overlay;

use std::sync::Arc;

use gpui::{
    div, prelude::*, px, App, ClickEvent, Context, Entity, EventEmitter, FocusHandle, Focusable,
    Render, SharedString, Subscription, Window,
};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    Callout, CalloutKind, IconButton, SplitPane, SplitPaneAxis, SplitPaneEvent, SplitPaneState,
};
use redesmyn_ui::utils::theme_for_window;
use redesmyn_ui_graph::GraphView;

use crate::command_palette::{
    CloseCommandPalette, SelectNextCommand, SelectPreviousCommand, ToggleCommandPalette,
};

use self::command_palette_overlay::CommandPaletteOverlay;

#[derive(Debug)]
pub struct DesktopModel {
    config: Arc<redesmyn_config::RustConfig>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
    control_plane_client: Option<ClientInProcEndpoint>,
}

impl DesktopModel {
    #[must_use]
    pub fn new(
        config: Arc<redesmyn_config::RustConfig>,
        daemon_host_id: Option<redesmyn_ids::HostId>,
        control_plane_client: Option<ClientInProcEndpoint>,
    ) -> Self {
        Self {
            config,
            daemon_host_id,
            control_plane_client,
        }
    }
}

pub struct RootView {
    split_pane: Entity<SplitPane>,
    workspace_pane: Entity<WorkspacePaneHost>,
    focus_handle: FocusHandle,
    command_palette: CommandPaletteOverlay,
    _subscriptions: Vec<Subscription>,
}

impl RootView {
    #[must_use]
    pub fn new(model: Entity<DesktopModel>, cx: &mut Context<Self>) -> Self {
        let initial_split_state = cx
            .try_global::<UiContext>()
            .map(|ui| ui.main_split_pane_state())
            .unwrap_or_else(SplitPaneState::default);

        let session_pane = cx.new(EpicSessionPaneHost::new);
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

        subscriptions.push(cx.subscribe(&workspace_pane, |this, _, event, cx| {
            if matches!(event, WorkspacePaneHostEvent::ToggleSessionsPane) {
                this.split_pane
                    .update(cx, |pane, cx| pane.toggle_collapsed(cx));
            }
        }));

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
            split_pane,
            workspace_pane,
            focus_handle,
            command_palette,
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
}

impl Focusable for RootView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for RootView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let mut root = div()
            .id(("desktop_root", cx.entity_id()))
            .relative()
            .key_context("Desktop")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::toggle_command_palette))
            .on_action(cx.listener(Self::close_command_palette))
            .on_action(cx.listener(Self::select_previous_command))
            .on_action(cx.listener(Self::select_next_command))
            .flex()
            .size_full()
            .bg(theme.colors.background)
            .child(self.split_pane.clone());

        if self.command_palette.is_open() {
            root = root.child(self.command_palette.render(window, cx));
        }

        root
    }
}

struct EpicSessionPaneHost;

impl EpicSessionPaneHost {
    fn new(_: &mut Context<Self>) -> Self {
        Self
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
                    .px(theme.spacing.md)
                    .py(theme.spacing.md)
                    .flex()
                    .flex_col()
                    .gap(theme.spacing.sm)
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("EpicSessionPaneHost (placeholder)")
                    .child("Drag the divider to resize; double-click to collapse/expand."),
            )
    }
}

#[derive(Clone, Debug)]
enum WorkspacePaneHostEvent {
    ToggleSessionsPane,
}

struct WorkspacePaneHost {
    focus_handle: FocusHandle,
    model: Entity<DesktopModel>,
    graph_view: Entity<GraphView>,
    sessions_collapsed: bool,
    ui_settings_error: Option<SharedString>,
}

impl EventEmitter<WorkspacePaneHostEvent> for WorkspacePaneHost {}

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

    fn emit_toggle(&mut self, _: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
        cx.emit(WorkspacePaneHostEvent::ToggleSessionsPane);
    }
}

impl Render for WorkspacePaneHost {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let model = self.model.read(cx);

        let workspace = cx.entity();
        let toggle_icon = if self.sessions_collapsed { "⟩" } else { "⟨" };
        let toggle_tooltip = if self.sessions_collapsed {
            "Show sessions pane"
        } else {
            "Hide sessions pane"
        };

        let header = div()
            .h(px(44.0))
            .px(theme.spacing.md)
            .flex()
            .flex_row()
            .items_center()
            .gap(theme.spacing.sm)
            .bg(theme.colors.surface_elevated)
            .child(
                IconButton::new(
                    ("workspace_toggle_sessions", cx.entity_id()),
                    div().child(toggle_icon),
                )
                .tooltip(toggle_tooltip)
                .on_click(move |event, window, cx| {
                    workspace.update(cx, |this, cx| this.emit_toggle(event, window, cx))
                }),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Workspace"),
            );

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

        let bootstrap_status = format!(
            "Bootstrap: control plane {} · daemon {} · host {} · client {}",
            if model.config.desktop.embed_control_plane {
                "embedded"
            } else {
                "external"
            },
            if model.config.desktop.embed_daemon {
                "embedded"
            } else {
                "external"
            },
            model
                .daemon_host_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "<none>".to_string()),
            if model.control_plane_client.is_some() {
                "in-proc"
            } else {
                "uds"
            }
        );

        body = body.child(
            div()
                .px(theme.spacing.md)
                .py(theme.spacing.sm)
                .text_sm()
                .text_color(theme.colors.foreground_muted)
                .child(bootstrap_status),
        );

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .child(header)
            .child(div().flex_1().child(body.child(self.graph_view.clone())))
            .track_focus(&self.focus_handle(cx))
    }
}
