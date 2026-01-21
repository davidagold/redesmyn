use std::sync::Arc;
use std::time::Duration;

use gpui::{
    div, prelude::*, px, App, AsyncApp, ClickEvent, Context, Entity, EventEmitter, FocusHandle,
    Focusable, Render, ScrollHandle, SharedString, Subscription, Task, Window,
};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ProgressPillKind, ScrollArea,
    SplitPane, SplitPaneAxis, SplitPaneEvent, SplitPaneState, TextButton, TextInput, TextInputEvent,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{UserActionState, theme_for_window};
use redesmyn_ui_graph::GraphView;

use crate::command_palette::{
    CloseCommandPalette, CommandGroup, CommandId, CommandRegistry, SelectNextCommand,
    SelectPreviousCommand, ToggleCommandPalette,
};

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
    command_registry: CommandRegistry,
    palette_open: bool,
    palette_mode: PaletteMode,
    palette_input: Entity<TextInput>,
    palette_scroll: ScrollHandle,
    palette_selected_index: usize,
    command_action: UserActionState,
    running_command: Option<CommandId>,
    running_command_task: Option<Task<()>>,
    selected_epic_slug: Option<SharedString>,
    graph_refresh_count: u64,
    _subscriptions: Vec<Subscription>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaletteMode {
    Commands,
    OpenEpic,
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

        let palette_input = cx.new(|cx| TextInput::new(cx).placeholder("Type a command…"));
        let palette_scroll = ScrollHandle::new();

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

        subscriptions.push(
            cx.subscribe(&palette_input, |this, _, event, cx| match event {
                TextInputEvent::Changed(_) => {
                    this.palette_selected_index = 0;
                    this.command_action.clear_error();
                    cx.notify();
                }
                TextInputEvent::Submitted(text) => {
                    this.on_palette_submit(text.clone(), cx);
                }
            }),
        );

        Self {
            split_pane,
            workspace_pane,
            focus_handle: cx.focus_handle(),
            command_registry: CommandRegistry::default(),
            palette_open: false,
            palette_mode: PaletteMode::Commands,
            palette_input,
            palette_scroll,
            palette_selected_index: 0,
            command_action: UserActionState::default(),
            running_command: None,
            running_command_task: None,
            selected_epic_slug: None,
            graph_refresh_count: 0,
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
        if self.palette_open {
            self.close_palette(window, cx);
        } else {
            self.open_palette(window, cx);
        }
    }

    fn close_command_palette(
        &mut self,
        _: &CloseCommandPalette,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.palette_open {
            return;
        }

        if self.palette_mode == PaletteMode::OpenEpic {
            self.exit_open_epic_mode(cx);
            window.focus(&self.palette_input.focus_handle(cx));
            return;
        }

        self.close_palette(window, cx);
    }

    fn select_previous_command(
        &mut self,
        _: &SelectPreviousCommand,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.palette_open || self.palette_mode != PaletteMode::Commands {
            return;
        }

        let visible = self.filtered_commands(cx);
        if visible.is_empty() {
            return;
        }

        if self.palette_selected_index == 0 {
            self.palette_selected_index = visible.len() - 1;
        } else {
            self.palette_selected_index = self.palette_selected_index.saturating_sub(1);
        }

        cx.notify();
    }

    fn select_next_command(
        &mut self,
        _: &SelectNextCommand,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.palette_open || self.palette_mode != PaletteMode::Commands {
            return;
        }

        let visible = self.filtered_commands(cx);
        if visible.is_empty() {
            return;
        }

        let next = self.palette_selected_index.saturating_add(1);
        self.palette_selected_index = if next >= visible.len() { 0 } else { next };
        cx.notify();
    }

    fn open_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let span = redesmyn_logging::redesmyn_info_span!("command_palette.open");
        let _guard = span.enter();

        self.palette_open = true;
        self.palette_mode = PaletteMode::Commands;
        self.palette_selected_index = 0;
        self.command_action.clear_error();
        self.running_command = None;

        self.palette_input
            .update(cx, |input, cx| input.set_text("", cx));

        window.focus(&self.palette_input.focus_handle(cx));
        cx.notify();
    }

    fn close_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let span = redesmyn_logging::redesmyn_info_span!("command_palette.close");
        let _guard = span.enter();

        self.palette_open = false;
        self.palette_mode = PaletteMode::Commands;
        self.command_action.clear_error();
        self.running_command = None;

        self.palette_input
            .update(cx, |input, cx| input.set_text("", cx));

        window.focus(&self.focus_handle);
        cx.notify();
    }

    fn on_palette_submit(&mut self, submitted: SharedString, cx: &mut Context<Self>) {
        if !self.palette_open {
            return;
        }

        if self.command_action.in_flight {
            return;
        }

        match self.palette_mode {
            PaletteMode::Commands => {
                let visible = self.filtered_commands(cx);
                let Some(command) = visible.get(self.palette_selected_index).copied() else {
                    self.command_action.fail("No matching commands.");
                    cx.notify();
                    return;
                };
                self.execute_command(command, cx);
            }
            PaletteMode::OpenEpic => {
                let slug = submitted.trim();
                if slug.is_empty() {
                    self.command_action.fail("Enter an epic slug to open.");
                    cx.notify();
                    return;
                }

                let span = redesmyn_logging::redesmyn_info_span!(
                    "command_palette.open_epic.select",
                    epic_slug = %slug
                );
                let _guard = span.enter();

                self.selected_epic_slug = Some(slug.to_string().into());
                self.exit_open_epic_mode(cx);
            }
        }
    }

    fn filtered_commands(&self, cx: &App) -> Vec<CommandId> {
        let query = self.palette_input.read(cx).text().as_ref();
        self.command_registry.matching(query)
    }

    fn disabled_reason_for(&self, command: CommandId) -> Option<SharedString> {
        match command {
            CommandId::RefreshGraph if self.selected_epic_slug.is_none() => {
                Some("Select an epic first.".into())
            }
            _ => None,
        }
    }

    fn execute_command(&mut self, command: CommandId, cx: &mut Context<Self>) {
        if self.command_action.in_flight {
            return;
        }

        if let Some(reason) = self.disabled_reason_for(command) {
            self.command_action.fail(reason);
            cx.notify();
            return;
        }

        if command == CommandId::OpenEpic {
            self.enter_open_epic_mode(cx);
            return;
        }

        let span = redesmyn_logging::redesmyn_info_span!(
            "command_palette.command.run",
            command_id = command.id_str(),
        );
        let _guard = span.enter();

        self.command_action.start();
        self.running_command = Some(command);
        cx.notify();

        match command {
            CommandId::ToggleTheme => {
                if let Err(error) = self.toggle_theme(cx) {
                    self.command_action.fail(error);
                } else {
                    self.command_action.succeed();
                }

                self.running_command = None;
                cx.notify();
            }
            CommandId::ToggleLeftSessionPane => {
                self.split_pane.update(cx, |pane, cx| pane.toggle_collapsed(cx));
                self.command_action.succeed();
                self.running_command = None;
                cx.notify();
            }
            CommandId::RefreshGraph => {
                let epic_slug = self
                    .selected_epic_slug
                    .clone()
                    .unwrap_or_else(|| "<none>".into());

                redesmyn_logging::tracing::info!(epic_slug = %epic_slug, "refreshing graph");

                self.running_command_task = Some(cx.spawn(
                    move |weak: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                        let cx = cx.clone();
                        async move {
                            gpui::Timer::after(Duration::from_millis(900)).await;
                            let Some(entity) = weak.upgrade() else {
                                return;
                            };

                            let _ = cx.update(|cx| {
                                entity.update(cx, |this, cx| {
                                    this.running_command_task = None;
                                    this.graph_refresh_count += 1;
                                    this.workspace_pane
                                        .update(cx, |pane, cx| pane.refresh_graph(cx));
                                    this.command_action.succeed();
                                    this.running_command = None;
                                    cx.notify();
                                });
                            });
                        }
                    },
                ));
            }
            CommandId::OpenEpic => {}
        }
    }

    fn toggle_theme(&mut self, cx: &mut Context<Self>) -> Result<(), SharedString> {
        let Some(ui) = cx.try_global::<UiContext>() else {
            return Err("UI context not initialized.".into());
        };

        let preference = ui.theme_preference();
        let next = match preference {
            ThemePreference::Light => ThemePreference::Dark,
            ThemePreference::Dark | ThemePreference::System => ThemePreference::Light,
        };

        match cx.global_mut::<UiContext>().set_theme_preference(next) {
            Ok(()) => {
                redesmyn_logging::tracing::info!(from = ?preference, to = ?next, "theme toggled");
                Ok(())
            }
            Err(err) => {
                redesmyn_logging::tracing::error!(error = %err, "failed to save theme preference");
                Err(err.to_string().into())
            }
        }
    }

    fn enter_open_epic_mode(&mut self, cx: &mut Context<Self>) {
        let span = redesmyn_logging::redesmyn_info_span!("command_palette.open_epic.enter");
        let _guard = span.enter();

        self.palette_mode = PaletteMode::OpenEpic;
        self.command_action.clear_error();
        self.palette_input
            .update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    fn exit_open_epic_mode(&mut self, cx: &mut Context<Self>) {
        self.palette_mode = PaletteMode::Commands;
        self.palette_selected_index = 0;
        self.command_action.clear_error();
        self.palette_input
            .update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    fn dismiss_command_error(&mut self, cx: &mut Context<Self>) {
        self.command_action.clear_error();
        cx.notify();
    }

    fn dismiss_palette_overlay(&mut self, cx: &mut Context<Self>) {
        self.palette_mode = PaletteMode::Commands;
        self.palette_open = false;
        self.command_action.clear_error();
        self.running_command = None;
        self.palette_input
            .update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    fn render_palette_overlay(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let root = cx.entity();

        let header_title = match self.palette_mode {
            PaletteMode::Commands => "Command palette",
            PaletteMode::OpenEpic => "Open epic…",
        };

        let running = self.running_command.map(|command| command.title());

        let left_header = div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child(header_title),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child(match self.palette_mode {
                        PaletteMode::Commands => "Type to filter, ↑/↓ to select, Enter to run.",
                        PaletteMode::OpenEpic => "Type an epic slug and press Enter.",
                    }),
            );

        let mut right_header = div().flex().flex_row().items_center().gap(theme.spacing.sm);

        if self.command_action.in_flight {
            if let Some(running) = running {
                right_header = right_header.child(
                    ProgressPill::new(format!("Running: {running}")).kind(ProgressPillKind::Accent),
                );
            } else {
                right_header = right_header
                    .child(ProgressPill::new("Running…").kind(ProgressPillKind::Accent));
            }
        }

        right_header = right_header.child(
            IconButton::new(("command_palette_close", cx.entity_id()), div().child("×"))
                .tooltip("Close (Esc)")
                .on_click({
                    let root = root.clone();
                    move |_, window, cx| {
                        let focus = root.read(cx).focus_handle.clone();
                        root.update(cx, |this, cx| this.dismiss_palette_overlay(cx));
                        window.focus(&focus);
                    }
                }),
        );

        let palette_header = div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .pb(theme.spacing.sm)
            .child(left_header)
            .child(right_header);

        let mut palette_body = div().flex().flex_col().gap(theme.spacing.sm);

        if let Some(error) = self.command_action.error.clone() {
            palette_body = palette_body.child(
                Callout::new(error)
                    .kind(CalloutKind::Danger)
                    .title("Command failed")
                    .action(
                        TextButton::new(("command_error_dismiss", cx.entity_id()), "Dismiss")
                            .kind(ButtonKind::Ghost)
                            .on_click({
                                let root = root.clone();
                                move |_, _, cx| {
                                    root.update(cx, |this, cx| this.dismiss_command_error(cx));
                                }
                            }),
                    ),
            );
        }

        palette_body = palette_body.child(self.palette_input.clone());

        match self.palette_mode {
            PaletteMode::Commands => {
                let visible = self.filtered_commands(cx);
                let selected_index = self
                    .palette_selected_index
                    .min(visible.len().saturating_sub(1));

                let mut last_group: Option<CommandGroup> = None;
                let mut list = div().flex().flex_col().gap(theme.spacing.xs);

                if visible.is_empty() {
                    list = list.child(
                        div()
                            .py(theme.spacing.sm)
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child("No commands match."),
                    );
                } else {
                    for (ix, command) in visible.iter().copied().enumerate() {
                        let group = command.group();
                        if Some(group) != last_group {
                            last_group = Some(group);
                            let label = match group {
                                CommandGroup::Navigation => "Navigation",
                                CommandGroup::Actions => "Actions",
                            };
                            list = list.child(
                                div()
                                    .pt(theme.spacing.sm)
                                    .text_xs()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(label),
                            );
                        }

                        let disabled_reason = self.disabled_reason_for(command);
                        let disabled = self.command_action.in_flight || disabled_reason.is_some();
                        let is_selected = ix == selected_index;

                        let mut row = div()
                            .id((command.id_str(), cx.entity_id()))
                            .flex()
                            .flex_col()
                            .gap(theme.spacing.xs)
                            .px(theme.spacing.md)
                            .py(theme.spacing.sm)
                            .rounded(theme.radius.md)
                            .border_1()
                            .border_color(theme.colors.border.opacity(0.6))
                            .bg(theme.colors.surface_elevated)
                            .when(is_selected, |this| {
                                this.border_color(theme.colors.ring).bg(theme.colors.accent)
                            })
                            .hover(|this| this.bg(theme.colors.accent))
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
                                            .child(command.title()),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(theme.colors.foreground_muted)
                                            .child(match group {
                                                CommandGroup::Navigation => "Nav",
                                                CommandGroup::Actions => "Action",
                                            }),
                                    ),
                            );

                        if let Some(reason) = disabled_reason.clone() {
                            row = row.child(
                                div()
                                    .text_xs()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(reason),
                            );
                        }

                        if disabled {
                            row = row.opacity(0.55).cursor_not_allowed();
                        } else {
                            row = row.cursor_pointer().on_click({
                                let root = root.clone();
                                move |_, _, cx| {
                                    root.update(cx, |this, cx| this.execute_command(command, cx));
                                }
                            });
                        }

                        list = list.child(row);
                    }
                }

                let list = ScrollArea::new(
                    ("command_palette_list", cx.entity_id()),
                    self.palette_scroll.clone(),
                )
                .scrollbar_width(px(10.0))
                .child(list);

                palette_body = palette_body.child(div().max_h(px(320.0)).child(list));
            }
            PaletteMode::OpenEpic => {
                palette_body = palette_body.child(
                    Callout::new(
                        "This is a v0 skeleton. Type an epic slug and press Enter to select it.",
                    )
                    .kind(CalloutKind::Info)
                    .title("Open epic"),
                );
                palette_body = palette_body.child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Esc returns to the command list."),
                );
            }
        }

        let palette_card = div()
            .key_context("CommandPalette")
            .w(px(680.0))
            .max_w(px(820.0))
            .px(theme.spacing.lg)
            .py(theme.spacing.lg)
            .bg(theme.colors.surface)
            .border_1()
            .border_color(theme.colors.border)
            .rounded(theme.radius.xl)
            .shadow_lg()
            .child(palette_header)
            .child(palette_body);

        div()
            .size_full()
            .child(
                div()
                    .size_full()
                    .bg(theme.colors.background.opacity(0.6))
                    .absolute()
                    .top_0()
                    .left_0()
                    .id(("command_palette_backdrop", cx.entity_id()))
                    .cursor_pointer()
                    .on_click({
                        let root = root.clone();
                        move |_, window, cx| {
                            let focus = root.read(cx).focus_handle.clone();
                            root.update(cx, |this, cx| this.dismiss_palette_overlay(cx));
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
                    .pt(px(120.0))
                    .child(palette_card),
            )
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

        if self.palette_open {
            root = root.child(self.render_palette_overlay(window, cx));
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
