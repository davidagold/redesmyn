use gpui::{
    App, Context, Entity, FocusHandle, Focusable, ScrollHandle, SharedString, Task, Window, div,
    prelude::*, px,
};

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ProgressPillKind, ScrollArea,
    SplitPane, TextButton, TextInput, TextInputEvent,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{UserActionState, theme_for_window};

use crate::command_palette::{CommandGroup, CommandId, CommandRegistry};

use super::{RootView, WorkspacePaneHost};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaletteMode {
    Commands,
    OpenEpic,
}

pub struct CommandPaletteOverlay {
    root_focus_handle: FocusHandle,
    split_pane: Entity<SplitPane>,
    workspace_pane: Entity<WorkspacePaneHost>,
    command_registry: CommandRegistry,
    open: bool,
    mode: PaletteMode,
    input: Entity<TextInput>,
    scroll: ScrollHandle,
    selected_index: usize,
    action: UserActionState,
    running_command: Option<CommandId>,
    running_task: Option<Task<()>>,
    selected_epic_slug: Option<SharedString>,
    graph_refresh_count: u64,
}

impl CommandPaletteOverlay {
    pub fn new(
        root_focus_handle: FocusHandle,
        split_pane: Entity<SplitPane>,
        workspace_pane: Entity<WorkspacePaneHost>,
        cx: &mut Context<RootView>,
    ) -> Self {
        let input = cx.new(|cx| TextInput::new(cx).placeholder("Type a command…"));
        Self {
            root_focus_handle,
            split_pane,
            workspace_pane,
            command_registry: CommandRegistry::default(),
            open: false,
            mode: PaletteMode::Commands,
            input,
            scroll: ScrollHandle::new(),
            selected_index: 0,
            action: UserActionState::default(),
            running_command: None,
            running_task: None,
            selected_epic_slug: None,
            graph_refresh_count: 0,
        }
    }

    pub fn input_entity(&self) -> Entity<TextInput> {
        self.input.clone()
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn visible_error(&self) -> Option<String> {
        if !self.open {
            return None;
        }

        self.action.error.as_ref().map(|err| err.to_string())
    }

    pub fn visible_in_flight_label(&self) -> Option<String> {
        if !self.open || !self.action.in_flight {
            return None;
        }

        Some(
            self.running_command
                .map(|command| command.title().to_string())
                .unwrap_or_else(|| "Running…".to_string()),
        )
    }

    pub fn handle_text_input_event(&mut self, event: TextInputEvent, cx: &mut Context<RootView>) {
        match event {
            TextInputEvent::Changed(_) => {
                self.selected_index = 0;
                self.action.clear_error();
                cx.notify();
            }
            TextInputEvent::Submitted(text) => {
                self.on_submit(text, cx);
            }
            TextInputEvent::PastedImages(_) => {}
        }
    }

    pub fn toggle(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        if self.open {
            self.close_palette(window, cx);
        } else {
            self.open_palette(window, cx);
        }
    }

    pub fn handle_close_action(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        if !self.open {
            return;
        }

        if self.mode == PaletteMode::OpenEpic {
            self.exit_open_epic_mode(cx);
            window.focus(&self.input.focus_handle(cx));
            return;
        }

        self.close_palette(window, cx);
    }

    pub fn select_previous(&mut self, cx: &mut Context<RootView>) {
        if !self.open || self.mode != PaletteMode::Commands {
            return;
        }

        let visible = self.filtered_commands(cx);
        if visible.is_empty() {
            return;
        }

        if self.selected_index == 0 {
            self.selected_index = visible.len() - 1;
        } else {
            self.selected_index = self.selected_index.saturating_sub(1);
        }

        cx.notify();
    }

    pub fn select_next(&mut self, cx: &mut Context<RootView>) {
        if !self.open || self.mode != PaletteMode::Commands {
            return;
        }

        let visible = self.filtered_commands(cx);
        if visible.is_empty() {
            return;
        }

        let next = self.selected_index.saturating_add(1);
        self.selected_index = if next >= visible.len() { 0 } else { next };
        cx.notify();
    }

    pub fn render(&self, window: &mut Window, cx: &mut Context<RootView>) -> impl IntoElement {
        self.render_overlay(window, cx)
    }

    fn open_palette(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        let span = redesmyn_logging::redesmyn_info_span!("command_palette.open");
        let _guard = span.enter();

        self.open = true;
        self.mode = PaletteMode::Commands;
        self.selected_index = 0;
        self.action.clear_error();

        if !self.action.in_flight {
            self.running_command = None;
        }

        self.input.update(cx, |input, cx| input.set_text("", cx));
        window.focus(&self.input.focus_handle(cx));
        cx.notify();
    }

    fn close_palette(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        let span = redesmyn_logging::redesmyn_info_span!("command_palette.close");
        let _guard = span.enter();

        self.open = false;
        self.mode = PaletteMode::Commands;
        self.action.clear_error();

        if !self.action.in_flight {
            self.running_command = None;
        }

        self.input.update(cx, |input, cx| input.set_text("", cx));
        window.focus(&self.root_focus_handle);
        cx.notify();
    }

    fn dismiss_command_error(&mut self, cx: &mut Context<RootView>) {
        self.action.clear_error();
        cx.notify();
    }

    fn dismiss_overlay(&mut self, cx: &mut Context<RootView>) {
        self.mode = PaletteMode::Commands;
        self.open = false;
        self.action.clear_error();

        if !self.action.in_flight {
            self.running_command = None;
        }

        self.input.update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    fn on_submit(&mut self, submitted: SharedString, cx: &mut Context<RootView>) {
        if !self.open || self.action.in_flight {
            return;
        }

        match self.mode {
            PaletteMode::Commands => {
                let visible = self.filtered_commands(cx);
                let Some(command) = visible.get(self.selected_index).copied() else {
                    self.action.fail("No matching commands.");
                    cx.notify();
                    return;
                };
                self.execute_command(command, cx);
            }
            PaletteMode::OpenEpic => {
                let slug = submitted.trim();
                if slug.is_empty() {
                    self.action.fail("Enter an epic slug to open.");
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
        let query = self.input.read(cx).text().as_ref();
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

    fn execute_command(&mut self, command: CommandId, cx: &mut Context<RootView>) {
        if self.action.in_flight {
            return;
        }

        if let Some(reason) = self.disabled_reason_for(command) {
            self.action.fail(reason);
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

        self.action.start();
        self.running_command = Some(command);
        cx.notify();

        match command {
            CommandId::ToggleTheme => {
                if let Err(error) = self.toggle_theme(cx) {
                    self.action.fail(error);
                } else {
                    self.action.succeed();
                }

                self.running_command = None;
                cx.notify();
            }
            CommandId::ToggleLeftSessionPane => {
                self.split_pane
                    .update(cx, |pane, cx| pane.toggle_collapsed(cx));
                self.action.succeed();
                self.running_command = None;
                cx.notify();
            }
            CommandId::RefreshGraph => {
                self.graph_refresh_count += 1;
                self.workspace_pane
                    .update(cx, |pane, cx| pane.refresh_graph(cx));
                self.action.succeed();
                self.running_command = None;
                cx.notify();
            }
            CommandId::OpenEpic => {}
        }
    }

    fn toggle_theme(&mut self, cx: &mut Context<RootView>) -> Result<(), SharedString> {
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

    fn enter_open_epic_mode(&mut self, cx: &mut Context<RootView>) {
        let span = redesmyn_logging::redesmyn_info_span!("command_palette.open_epic.enter");
        let _guard = span.enter();

        self.mode = PaletteMode::OpenEpic;
        self.action.clear_error();
        self.input.update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    fn exit_open_epic_mode(&mut self, cx: &mut Context<RootView>) {
        self.mode = PaletteMode::Commands;
        self.selected_index = 0;
        self.action.clear_error();
        self.input.update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    fn render_overlay(&self, window: &mut Window, cx: &mut Context<RootView>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let root = cx.entity();

        let header_title = match self.mode {
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
                    .child(match self.mode {
                        PaletteMode::Commands => "Type to filter, ↑/↓ to select, Enter to run.",
                        PaletteMode::OpenEpic => "Type an epic slug and press Enter.",
                    }),
            );

        let mut right_header = div().flex().flex_row().items_center().gap(theme.spacing.sm);

        if self.action.in_flight {
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
                        root.update(cx, |this, cx| {
                            this.command_palette.dismiss_overlay(cx);
                            this.ui_updates.bump();
                        });
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

        if let Some(error) = self.action.error.clone() {
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
                                    root.update(cx, |this, cx| {
                                        this.command_palette.dismiss_command_error(cx);
                                        this.ui_updates.bump();
                                    });
                                }
                            }),
                    ),
            );
        }

        palette_body = palette_body.child(self.input.clone());

        match self.mode {
            PaletteMode::Commands => {
                let visible = self.filtered_commands(cx);
                let selected_index = self.selected_index.min(visible.len().saturating_sub(1));

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
                        let disabled = self.action.in_flight || disabled_reason.is_some();
                        let is_selected = ix == selected_index;

                        let mut row = div()
                            .id((command.id_str(), cx.entity_id()))
                            .flex()
                            .flex_col()
                            .gap(theme.spacing.xs)
                            .px(theme.spacing.md)
                            .py(theme.spacing.sm)
                            .rounded(theme.radius.md)
                            .bg(theme.colors.surface_elevated)
                            .when(is_selected, |this| {
                                this.bg(theme.colors.accent)
                                    .border_1()
                                    .border_color(theme.colors.ring)
                            })
                            .when(!is_selected, |this| {
                                this.hover(|this| this.bg(theme.colors.accent))
                            });

                        row = row.child(
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
                                    root.update(cx, |this, cx| {
                                        this.command_palette.execute_command(command, cx);
                                        this.ui_updates.bump();
                                    });
                                }
                            });
                        }

                        list = list.child(row);
                    }
                }

                let list = ScrollArea::new(
                    ("command_palette_list", cx.entity_id()),
                    self.scroll.clone(),
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
            .absolute()
            .top_0()
            .left_0()
            .child(
                div()
                    .size_full()
                    .bg(theme.colors.background.opacity(0.0))
                    .absolute()
                    .top_0()
                    .left_0()
                    .id(("command_palette_backdrop", cx.entity_id()))
                    .cursor_pointer()
                    .on_click({
                        let root = root.clone();
                        move |_, window, cx| {
                            let focus = root.read(cx).focus_handle.clone();
                            root.update(cx, |this, cx| {
                                this.command_palette.dismiss_overlay(cx);
                                this.ui_updates.bump();
                            });
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
