use std::path::PathBuf;
use std::time::Duration;

use gpui::{
    AnyElement, Context, Entity, FocusHandle, MouseButton, ScrollHandle, SharedString, WeakEntity,
    Window, div, prelude::*, px,
};

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ScrollArea, TextArea, TextButton, TextInput,
    TextInputEvent,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{
    TransitionMap, UserActionState, theme_for_window, ui_test_mode_animation_duration,
};

use crate::orchestration_config::{
    AgentKindSelection, OrchestrationConfigError, OrchestrationDefaults, SandboxNetworkMode,
    SandboxType, load_effective_defaults, repo_config_path, repo_root_from_cwd,
    write_repo_defaults,
};

use super::RootView;

const SETTINGS_DIALOG_TRANSITION_KEY: &str = "settings_dialog";

const BUILT_IN_PRELUDE_TEMPLATE: &str = "Redesmyn agent prelude\n\
\n\
Assignment\n\
\n\
- You are assigned task {task_id}: {task_title}.\n\
- Read the task doc at {task_doc} and implement its requirements.\n\
\n\
Objective\n\
\n\
- Complete the task end-to-end: implement, validate, and leave the branch in a clean state.\n\
\n\
Context\n\
\n\
- Epic: {epic_slug} (read {epic_readme})\n\
- Branch: {branch}\n\
- Worktree: {worktree}\n\
\n\
Process\n\
\n\
- Read AGENTS.md at repo root and follow it.\n\
- Read the epic README and the task README before coding.\n\
- Skim related tasks (parent, children, blockers) to understand context and avoid conflicts.\n\
- Keep changes small, well-typed, and easy to review; avoid unrelated changes.\n\
- If requirements or context are unclear, ask before making big assumptions.\n\
\n\
Requirements\n\
\n\
- Run `just check` before you finish.\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingsSection {
    Appearance,
    Agents,
}

pub struct SettingsDialog {
    root_focus_handle: FocusHandle,
    visible: bool,
    transitions: TransitionMap<&'static str>,
    section: SettingsSection,
    scroll: ScrollHandle,
    repo_root: Option<PathBuf>,
    repo_config_path: Option<PathBuf>,
    baseline: OrchestrationDefaults,
    draft: OrchestrationDefaults,
    show_default_prelude: bool,
    command_input: Entity<TextInput>,
    prelude_input: Entity<TextArea>,
    save: UserActionState,
    notice: Option<SharedString>,
}

impl SettingsDialog {
    pub fn new(root_focus_handle: FocusHandle, cx: &mut Context<RootView>) -> Self {
        let command_input = cx.new(|cx| TextInput::new(cx).placeholder("codex"));
        let prelude_input = cx.new(|cx| {
            TextArea::new(cx)
                .placeholder("Optional. Leave blank to use the built-in prelude.")
                .min_rows(12)
                .max_rows(12)
        });

        Self {
            root_focus_handle,
            visible: false,
            transitions: TransitionMap::new(),
            section: SettingsSection::Appearance,
            scroll: ScrollHandle::new(),
            repo_root: None,
            repo_config_path: None,
            baseline: OrchestrationDefaults::default(),
            draft: OrchestrationDefaults::default(),
            show_default_prelude: false,
            command_input,
            prelude_input,
            save: UserActionState::default(),
            notice: None,
        }
    }

    pub fn command_input_entity(&self) -> Entity<TextInput> {
        self.command_input.clone()
    }

    pub fn prelude_input_entity(&self) -> Entity<TextArea> {
        self.prelude_input.clone()
    }

    pub fn sync_visibility(
        &mut self,
        visible: bool,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) {
        if visible == self.visible {
            return;
        }

        self.visible = visible;

        if visible {
            self.open(window, cx);
        } else {
            window.focus(&self.root_focus_handle);
        }
    }

    pub fn handle_command_input_event(
        &mut self,
        event: TextInputEvent,
        cx: &mut Context<RootView>,
    ) {
        let TextInputEvent::Changed(value) = event else {
            return;
        };

        let trimmed = value.trim();
        self.draft.harness.command = (!trimmed.is_empty()).then(|| trimmed.to_string());
        self.notice = None;
        self.save.clear_error();
        cx.notify();
    }

    pub fn handle_prelude_input_event(
        &mut self,
        event: TextInputEvent,
        cx: &mut Context<RootView>,
    ) {
        let TextInputEvent::Changed(value) = event else {
            return;
        };

        self.draft.harness.prelude = (!value.trim().is_empty()).then(|| value.to_string());
        self.notice = None;
        self.save.clear_error();
        cx.notify();
    }

    pub fn render(
        &mut self,
        root: &Entity<RootView>,
        visible: bool,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> Option<AnyElement> {
        let duration = ui_test_mode_animation_duration(Duration::from_millis(140));
        let opacity = self.transitions.opacity_for_render(
            SETTINGS_DIALOG_TRANSITION_KEY,
            visible,
            duration,
            window,
        );

        if opacity.abs() < 1e-3 {
            return None;
        }

        let theme = theme_for_window(window, cx);
        let overlay_bg = theme.colors.background.opacity(0.85);

        let close_on_background = {
            let root = root.clone();
            move |_: &gpui::MouseDownEvent, _: &mut Window, cx: &mut gpui::App| {
                root.update(cx, |this, cx| this.close_panel(cx));
                cx.stop_propagation();
            }
        };

        let overlay = div()
            .absolute()
            .inset_0()
            .opacity(opacity)
            .child(
                div()
                    .absolute()
                    .inset_0()
                    .bg(overlay_bg)
                    .on_mouse_down(MouseButton::Left, close_on_background),
            )
            .child(self.dialog_card(root, window, cx));

        Some(overlay.into_any_element())
    }

    fn open(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        self.section = SettingsSection::Appearance;
        self.show_default_prelude = false;
        self.save = UserActionState::default();
        self.notice = None;

        match self.load_defaults() {
            Ok((repo_root, repo_config_path, defaults)) => {
                self.repo_root = Some(repo_root);
                self.repo_config_path = Some(repo_config_path);
                self.baseline = defaults.clone();
                self.draft = defaults;
            }
            Err(err) => {
                self.repo_root = None;
                self.repo_config_path = None;
                self.baseline = OrchestrationDefaults::default();
                self.draft = OrchestrationDefaults::default();
                self.save.fail(err.to_string());
            }
        }

        let command = self.draft.harness.command.clone().unwrap_or_default();
        self.command_input
            .update(cx, move |input, cx| input.set_text(command, cx));

        let prelude = self.draft.harness.prelude.clone().unwrap_or_default();
        self.prelude_input
            .update(cx, move |input, cx| input.set_text(prelude, cx));

        window.focus(&self.root_focus_handle);
        cx.notify();
    }

    fn load_defaults(
        &self,
    ) -> Result<(PathBuf, PathBuf, OrchestrationDefaults), OrchestrationConfigError> {
        let repo_root = repo_root_from_cwd()?;
        let repo_config_path = repo_config_path(&repo_root);
        let defaults = load_effective_defaults(&repo_root)?;
        Ok((repo_root, repo_config_path, defaults))
    }

    fn dialog_card(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let close_button = IconButton::new(("settings_close", cx.entity_id()), div().child("×"))
            .tooltip("Close")
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| this.close_panel(cx));
                }
            });

        let config_path_label: SharedString = self
            .repo_config_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "config.toml".to_string())
            .into();

        let mut header_right = div().flex().flex_row().items_center().gap(theme.spacing.sm);

        header_right = header_right.child(self.save_reset_buttons(root, window, cx));
        header_right = header_right.child(close_button);

        let header = div()
            .px(theme.spacing.lg)
            .py(theme.spacing.md)
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .border_b_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(theme.spacing.xs)
                    .child(
                        div()
                            .text_lg()
                            .text_color(theme.colors.foreground)
                            .child("Settings"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child(format!("Updates `{}` (repo scope).", config_path_label)),
                    ),
            )
            .child(header_right);

        let body = div()
            .flex()
            .flex_row()
            .min_h(px(0.0))
            .child(self.section_nav(root, window, cx))
            .child(self.section_body(root, window, cx));

        div()
            .absolute()
            .inset_0()
            .p(theme.spacing.xl)
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .key_context("SettingsDialog")
                    .w_full()
                    .max_w(px(960.0))
                    .h_full()
                    .max_h(px(680.0))
                    .rounded(theme.radius.xl)
                    .bg(theme.colors.surface)
                    .border_1()
                    .border_color(theme.colors.border.opacity(0.7))
                    .shadow_lg()
                    .overflow_hidden()
                    .child(header)
                    .child(body),
            )
            .into_any_element()
    }

    fn save_reset_buttons(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let dirty = self.is_dirty();
        let theme = theme_for_window(window, cx);

        let save_button = TextButton::new(("settings_save", cx.entity_id()), "Save")
            .kind(ButtonKind::Ghost)
            .small()
            .disabled(self.save.in_flight || !dirty)
            .disabled_reason(if self.save.in_flight {
                "Saving…"
            } else {
                "No changes"
            })
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| this.settings_dialog.save(cx));
                }
            });

        let reset_button = TextButton::new(("settings_reset", cx.entity_id()), "Reset")
            .kind(ButtonKind::Ghost)
            .small()
            .disabled(self.save.in_flight || !dirty)
            .disabled_reason(if self.save.in_flight {
                "Saving…"
            } else {
                "No changes"
            })
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| this.settings_dialog.reset(cx));
                }
            });

        div()
            .flex()
            .flex_row()
            .items_center()
            .rounded(theme.radius.md)
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .overflow_hidden()
            .child(save_button)
            .child(
                div()
                    .border_l_1()
                    .border_color(theme.colors.border.opacity(0.6)),
            )
            .child(reset_button)
            .into_any_element()
    }

    fn section_nav(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let nav_item = |id: &'static str, label: &'static str, section: SettingsSection| {
            let selected = self.section == section;
            let kind = if selected {
                ButtonKind::Secondary
            } else {
                ButtonKind::Ghost
            };

            TextButton::new((id, cx.entity_id()), label)
                .kind(kind)
                .small()
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| {
                            this.settings_dialog.section = section;
                            cx.notify();
                        });
                    }
                })
        };

        div()
            .w(px(220.0))
            .p(theme.spacing.lg)
            .border_r_1()
            .border_color(theme.colors.border.opacity(0.6))
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Sections"),
            )
            .child(nav_item(
                "settings_nav_appearance",
                "Appearance",
                SettingsSection::Appearance,
            ))
            .child(nav_item(
                "settings_nav_agents",
                "Agents",
                SettingsSection::Agents,
            ))
            .into_any_element()
    }

    fn section_body(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let mut content = div()
            .p(theme.spacing.lg)
            .flex()
            .flex_col()
            .gap(theme.spacing.lg);

        if let Some(error) = self.save.error.clone() {
            content = content.child(
                Callout::new(error)
                    .kind(CalloutKind::Danger)
                    .title("Settings error"),
            );
        } else if let Some(notice) = self.notice.clone() {
            content = content.child(Callout::new(notice).kind(CalloutKind::Info));
        }

        content = match self.section {
            SettingsSection::Appearance => content.child(self.section_appearance(root, window, cx)),
            SettingsSection::Agents => content.child(self.section_agents(root, window, cx)),
        };

        ScrollArea::new(("settings_scroll", cx.entity_id()), self.scroll.clone())
            .bg(theme.colors.surface)
            .child(content)
            .into_any_element()
    }

    fn section_appearance(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let current = cx
            .try_global::<UiContext>()
            .map(|ui| ui.theme_preference())
            .unwrap_or(ThemePreference::System);

        let pref_button = |id: &'static str, label: &'static str, pref: ThemePreference| {
            let selected = current == pref;
            TextButton::new((id, cx.entity_id()), label)
                .kind(if selected {
                    ButtonKind::Secondary
                } else {
                    ButtonKind::Ghost
                })
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.set_theme_preference(pref, cx));
                    }
                })
        };

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_lg()
                    .text_color(theme.colors.foreground)
                    .child("Appearance"),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Theme preference"),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(theme.spacing.xs)
                    .child(pref_button("theme_light", "Light", ThemePreference::Light))
                    .child(pref_button("theme_dark", "Dark", ThemePreference::Dark))
                    .child(pref_button(
                        "theme_system",
                        "System",
                        ThemePreference::System,
                    )),
            )
            .into_any_element()
    }

    fn section_agents(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let mut body = div().flex().flex_col().gap(theme.spacing.lg);

        body = body
            .child(
                div()
                    .text_lg()
                    .text_color(theme.colors.foreground)
                    .child("Agents"),
            )
            .child(self.agent_section_agent(root, window, cx))
            .child(self.agent_section_prelude(root, window, cx));

        body.into_any_element()
    }

    fn agent_section_agent(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let agent_kind_buttons = div()
            .flex()
            .flex_row()
            .gap(theme.spacing.xs)
            .child(self.agent_kind_button(
                root,
                cx,
                "agent_kind_auto",
                "Auto",
                AgentKindSelection::Auto,
            ))
            .child(self.agent_kind_button(
                root,
                cx,
                "agent_kind_generic",
                "Generic",
                AgentKindSelection::Generic,
            ))
            .child(self.agent_kind_button(
                root,
                cx,
                "agent_kind_codex",
                "Codex",
                AgentKindSelection::Codex,
            ))
            .child(self.agent_kind_button(
                root,
                cx,
                "agent_kind_claude_code",
                "Claude Code",
                AgentKindSelection::ClaudeCode,
            ));

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Agent"),
            )
            .child(self.labeled_text_input(
                "Agent command",
                self.command_input.clone(),
                "Shell command used to start the agent inside each task’s worktree (e.g. `codex`).",
                window,
                cx,
            ))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(theme.spacing.xs)
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child("Agent kind"),
                    )
                    .child(agent_kind_buttons)
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child("Auto infers the agent from your command; switch if wrong."),
                    ),
            )
            .child(self.segmented_bool_setting(
                root,
                "Run mode",
                "Detached",
                "Foreground",
                self.draft.harness.detach,
                |draft, next| draft.harness.detach = next,
                "Detached runs in a tmux session; foreground runs in your current terminal.",
                window,
                cx,
            ))
            .child(self.sandbox_settings(root, window, cx))
            .into_any_element()
    }

    fn sandbox_settings(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let sandbox_type_buttons = div()
            .flex()
            .flex_row()
            .gap(theme.spacing.xs)
            .child(self.sandbox_type_button(root, cx, "sandbox_none", "Off", SandboxType::None))
            .child(self.sandbox_type_button(
                root,
                cx,
                "sandbox_worktree",
                "Worktree",
                SandboxType::Worktree,
            ));

        let deny_network_disabled = self.draft.sandbox_type != SandboxType::Worktree;

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Sandbox"),
            )
            .child(sandbox_type_buttons)
            .child(self.boolean_row(
                root,
                "deny_network",
                "Deny network",
                self.draft.sandbox_network == SandboxNetworkMode::Deny,
                deny_network_disabled,
                |draft, enabled| {
                    draft.sandbox_network = if enabled {
                        SandboxNetworkMode::Deny
                    } else {
                        SandboxNetworkMode::Allow
                    };
                },
                window,
                cx,
            ))
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Restricts agent writes to the task worktree and Redesmyn state. Enable “Deny network” to force offline operation."),
            )
            .into_any_element()
    }

    fn agent_section_prelude(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let show_default_button = TextButton::new(
            ("settings_show_default_prelude", cx.entity_id()),
            "Show default",
        )
        .kind(ButtonKind::Ghost)
        .small()
        .on_click({
            let root = root.clone();
            move |_, _, cx| {
                root.update(cx, |this, cx| {
                    this.settings_dialog.show_default_prelude =
                        !this.settings_dialog.show_default_prelude;
                    cx.notify();
                });
            }
        });

        let mut default_block = div().into_any_element();
        if self.show_default_prelude {
            let fill_button = TextButton::new(
                ("settings_fill_default_prelude", cx.entity_id()),
                "Fill as starting point",
            )
            .kind(ButtonKind::Secondary)
            .small()
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| {
                        this.settings_dialog.prelude_input.update(cx, |input, cx| {
                            input.set_text(BUILT_IN_PRELUDE_TEMPLATE, cx);
                        });
                        this.settings_dialog.draft.harness.prelude =
                            Some(BUILT_IN_PRELUDE_TEMPLATE.to_string());
                        this.settings_dialog.notice = None;
                        this.settings_dialog.save.clear_error();
                        cx.notify();
                    });
                }
            });

            default_block = div()
                .mt(theme.spacing.sm)
                .p(theme.spacing.md)
                .rounded(theme.radius.md)
                .bg(theme.colors.surface_elevated.opacity(0.35))
                .border_1()
                .border_color(theme.colors.border.opacity(0.6))
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
                                .child("Default prelude"),
                        )
                        .child(fill_button),
                )
                .child(
                    div()
                        .mt(theme.spacing.sm)
                        .p(theme.spacing.sm)
                        .rounded(theme.radius.sm)
                        .bg(theme.colors.background.opacity(0.25))
                        .border_1()
                        .border_color(theme.colors.border.opacity(0.55))
                        .child(
                            div()
                                .font(theme.typography.mono.font.clone())
                                .text_size(theme.typography.mono.size)
                                .text_color(theme.colors.foreground_muted)
                                .child(BUILT_IN_PRELUDE_TEMPLATE),
                        ),
                )
                .into_any_element();
        }

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Prelude"),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child("Agent prelude"),
                    )
                    .child(show_default_button),
            )
            .child(
                div()
                    .border_1()
                    .border_color(theme.colors.border.opacity(0.6))
                    .rounded(theme.radius.md)
                    .overflow_hidden()
                    .child(self.prelude_input.clone()),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Sent to the agent right after the harness starts. Use it to point the agent at relevant docs and guidance."),
            )
            .child(default_block)
            .child(self.prelude_delivery_settings(root, window, cx))
            .child(self.placeholder_reference(window, cx))
            .into_any_element()
    }

    fn prelude_delivery_settings(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let send_enabled = self.draft.harness.send_prelude;
        let submit_enabled = self.draft.harness.submit_prelude;

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Prelude delivery"),
            )
            .child(self.boolean_row(
                root,
                "auto_send_prelude",
                "Auto-send prelude",
                send_enabled,
                false,
                |draft, enabled| {
                    draft.harness.send_prelude = enabled;
                    if !enabled {
                        draft.harness.submit_prelude = false;
                    }
                },
                window,
                cx,
            ))
            .child(self.boolean_row(
                root,
                "press_enter_submit",
                "Press Enter to submit",
                submit_enabled,
                !send_enabled,
                |draft, enabled| {
                    draft.harness.submit_prelude = enabled;
                },
                window,
                cx,
            ))
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Auto-send types the prelude into the harness. Press Enter submits it so the agent starts working immediately."),
            )
            .into_any_element()
    }

    fn placeholder_reference(
        &mut self,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let placeholder_row = |key: &'static str, description: &'static str| {
            div()
                .flex()
                .flex_row()
                .items_start()
                .gap(theme.spacing.md)
                .child(
                    div()
                        .w(px(140.0))
                        .font(theme.typography.mono.font.clone())
                        .text_size(theme.typography.mono.size)
                        .text_color(theme.colors.foreground.opacity(0.8))
                        .child(key),
                )
                .child(
                    div()
                        .flex_1()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child(description),
                )
        };

        let placeholders = div()
            .mt(theme.spacing.sm)
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(placeholder_row("{task_id}", "Task numeric id."))
            .child(placeholder_row("{task_title}", "Task title."))
            .child(placeholder_row(
                "{task_doc}",
                "Task README path (if available).",
            ))
            .child(placeholder_row("{epic_slug}", "Epic slug."))
            .child(placeholder_row("{epic_readme}", "Epic README path."))
            .child(placeholder_row("{branch}", "Branch name for the task."))
            .child(placeholder_row(
                "{worktree}",
                "Absolute path to the task worktree.",
            ));

        div()
            .p(theme.spacing.md)
            .rounded(theme.radius.md)
            .bg(theme.colors.surface_elevated.opacity(0.25))
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Available placeholders"),
            )
            .child(placeholders)
            .into_any_element()
    }

    fn labeled_text_input(
        &mut self,
        label: &'static str,
        input: Entity<TextInput>,
        help: &'static str,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(label),
            )
            .child(
                div()
                    .border_1()
                    .border_color(theme.colors.border.opacity(0.6))
                    .rounded(theme.radius.md)
                    .overflow_hidden()
                    .child(input),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(help),
            )
            .into_any_element()
    }

    fn segmented_bool_setting(
        &mut self,
        root: &Entity<RootView>,
        label: &'static str,
        true_label: &'static str,
        false_label: &'static str,
        current: bool,
        apply: impl Fn(&mut OrchestrationDefaults, bool) + Copy + 'static,
        help: &'static str,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);
        let selected_true = current;
        let selected_false = !current;

        let button = |id: &'static str, label: &'static str, next: bool, selected: bool| {
            TextButton::new((id, cx.entity_id()), label)
                .kind(if selected {
                    ButtonKind::Secondary
                } else {
                    ButtonKind::Ghost
                })
                .small()
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| {
                            apply(&mut this.settings_dialog.draft, next);
                            if !this.settings_dialog.draft.harness.send_prelude {
                                this.settings_dialog.draft.harness.submit_prelude = false;
                            }
                            this.settings_dialog.notice = None;
                            this.settings_dialog.save.clear_error();
                            cx.notify();
                        });
                    }
                })
        };

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(label),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap(theme.spacing.xs)
                    .child(button("seg_true", true_label, true, selected_true))
                    .child(button("seg_false", false_label, false, selected_false)),
            )
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child(help),
            )
            .into_any_element()
    }

    fn boolean_row(
        &mut self,
        root: &Entity<RootView>,
        id: &'static str,
        label: &'static str,
        enabled: bool,
        disabled: bool,
        apply: impl Fn(&mut OrchestrationDefaults, bool) + Copy + 'static,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);
        let kind = if enabled {
            ButtonKind::Secondary
        } else {
            ButtonKind::Ghost
        };

        let mut toggle = TextButton::new((id, cx.entity_id()), if enabled { "On" } else { "Off" })
            .kind(kind)
            .small()
            .disabled(disabled);

        if disabled {
            toggle = toggle.disabled_reason("Disabled");
        }

        let toggle = toggle.on_click({
            let root = root.clone();
            move |_, _, cx| {
                root.update(cx, |this, cx| {
                    apply(&mut this.settings_dialog.draft, !enabled);
                    if !this.settings_dialog.draft.harness.send_prelude {
                        this.settings_dialog.draft.harness.submit_prelude = false;
                    }
                    this.settings_dialog.notice = None;
                    this.settings_dialog.save.clear_error();
                    cx.notify();
                });
            }
        });

        div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .text_color(if disabled {
                        theme.colors.foreground_muted
                    } else {
                        theme.colors.foreground
                    })
                    .child(label),
            )
            .child(toggle)
            .into_any_element()
    }

    fn agent_kind_button(
        &mut self,
        root: &Entity<RootView>,
        cx: &mut Context<RootView>,
        id: &'static str,
        label: &'static str,
        kind: AgentKindSelection,
    ) -> AnyElement {
        let selected = self.draft.harness.agent_kind == kind;
        TextButton::new((id, cx.entity_id()), label)
            .kind(if selected {
                ButtonKind::Secondary
            } else {
                ButtonKind::Ghost
            })
            .small()
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| {
                        this.settings_dialog.draft.harness.agent_kind = kind;
                        this.settings_dialog.notice = None;
                        this.settings_dialog.save.clear_error();
                        cx.notify();
                    });
                }
            })
            .into_any_element()
    }

    fn sandbox_type_button(
        &mut self,
        root: &Entity<RootView>,
        cx: &mut Context<RootView>,
        id: &'static str,
        label: &'static str,
        kind: SandboxType,
    ) -> AnyElement {
        let selected = self.draft.sandbox_type == kind;
        TextButton::new((id, cx.entity_id()), label)
            .kind(if selected {
                ButtonKind::Secondary
            } else {
                ButtonKind::Ghost
            })
            .small()
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| {
                        this.settings_dialog.draft.sandbox_type = kind;
                        if kind != SandboxType::Worktree {
                            this.settings_dialog.draft.sandbox_network = SandboxNetworkMode::Allow;
                        }
                        this.settings_dialog.notice = None;
                        this.settings_dialog.save.clear_error();
                        cx.notify();
                    });
                }
            })
            .into_any_element()
    }

    fn is_dirty(&self) -> bool {
        self.draft != self.baseline
    }

    fn reset(&mut self, cx: &mut Context<RootView>) {
        if self.save.in_flight || !self.is_dirty() {
            return;
        }

        self.draft = self.baseline.clone();
        self.show_default_prelude = false;
        self.notice = None;
        self.save.clear_error();

        let command = self.draft.harness.command.clone().unwrap_or_default();
        self.command_input
            .update(cx, move |input, cx| input.set_text(command, cx));

        let prelude = self.draft.harness.prelude.clone().unwrap_or_default();
        self.prelude_input
            .update(cx, move |input, cx| input.set_text(prelude, cx));

        cx.notify();
    }

    fn save(&mut self, cx: &mut Context<RootView>) {
        if self.save.in_flight || !self.is_dirty() {
            return;
        }

        let Some(repo_root) = self.repo_root.clone() else {
            self.save.fail("Repo root unavailable; reopen settings.");
            cx.notify();
            return;
        };

        let desired = self.draft.clone();
        self.save.start();
        self.notice = None;

        cx.spawn(move |root: WeakEntity<RootView>, cx: &mut gpui::AsyncApp| {
            let mut cx = cx.clone();
            async move {
                let result = write_repo_defaults(&repo_root, &desired);
                if let Some(root) = root.upgrade() {
                    let _ = root.update(&mut cx, |this, cx| match result {
                        Ok(()) => {
                            this.settings_dialog.baseline = desired;
                            this.settings_dialog.save.succeed();
                            this.settings_dialog.notice = Some("Saved".into());
                            cx.notify();
                        }
                        Err(err) => {
                            this.settings_dialog.save.fail(err.to_string());
                            cx.notify();
                        }
                    });
                }
            }
        })
        .detach();
    }
}
