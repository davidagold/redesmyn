use std::path::PathBuf;
use std::time::Duration;

use gpui::{
    AnyElement, Context, Entity, FocusHandle, FontWeight, MouseButton, ScrollHandle, SharedString,
    WeakEntity, Window, div, prelude::*, px,
};

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ScrollArea, TextArea, TextButton, TextInputEvent,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{
    TransitionMap, UserActionState, theme_for_window, ui_test_mode_animation_duration,
};

use crate::orchestration_config::{
    AgentKindSelection, CodexApprovalPolicyDefault, CodexSandboxPolicyDefault,
    OrchestrationConfigError, OrchestrationDefaults, SandboxNetworkMode, SandboxType,
    load_effective_defaults, repo_config_path, repo_root_from_cwd, write_repo_defaults,
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
    default_prelude_open: bool,
    prelude_input: Entity<TextArea>,
    save: UserActionState,
    notice: Option<SharedString>,
}

impl SettingsDialog {
    pub fn new(root_focus_handle: FocusHandle, cx: &mut Context<RootView>) -> Self {
        let prelude_input = cx.new(|cx| {
            TextArea::new(cx)
                .placeholder("Optional. Leave blank to use the built-in prelude.")
                .min_rows(4)
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
            default_prelude_open: false,
            prelude_input,
            save: UserActionState::default(),
            notice: None,
        }
    }

    pub fn prelude_input_entity(&self) -> Entity<TextArea> {
        self.prelude_input.clone()
    }

    pub fn dismiss_default_prelude_overlay(&mut self) -> bool {
        if !self.default_prelude_open {
            return false;
        }

        self.default_prelude_open = false;
        true
    }

    pub fn set_section_for_ui_driver(
        &mut self,
        section: redesmyn_protocol::ui_driver::SettingsDialogSection,
        cx: &mut Context<RootView>,
    ) {
        let next = match section {
            redesmyn_protocol::ui_driver::SettingsDialogSection::Appearance => {
                SettingsSection::Appearance
            }
            redesmyn_protocol::ui_driver::SettingsDialogSection::Agents => SettingsSection::Agents,
        };

        self.set_section(next, cx);
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

    fn set_section(&mut self, section: SettingsSection, cx: &mut Context<RootView>) {
        if self.section == section {
            return;
        }

        self.section = section;
        self.scroll.set_offset(gpui::point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub fn render(
        &mut self,
        root: &Entity<RootView>,
        visible: bool,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> Option<AnyElement> {
        let duration = ui_test_mode_animation_duration(Duration::from_millis(70));
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
                    .occlude()
                    .on_mouse_down(MouseButton::Left, close_on_background),
            )
            .child(self.dialog_card(root, window, cx));

        Some(overlay.into_any_element())
    }

    fn open(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        self.default_prelude_open = false;
        self.save = UserActionState::default();
        self.notice = None;
        self.scroll.set_offset(gpui::point(px(0.0), px(0.0)));

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
                            .child("Updates `config.toml` (repo scope)."),
                    ),
            )
            .child(header_right);

        let body = div()
            .flex()
            .flex_row()
            .flex_1()
            .min_h(px(0.0))
            .child(self.section_nav(root, window, cx))
            .child(self.section_body(root, window, cx));

        let mut card = div()
            .key_context("SettingsDialog")
            .flex()
            .flex_col()
            .relative()
            .w_full()
            .max_w(px(960.0))
            .h_full()
            .max_h(px(680.0))
            .rounded(theme.radius.xl)
            .bg(theme.colors.surface)
            .border_1()
            .border_color(theme.colors.border.opacity(0.7))
            .shadow_lg()
            .occlude()
            .overflow_hidden()
            .child(header)
            .child(body);

        if self.default_prelude_open {
            card = card.child(self.default_prelude_overlay(root, window, cx));
        }

        div()
            .absolute()
            .inset_0()
            .p(theme.spacing.xl)
            .flex()
            .items_center()
            .justify_center()
            .child(card)
            .into_any_element()
    }

    fn default_prelude_overlay(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let close_on_background = {
            let root = root.clone();
            move |_: &gpui::MouseDownEvent, _: &mut Window, cx: &mut gpui::App| {
                root.update(cx, |this, cx| {
                    this.settings_dialog.default_prelude_open = false;
                    cx.notify();
                });
                cx.stop_propagation();
            }
        };

        let close_on_click = {
            let root = root.clone();
            move |_: &gpui::ClickEvent, _: &mut Window, cx: &mut gpui::App| {
                root.update(cx, |this, cx| {
                    this.settings_dialog.default_prelude_open = false;
                    cx.notify();
                });
                cx.stop_propagation();
            }
        };

        let fill_button = TextButton::new(
            ("settings_fill_default_prelude", cx.entity_id()),
            "Fill as starting point",
        )
        .kind(ButtonKind::Secondary)
        .small()
        .on_click({
            let root = root.clone();
            move |_: &gpui::ClickEvent, _: &mut Window, cx: &mut gpui::App| {
                root.update(cx, |this, cx| {
                    this.settings_dialog.prelude_input.update(cx, |input, cx| {
                        input.set_text(BUILT_IN_PRELUDE_TEMPLATE, cx);
                    });
                    this.settings_dialog.draft.harness.prelude =
                        Some(BUILT_IN_PRELUDE_TEMPLATE.to_string());
                    this.settings_dialog.default_prelude_open = false;
                    this.settings_dialog.notice = None;
                    this.settings_dialog.save.clear_error();
                    cx.notify();
                });
                cx.stop_propagation();
            }
        });

        let close_button =
            IconButton::new(("settings_close_default_prelude", cx.entity_id()), div().child("×"))
                .tooltip("Close")
                .on_click(close_on_click);

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
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Default prelude"),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .child(fill_button)
                    .child(close_button),
            );

        let body_scroll = ScrollArea::new(
            ("settings_default_prelude_scroll", cx.entity_id()),
            ScrollHandle::new(),
        )
        .bg(theme.colors.background.opacity(0.25))
        .child(
            div()
                .p(theme.spacing.lg)
                .font(theme.typography.mono.font.clone())
                .text_size(theme.typography.mono.size)
                .text_color(theme.colors.foreground_muted)
                .child(BUILT_IN_PRELUDE_TEMPLATE),
        );

        let panel = div()
            .w_full()
            .max_w(px(780.0))
            .h_full()
            .max_h(px(560.0))
            .bg(theme.colors.surface_elevated)
            .border_1()
            .border_color(theme.colors.border.opacity(0.8))
            .rounded(theme.radius.lg)
            .shadow_lg()
            .overflow_hidden()
            .occlude()
            .flex()
            .flex_col()
            .child(header)
            .child(div().flex_1().min_h(px(0.0)).child(body_scroll));

        div()
            .absolute()
            .inset_0()
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .absolute()
                    .inset_0()
                    .bg(theme.colors.background.opacity(0.55))
                    .occlude()
                    .on_mouse_down(MouseButton::Left, close_on_background),
            )
            .child(panel)
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
            TextButton::new((id, cx.entity_id()), label)
                .kind(if selected {
                    ButtonKind::Secondary
                } else {
                    ButtonKind::Ghost
                })
                .small()
                .menu_item()
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| {
                            this.settings_dialog.set_section(section, cx);
                        });
                    }
                })
        };

        div()
            .w(px(220.0))
            .h_full()
            .p(theme.spacing.md)
            .border_r_1()
            .border_color(theme.colors.border.opacity(0.6))
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
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

        div()
            .flex_1()
            .min_h(px(0.0))
            .min_w(px(0.0))
            .child(
                ScrollArea::new(("settings_scroll", cx.entity_id()), self.scroll.clone())
                    .bg(theme.colors.surface)
                    .child(content),
            )
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

        let theme_choices = [
            ("theme_light", "Light", ThemePreference::Light),
            ("theme_dark", "Dark", ThemePreference::Dark),
            ("theme_system", "System", ThemePreference::System),
        ];

        let mut choices = div()
            .flex()
            .flex_row()
            .rounded(theme.radius.md)
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .overflow_hidden();

        for (idx, (id, label, pref)) in theme_choices.into_iter().enumerate() {
            let selected = current == pref;
            let mut cell = div()
                .id((id, cx.entity_id()))
                .px(theme.spacing.sm)
                .py(theme.spacing.xs)
                .text_xs()
                .text_color(theme.colors.foreground)
                .when(selected, |this| this.bg(theme.colors.accent))
                .when(!selected, |this| {
                    this.hover(|this| this.bg(theme.colors.surface_elevated.opacity(0.35)))
                })
                .cursor_pointer()
                .child(label);

                cell = cell.on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.set_theme_preference(pref, cx));
                    }
                });

            if idx > 0 {
                choices = choices.child(
                    div()
                        .border_l_1()
                        .border_color(theme.colors.border.opacity(0.6)),
                );
            }
            choices = choices.child(cell);
        }

        div()
            .flex()
            .flex_col()
            .items_start()
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
            .child(choices)
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
        let agent_section = self.agent_section_agent(root, window, cx);
        let prelude_section = self.agent_section_prelude(root, window, cx);

        body = body
            .child(
                div()
                    .text_lg()
                    .text_color(theme.colors.foreground)
                    .child("Agents"),
            )
            .child(self.section_card("Agent", agent_section, window, cx))
            .child(self.section_card("Prelude", prelude_section, window, cx));

        body.into_any_element()
    }

    fn section_card(
        &mut self,
        title: &'static str,
        body: AnyElement,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        div()
            .p(theme.spacing.lg)
            .rounded(theme.radius.lg)
            .bg(theme.colors.surface_elevated.opacity(0.2))
            .flex()
            .flex_col()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.colors.foreground)
                    .child(title),
            )
            .child(body)
            .into_any_element()
    }

    fn agent_section_agent(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let mut body = div()
            .flex()
            .flex_col()
            .gap(theme.spacing.lg)
            .child(self.segmented_choice_setting(
                root,
                "agent_kind",
                "Agent",
                &[
                    ("codex", "Codex", AgentKindSelection::Codex),
                    ("claude_code", "Claude Code", AgentKindSelection::ClaudeCode),
                ],
                self.draft.harness.agent_kind,
                |draft, next| draft.harness.agent_kind = next,
                "Choose the agent. Redesmyn will select the best backend for structured operation.",
                window,
                cx,
            ))
            .child(self.sandbox_settings(root, window, cx));

        if self.draft.harness.agent_kind == AgentKindSelection::Codex {
            body = body.child(self.codex_session_defaults_settings(root, window, cx));
        } else {
            body = body.child(
                div()
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child("Session defaults are currently only configurable for Codex."),
            );
        }

        body.into_any_element()
    }

    fn sandbox_settings(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let deny_network_disabled = self.draft.sandbox_type != SandboxType::Worktree;

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.colors.foreground)
                    .child("Worktree sandbox"),
            )
            .child(self.segmented_choice_setting(
                root,
                "sandbox_type",
                "Type",
                &[
                    ("off", "Off", SandboxType::None),
                    ("worktree", "Worktree", SandboxType::Worktree),
                ],
                self.draft.sandbox_type,
                |draft, next| {
                    draft.sandbox_type = next;
                    if next != SandboxType::Worktree {
                        draft.sandbox_network = SandboxNetworkMode::Allow;
                    }
                },
                "Restricts agent writes to the task worktree and Redesmyn state.",
                window,
                cx,
            ))
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
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child("Enable “Deny network” to force offline operation."),
            )
            .into_any_element()
    }

    fn codex_session_defaults_settings(
        &mut self,
        root: &Entity<RootView>,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.colors.foreground)
                    .child("Session defaults"),
            )
            .child(self.segmented_choice_setting(
                root,
                "codex_approval_policy_default",
                "Permissions",
                &[
                    ("default", "Default", CodexApprovalPolicyDefault::Default),
                    (
                        "unless_trusted",
                        "Unless trusted",
                        CodexApprovalPolicyDefault::UnlessTrusted,
                    ),
                    ("on_request", "On request", CodexApprovalPolicyDefault::OnRequest),
                    ("on_failure", "On failure", CodexApprovalPolicyDefault::OnFailure),
                    ("never", "Never", CodexApprovalPolicyDefault::Never),
                ],
                self.draft.session_defaults.codex.approval_policy,
                |draft, next| draft.session_defaults.codex.approval_policy = next,
                "Default Codex approvals policy for new sessions.",
                window,
                cx,
            ))
            .child(self.segmented_choice_setting(
                root,
                "codex_sandbox_policy_default",
                "Sandbox",
                &[
                    ("default", "Default", CodexSandboxPolicyDefault::Default),
                    ("read_only", "Read-only", CodexSandboxPolicyDefault::ReadOnly),
                    (
                        "workspace_write",
                        "Workspace write",
                        CodexSandboxPolicyDefault::WorkspaceWrite,
                    ),
                    (
                        "danger_full_access",
                        "Danger: full access",
                        CodexSandboxPolicyDefault::DangerFullAccess,
                    ),
                ],
                self.draft.session_defaults.codex.sandbox_policy,
                |draft, next| draft.session_defaults.codex.sandbox_policy = next,
                "Default Codex sandbox policy for new sessions.",
                window,
                cx,
            ))
            .child(
                div()
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child("You can override these per session from the composer Settings menu."),
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
            if self.default_prelude_open {
                "Hide default"
            } else {
                "View default"
            },
        )
        .kind(ButtonKind::Ghost)
        .small()
        .on_click({
            let root = root.clone();
            move |_, _, cx| {
                root.update(cx, |this, cx| {
                    this.settings_dialog.default_prelude_open =
                        !this.settings_dialog.default_prelude_open;
                    cx.notify();
                });
            }
        });

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.lg)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.colors.foreground_muted)
                            .child("Agent prelude"),
                    )
                    .child(show_default_button),
            )
            .child(
                div()
                    .font(theme.typography.mono.font.clone())
                    .text_size(theme.typography.mono.size)
                    .child(self.prelude_input.clone()),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child("Sent to the agent right after the harness starts. Use it to point the agent at relevant docs and guidance."),
            )
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
            .gap(theme.spacing.md)
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.colors.foreground)
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
                    .text_xs()
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
                        .text_xs()
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

        let card = div()
            .p(theme.spacing.md)
            .rounded(theme.radius.md)
            .bg(theme.colors.surface_elevated.opacity(0.25))
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(theme.colors.foreground_muted)
                    .child("Available placeholders"),
            )
            .child(placeholders)
            .into_any_element();

        div()
            .flex()
            .flex_row()
            .child(card)
            .into_any_element()
    }

    fn segmented_choice_setting<T: Copy + PartialEq + 'static>(
        &mut self,
        root: &Entity<RootView>,
        id: &'static str,
        label: &'static str,
        choices: &[(&'static str, &'static str, T)],
        current: T,
        apply: impl Fn(&mut OrchestrationDefaults, T) + Copy + 'static,
        help: &'static str,
        window: &mut Window,
        cx: &mut Context<RootView>,
    ) -> AnyElement {
        let theme = theme_for_window(window, cx);

        let mut group = div()
            .flex()
            .flex_row()
            .rounded(theme.radius.md)
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .overflow_hidden();

        for (idx, (suffix, segment_label, value)) in choices.iter().enumerate() {
            let selected = current == *value;
            let mut cell = div()
                .id((gpui::ElementId::from((id, cx.entity_id())), *suffix))
                .px(theme.spacing.sm)
                .py(theme.spacing.xs)
                .text_xs()
                .text_color(theme.colors.foreground)
                .when(selected, |this| this.bg(theme.colors.accent))
                .when(!selected, |this| {
                    this.hover(|this| this.bg(theme.colors.surface_elevated.opacity(0.35)))
                })
                .cursor_pointer()
                .child(*segment_label);

            cell = cell.on_click({
                let root = root.clone();
                let value = *value;
                move |_, _, cx| {
                    root.update(cx, |this, cx| {
                        apply(&mut this.settings_dialog.draft, value);
                        if !this.settings_dialog.draft.harness.send_prelude {
                            this.settings_dialog.draft.harness.submit_prelude = false;
                        }
                        this.settings_dialog.notice = None;
                        this.settings_dialog.save.clear_error();
                        cx.notify();
                    });
                }
            });

            if idx > 0 {
                group = group.child(
                    div()
                        .border_l_1()
                        .border_color(theme.colors.border.opacity(0.6)),
                );
            }

            group = group.child(cell);
        }

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .justify_between()
                    .gap(theme.spacing.md)
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child(label),
                    )
                    .child(group),
            )
            .child(
                div()
                    .text_xs()
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

        let segments = [("off", false, "Off"), ("on", true, "On")];

        let mut group = div()
            .flex()
            .flex_row()
            .rounded(theme.radius.md)
            .border_1()
            .border_color(theme.colors.border.opacity(0.6))
            .overflow_hidden();

        for (idx, (suffix, value, segment_label)) in segments.into_iter().enumerate() {
            let selected = enabled == value;
            let mut cell = div()
                .id((gpui::ElementId::from((id, cx.entity_id())), suffix))
                .px(theme.spacing.sm)
                .py(theme.spacing.xs)
                .text_xs()
                .text_color(theme.colors.foreground)
                .when(selected, |this| this.bg(theme.colors.accent))
                .when(!selected, |this| {
                    this.hover(|this| this.bg(theme.colors.surface_elevated.opacity(0.35)))
                })
                .when(disabled, |this| this.opacity(0.55).cursor_not_allowed())
                .when(!disabled, |this| this.cursor_pointer())
                .child(segment_label);

            if !disabled {
                cell = cell.on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| {
                            apply(&mut this.settings_dialog.draft, value);
                            if !this.settings_dialog.draft.harness.send_prelude {
                                this.settings_dialog.draft.harness.submit_prelude = false;
                            }
                            this.settings_dialog.notice = None;
                            this.settings_dialog.save.clear_error();
                            cx.notify();
                        });
                    }
                });
            }

            if idx > 0 {
                group = group.child(
                    div()
                        .border_l_1()
                        .border_color(theme.colors.border.opacity(0.6)),
                );
            }
            group = group.child(cell);
        }

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
            .child(group)
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
        self.default_prelude_open = false;
        self.notice = None;
        self.save.clear_error();

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
