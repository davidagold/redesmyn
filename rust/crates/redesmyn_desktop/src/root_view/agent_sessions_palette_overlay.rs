use std::collections::{BTreeMap, HashSet};

use gpui::{
    App, Context, ElementId, Entity, FocusHandle, Focusable, ScrollHandle, SharedString, Window,
    div, point, prelude::*, px,
};

use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::{RepoScope, Timestamp};
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ProgressPillKind, ScrollArea,
    TextButton, TextInput, TextInputEvent,
};
use redesmyn_ui::styles::UiTheme;
use redesmyn_ui::utils::theme_for_window;

use super::RootView;

const RECENT_SESSIONS_LIMIT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentSessionKind {
    Task,
    Chat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSessionNavigation {
    Task {
        epic_slug: String,
        task_id: TaskId,
    },
    Chat {
        epic_slug: String,
        session_id: SessionId,
    },
    Disabled {
        reason: SharedString,
    },
}

#[derive(Debug, Clone)]
pub struct AgentSessionPaletteEntry {
    pub session_id: SessionId,
    pub kind: AgentSessionKind,
    pub repo_scope: Option<RepoScope>,
    pub repo_group_label: String,
    pub epic_slug: Option<String>,
    pub epic_title: Option<String>,
    pub task_slug: Option<String>,
    pub task_title: Option<String>,
    pub summary: Option<String>,
    pub last_activity: Option<Timestamp>,
    pub is_active: bool,
    pub navigation: AgentSessionNavigation,
}

impl AgentSessionPaletteEntry {
    fn primary_label(&self) -> String {
        match self.kind {
            AgentSessionKind::Task => self
                .task_title
                .clone()
                .or_else(|| self.task_slug.clone())
                .or_else(|| self.summary.clone())
                .unwrap_or_else(|| "Task session".to_string()),
            AgentSessionKind::Chat => self
                .summary
                .clone()
                .or_else(|| self.epic_title.clone())
                .unwrap_or_else(|| "Chat session".to_string()),
        }
    }

    fn subtitle(&self) -> String {
        let mut parts = Vec::new();

        if let Some(epic_slug) = self.epic_slug.as_ref() {
            parts.push(epic_slug.clone());
        }

        if let Some(task_slug) = self.task_slug.as_ref() {
            parts.push(task_slug.clone());
        }

        if parts.is_empty() {
            "No epic binding".to_string()
        } else {
            parts.join(" • ")
        }
    }

    fn group_epic_label(&self) -> String {
        self.epic_slug
            .clone()
            .unwrap_or_else(|| "No epic".to_string())
    }

    fn kind_label(&self) -> &'static str {
        match self.kind {
            AgentSessionKind::Task => "Task",
            AgentSessionKind::Chat => "Chat",
        }
    }

    fn search_matches(&self, query: &str) -> bool {
        if query.is_empty() {
            return true;
        }

        let mut haystacks = vec![
            self.repo_group_label.clone(),
            self.primary_label(),
            self.subtitle(),
            self.kind_label().to_string(),
            self.session_id.to_string(),
        ];

        if let Some(epic_slug) = self.epic_slug.as_ref() {
            haystacks.push(epic_slug.clone());
        }
        if let Some(epic_title) = self.epic_title.as_ref() {
            haystacks.push(epic_title.clone());
        }
        if let Some(task_slug) = self.task_slug.as_ref() {
            haystacks.push(task_slug.clone());
        }
        if let Some(task_title) = self.task_title.as_ref() {
            haystacks.push(task_title.clone());
        }
        if let Some(summary) = self.summary.as_ref() {
            haystacks.push(summary.clone());
        }

        haystacks
            .into_iter()
            .any(|haystack| haystack.to_ascii_lowercase().contains(query))
    }
}

#[derive(Debug)]
struct GroupedSection {
    repo_label: String,
    epic_label: String,
    indices: Vec<usize>,
}

#[derive(Debug, Default)]
struct DisplayRows {
    ordered_indices: Vec<usize>,
    recent_indices: Vec<usize>,
    grouped_sections: Vec<GroupedSection>,
}

pub struct AgentSessionsPaletteOverlay {
    root_focus_handle: FocusHandle,
    open: bool,
    input: Entity<TextInput>,
    scroll: ScrollHandle,
    selected_index: usize,
    loading: bool,
    action_in_flight: bool,
    archiving_session_id: Option<SessionId>,
    show_unreachable_entries: bool,
    error: Option<SharedString>,
    entries: Vec<AgentSessionPaletteEntry>,
}

impl AgentSessionsPaletteOverlay {
    pub fn new(root_focus_handle: FocusHandle, cx: &mut Context<RootView>) -> Self {
        let input = cx.new(|cx| TextInput::new(cx).placeholder("Filter sessions…"));
        Self {
            root_focus_handle,
            open: false,
            input,
            scroll: ScrollHandle::new(),
            selected_index: 0,
            loading: false,
            action_in_flight: false,
            archiving_session_id: None,
            show_unreachable_entries: false,
            error: None,
            entries: Vec::new(),
        }
    }

    pub fn input_entity(&self) -> Entity<TextInput> {
        self.input.clone()
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    pub fn toggle(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        if self.open {
            self.close_palette(window, cx);
        } else {
            self.open_palette(window, cx);
        }
    }

    pub fn start_loading(&mut self, cx: &mut Context<RootView>) {
        self.loading = true;
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.error = None;
        self.selected_index = 0;
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub fn finish_loading(
        &mut self,
        entries: Vec<AgentSessionPaletteEntry>,
        cx: &mut Context<RootView>,
    ) {
        self.entries = entries;
        self.loading = false;
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.error = None;
        self.selected_index = 0;
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub fn fail_loading(&mut self, message: impl Into<SharedString>, cx: &mut Context<RootView>) {
        self.loading = false;
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.error = Some(message.into());
        self.selected_index = 0;
        cx.notify();
    }

    pub fn handle_text_input_event(
        &mut self,
        event: TextInputEvent,
        cx: &mut Context<RootView>,
    ) -> Option<AgentSessionPaletteEntry> {
        match event {
            TextInputEvent::Changed(_) => {
                self.selected_index = 0;
                self.error = None;
                self.scroll.set_offset(point(px(0.0), px(0.0)));
                cx.notify();
                None
            }
            TextInputEvent::Submitted(_) => self.activate_selected(cx),
            TextInputEvent::PastedImages(_) => None,
        }
    }

    pub fn activate_selected_entry(
        &mut self,
        cx: &mut Context<RootView>,
    ) -> Option<AgentSessionPaletteEntry> {
        self.activate_selected(cx)
    }

    pub fn set_show_unreachable_entries(&mut self, show: bool, cx: &mut Context<RootView>) {
        if self.show_unreachable_entries == show {
            return;
        }
        self.show_unreachable_entries = show;
        self.selected_index = 0;
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub fn handle_close_action(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        if !self.open {
            return;
        }
        self.close_palette(window, cx);
    }

    pub fn select_previous(&mut self, cx: &mut Context<RootView>) {
        if !self.open || self.is_blocking_loading() || self.action_in_flight {
            return;
        }

        let display = self.display_rows(cx);
        if display.ordered_indices.is_empty() {
            return;
        }

        if self.selected_index == 0 {
            self.selected_index = display.ordered_indices.len() - 1;
        } else {
            self.selected_index = self.selected_index.saturating_sub(1);
        }

        self.scroll_selected_into_view(cx);
        cx.notify();
    }

    pub fn select_next(&mut self, cx: &mut Context<RootView>) {
        if !self.open || self.is_blocking_loading() || self.action_in_flight {
            return;
        }

        let display = self.display_rows(cx);
        if display.ordered_indices.is_empty() {
            return;
        }

        let next = self.selected_index.saturating_add(1);
        self.selected_index = if next >= display.ordered_indices.len() {
            0
        } else {
            next
        };

        self.scroll_selected_into_view(cx);
        cx.notify();
    }

    pub fn set_selected_display_index(&mut self, display_index: usize, cx: &mut Context<RootView>) {
        if !self.open || self.is_blocking_loading() || self.action_in_flight {
            return;
        }

        let display = self.display_rows(cx);
        if display.ordered_indices.is_empty() {
            return;
        }

        let clamped = display_index.min(display.ordered_indices.len().saturating_sub(1));
        if self.selected_index == clamped {
            return;
        }

        self.selected_index = clamped;
        cx.notify();
    }

    pub fn activate_entry_by_session_id(
        &mut self,
        session_id: SessionId,
        cx: &mut Context<RootView>,
    ) -> Option<AgentSessionPaletteEntry> {
        if self.is_blocking_loading() || self.action_in_flight {
            return None;
        }

        let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.session_id == session_id)
            .cloned()
        else {
            return None;
        };

        self.begin_activation(entry, cx)
    }

    pub fn complete_activation_success(&mut self, cx: &mut Context<RootView>) {
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.dismiss_overlay(cx);
    }

    pub fn complete_activation_failure(
        &mut self,
        message: impl Into<SharedString>,
        cx: &mut Context<RootView>,
    ) {
        self.action_in_flight = false;
        self.error = Some(message.into());
        cx.notify();
    }

    pub fn dismiss_overlay(&mut self, cx: &mut Context<RootView>) {
        self.open = false;
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.error = None;
        self.selected_index = 0;
        self.input.update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    pub fn start_archiving(&mut self, session_id: SessionId, cx: &mut Context<RootView>) {
        self.action_in_flight = true;
        self.archiving_session_id = Some(session_id);
        self.error = None;
        cx.notify();
    }

    pub fn finish_archiving(&mut self, cx: &mut Context<RootView>) {
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.error = None;
        cx.notify();
    }

    pub fn fail_archiving(&mut self, message: impl Into<SharedString>, cx: &mut Context<RootView>) {
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.error = Some(message.into());
        cx.notify();
    }

    pub fn render(&self, window: &mut Window, cx: &mut Context<RootView>) -> impl IntoElement {
        self.render_overlay(window, cx)
    }

    fn begin_activation(
        &mut self,
        entry: AgentSessionPaletteEntry,
        cx: &mut Context<RootView>,
    ) -> Option<AgentSessionPaletteEntry> {
        match entry.navigation {
            AgentSessionNavigation::Disabled { ref reason } => {
                self.error = Some(reason.clone());
                cx.notify();
                None
            }
            AgentSessionNavigation::Task { .. } | AgentSessionNavigation::Chat { .. } => {
                self.error = None;
                self.action_in_flight = true;
                self.archiving_session_id = None;
                cx.notify();
                Some(entry)
            }
        }
    }

    fn activate_selected(
        &mut self,
        cx: &mut Context<RootView>,
    ) -> Option<AgentSessionPaletteEntry> {
        if !self.open || self.is_blocking_loading() || self.action_in_flight {
            return None;
        }

        let display = self.display_rows(cx);
        let selected_index = self
            .selected_index
            .min(display.ordered_indices.len().saturating_sub(1));
        let Some(entry_index) = display.ordered_indices.get(selected_index).copied() else {
            self.error = Some("No matching sessions.".into());
            cx.notify();
            return None;
        };

        let Some(entry) = self.entries.get(entry_index).cloned() else {
            self.error = Some("Unable to resolve selected session.".into());
            cx.notify();
            return None;
        };

        self.begin_activation(entry, cx)
    }

    fn open_palette(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        self.open = true;
        self.selected_index = 0;
        self.error = None;
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.input.update(cx, |input, cx| input.set_text("", cx));
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        window.focus(&self.input.focus_handle(cx));
        cx.notify();
    }

    fn is_blocking_loading(&self) -> bool {
        self.loading && self.entries.is_empty()
    }

    fn close_palette(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        self.open = false;
        self.selected_index = 0;
        self.error = None;
        self.action_in_flight = false;
        self.archiving_session_id = None;
        self.input.update(cx, |input, cx| input.set_text("", cx));
        window.focus(&self.root_focus_handle);
        cx.notify();
    }

    fn filtered_indices(&self, cx: &App) -> Vec<usize> {
        let query = self.input.read(cx).text().to_ascii_lowercase();
        let query = query.trim();

        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                self.show_unreachable_entries
                    || !matches!(entry.navigation, AgentSessionNavigation::Disabled { .. })
            })
            .filter(|(_, entry)| entry.search_matches(query))
            .map(|(index, _)| index)
            .collect()
    }

    fn display_rows(&self, cx: &App) -> DisplayRows {
        let mut filtered = self.filtered_indices(cx);
        filtered.sort_by(|left, right| {
            cmp_last_activity_desc(
                self.entries
                    .get(*left)
                    .and_then(|entry| entry.last_activity),
                self.entries
                    .get(*right)
                    .and_then(|entry| entry.last_activity),
            )
            .then_with(|| {
                self.entries[*left]
                    .session_id
                    .to_string()
                    .cmp(&self.entries[*right].session_id.to_string())
            })
        });

        let mut recent_indices: Vec<usize> = filtered
            .iter()
            .copied()
            .filter(|index| {
                self.entries
                    .get(*index)
                    .is_some_and(|entry| entry.is_active)
            })
            .collect();
        recent_indices.truncate(RECENT_SESSIONS_LIMIT);

        let recent_session_ids: HashSet<SessionId> = recent_indices
            .iter()
            .filter_map(|index| self.entries.get(*index))
            .map(|entry| entry.session_id)
            .collect();

        let mut grouped: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
        for index in filtered {
            let Some(entry) = self.entries.get(index) else {
                continue;
            };
            if recent_session_ids.contains(&entry.session_id) {
                continue;
            }

            grouped
                .entry((entry.repo_group_label.clone(), entry.group_epic_label()))
                .or_default()
                .push(index);
        }

        let mut grouped_sections = Vec::new();
        for ((repo_label, epic_label), mut indices) in grouped {
            indices.sort_by(|left, right| {
                cmp_last_activity_desc(
                    self.entries
                        .get(*left)
                        .and_then(|entry| entry.last_activity),
                    self.entries
                        .get(*right)
                        .and_then(|entry| entry.last_activity),
                )
                .then_with(|| {
                    self.entries[*left]
                        .session_id
                        .to_string()
                        .cmp(&self.entries[*right].session_id.to_string())
                })
            });

            grouped_sections.push(GroupedSection {
                repo_label,
                epic_label,
                indices,
            });
        }

        let mut ordered_indices = recent_indices.clone();
        for section in &grouped_sections {
            ordered_indices.extend(section.indices.iter().copied());
        }

        DisplayRows {
            ordered_indices,
            recent_indices,
            grouped_sections,
        }
    }

    fn scroll_selected_into_view(&self, cx: &App) {
        let display = self.display_rows(cx);
        let Some(child_index) = self.scroll_child_index_for_selected(&display) else {
            return;
        };
        self.scroll.scroll_to_item(child_index);
    }

    fn scroll_child_index_for_selected(&self, display: &DisplayRows) -> Option<usize> {
        if display.ordered_indices.is_empty() {
            return None;
        }

        let selected_display_index = self
            .selected_index
            .min(display.ordered_indices.len().saturating_sub(1));
        let mut display_index = 0usize;
        let mut child_index = 0usize;

        if !display.recent_indices.is_empty() {
            child_index += 1;
            for _ in &display.recent_indices {
                if display_index == selected_display_index {
                    return Some(child_index);
                }
                display_index += 1;
                child_index += 1;
            }
        }

        for section in &display.grouped_sections {
            child_index += 1;
            for _ in &section.indices {
                if display_index == selected_display_index {
                    return Some(child_index);
                }
                display_index += 1;
                child_index += 1;
            }
        }

        None
    }

    fn render_overlay(&self, window: &mut Window, cx: &mut Context<RootView>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let root = cx.entity();
        let display = self.display_rows(cx);
        let selected_index = self
            .selected_index
            .min(display.ordered_indices.len().saturating_sub(1));

        let left_header = div()
            .flex()
            .flex_col()
            .gap(theme.spacing.xs)
            .child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Agent sessions"),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child("Type to filter, ↑/↓ to select, Enter to open."),
            );

        let mut right_header = div().flex().flex_row().items_center().gap(theme.spacing.sm);

        if self.loading && self.entries.is_empty() {
            right_header = right_header
                .child(ProgressPill::new("Loading sessions…").kind(ProgressPillKind::Accent));
        } else if self.loading {
            right_header = right_header
                .child(ProgressPill::new("Refreshing sessions…").kind(ProgressPillKind::Accent));
        } else if self.action_in_flight {
            let label = if self.archiving_session_id.is_some() {
                "Archiving session…"
            } else {
                "Opening session…"
            };
            right_header =
                right_header.child(ProgressPill::new(label).kind(ProgressPillKind::Accent));
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

        if let Some(error) = self.error.as_ref() {
            palette_body = palette_body.child(
                Callout::new(error.clone())
                    .kind(CalloutKind::Danger)
                    .title("Session palette"),
            );
        }

        palette_body = palette_body.child(self.input.clone());

        let mut list = ScrollArea::new(
            ("agent_sessions_palette_list", cx.entity_id()),
            self.scroll.clone(),
        )
        .scrollbar_width(px(10.0));

        if self.loading && self.entries.is_empty() {
            list = list.child(
                div()
                    .py(theme.spacing.sm)
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Loading sessions…"),
            );
        } else if display.ordered_indices.is_empty() {
            list = list.child(
                div()
                    .py(theme.spacing.sm)
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("No sessions match."),
            );
        } else {
            if !display.recent_indices.is_empty() {
                list = list.child(
                    div()
                        .pt(theme.spacing.sm)
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(theme.spacing.xs)
                        .text_xs()
                        .text_color(theme.colors.foreground_muted)
                        .child(div().text_xs().child("◷"))
                        .child(div().text_xs().child("Recent sessions")),
                );

                for entry_index in &display.recent_indices {
                    let Some(entry) = self.entries.get(*entry_index) else {
                        continue;
                    };
                    let row_display_index = display
                        .ordered_indices
                        .iter()
                        .position(|index| index == entry_index)
                        .unwrap_or(0);

                    list = list.child(render_entry_row(
                        entry,
                        row_display_index == selected_index,
                        row_display_index,
                        true,
                        self.is_blocking_loading() || self.action_in_flight,
                        self.archiving_session_id,
                        &root,
                        cx,
                        &theme,
                    ));
                }
            }

            for section in &display.grouped_sections {
                list = list.child(
                    div()
                        .pt(theme.spacing.sm)
                        .text_xs()
                        .text_color(theme.colors.foreground_muted)
                        .child(format!("{} • {}", section.repo_label, section.epic_label)),
                );

                for entry_index in &section.indices {
                    let Some(entry) = self.entries.get(*entry_index) else {
                        continue;
                    };
                    let row_display_index = display
                        .ordered_indices
                        .iter()
                        .position(|index| index == entry_index)
                        .unwrap_or(0);

                    list = list.child(render_entry_row(
                        entry,
                        row_display_index == selected_index,
                        row_display_index,
                        false,
                        self.is_blocking_loading() || self.action_in_flight,
                        self.archiving_session_id,
                        &root,
                        cx,
                        &theme,
                    ));
                }
            }
        }

        palette_body = palette_body.child(div().max_h(px(420.0)).child(list));

        let close_button = IconButton::new(
            ("agent_sessions_palette_close", cx.entity_id()),
            div().child("×"),
        )
        .tooltip("Close (Esc)")
        .on_click({
            let root = root.clone();
            move |_, window, cx| {
                let focus = root.read(cx).focus_handle.clone();
                root.update(cx, |this, cx| {
                    this.agent_sessions_palette.dismiss_overlay(cx);
                    this.ui_updates.bump();
                });
                window.focus(&focus);
            }
        });

        let palette_card = div()
            .key_context("AgentSessionsPalette")
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
                    .id(("agent_sessions_palette_backdrop", cx.entity_id()))
                    .cursor_pointer()
                    .on_scroll_wheel(|_, _, cx| cx.stop_propagation())
                    .on_click({
                        let root = root.clone();
                        move |_, window, cx| {
                            let focus = root.read(cx).focus_handle.clone();
                            root.update(cx, |this, cx| {
                                this.agent_sessions_palette.dismiss_overlay(cx);
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
                    .pt(px(104.0))
                    .child(palette_card),
            )
    }
}

fn render_entry_row(
    entry: &AgentSessionPaletteEntry,
    selected: bool,
    display_index: usize,
    show_repo_prefix: bool,
    globally_disabled: bool,
    archiving_session_id: Option<SessionId>,
    root: &Entity<RootView>,
    cx: &mut Context<RootView>,
    theme: &UiTheme,
) -> impl IntoElement {
    let disabled_reason = match &entry.navigation {
        AgentSessionNavigation::Disabled { reason } => Some(reason.clone()),
        AgentSessionNavigation::Task { .. } | AgentSessionNavigation::Chat { .. } => None,
    };
    let navigation_disabled = disabled_reason.is_some();
    let subtitle = if show_repo_prefix {
        let base = entry.subtitle();
        format!("{} • {base}", entry.repo_group_label)
    } else {
        entry.subtitle()
    };

    let base_row_id = ElementId::from(("agent_session_row", cx.entity_id()));
    let mut row = div()
        .id((base_row_id, entry.session_id.to_string()))
        .flex()
        .flex_col()
        .gap(px(2.0))
        .px(theme.spacing.sm)
        .py(theme.spacing.xs)
        .rounded(theme.radius.sm)
        .border_1()
        .border_color(theme.colors.border.opacity(0.0))
        .when(selected, |this| this.border_color(theme.colors.ring))
        .child(
            div()
                .flex()
                .flex_row()
                .items_center()
                .justify_between()
                .gap(theme.spacing.sm)
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .child(entry.primary_label()),
                )
                .child(
                    div()
                        .w(px(100.0))
                        .h(px(20.0))
                        .flex()
                        .flex_row()
                        .items_center()
                        .justify_end()
                        .child({
                            let can_archive = entry.kind == AgentSessionKind::Chat
                                && entry.repo_scope.is_some()
                                && selected;
                            let is_archiving =
                                archiving_session_id.is_some_and(|id| id == entry.session_id);
                            if can_archive {
                                let label = if is_archiving {
                                    "Archiving…"
                                } else {
                                    "Archive"
                                };
                                let root = root.clone();
                                let session_id = entry.session_id;
                                let repo_scope = entry.repo_scope;
                                TextButton::new(
                                    (
                                        gpui::ElementId::from((
                                            "agent_session_archive",
                                            cx.entity_id(),
                                        )),
                                        entry.session_id.to_string(),
                                    ),
                                    label,
                                )
                                .kind(ButtonKind::Ghost)
                                .small()
                                .disabled(globally_disabled || is_archiving)
                                .disabled_reason("Wait for the current action to finish.")
                                .tooltip("Archive chat")
                                .on_click(move |_, _, cx| {
                                    if let Some(scope) = repo_scope {
                                        root.update(cx, |this, cx| {
                                            this.archive_agent_session_palette_chat(
                                                scope, session_id, cx,
                                            );
                                            this.ui_updates.bump();
                                        });
                                    }
                                    cx.stop_propagation();
                                })
                                .into_any_element()
                            } else {
                                div().into_any_element()
                            }
                        }),
                ),
        )
        .child(
            div()
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child(subtitle),
        );

    if let Some(reason) = disabled_reason {
        let reason_normalized = reason.to_ascii_lowercase();
        let should_render_reason = !reason_normalized.contains("no epic binding");
        if should_render_reason {
            row = row.child(
                div()
                    .text_xs()
                    .text_color(theme.colors.foreground_muted)
                    .child(reason),
            );
        }
    }

    if navigation_disabled {
        row = row.opacity(0.55);
    }

    if globally_disabled {
        row = row.cursor_not_allowed();
    } else {
        row = row.on_mouse_move(cx.listener(move |this, _, _, cx| {
            this.agent_sessions_palette
                .set_selected_display_index(display_index, cx);
            this.ui_updates.bump();
        }));

        if navigation_disabled {
            row = row.cursor_default();
        } else {
            let session_id = entry.session_id;
            let root = root.clone();
            row = row.cursor_pointer().on_click(move |_, window, cx| {
                let focus = root.read(cx).focus_handle.clone();
                root.update(cx, |this, cx| {
                    this.activate_agent_session_palette_entry_by_session_id(session_id, cx);
                    this.ui_updates.bump();
                });
                if !root.read(cx).agent_sessions_palette.is_open() {
                    window.focus(&focus);
                }
            });
        }
    }

    row
}

fn cmp_last_activity_desc(left: Option<Timestamp>, right: Option<Timestamp>) -> std::cmp::Ordering {
    right.cmp(&left)
}
