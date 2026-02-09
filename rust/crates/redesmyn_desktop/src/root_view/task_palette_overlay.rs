use std::collections::HashSet;

use gpui::{
    App, Context, ElementId, Entity, FocusHandle, Focusable, ScrollHandle, SharedString, Window,
    div, point, prelude::*, px,
};

use redesmyn_ids::TaskId;
use redesmyn_protocol::{Timestamp, client::MergeReadiness, client::TaskState};
use redesmyn_ui::components::{ProgressPillKind, ScrollArea, TextInput, TextInputEvent};
use redesmyn_ui::styles::UiTheme;
use redesmyn_ui::utils::theme_for_window;

use super::RootView;
use super::palette_overlay::{SharedPaletteOverlay, render_shared_palette_overlay};

const RECENT_TASKS_LIMIT: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskPaletteNavigation {
    Task { epic_slug: String, task_id: TaskId },
    Disabled { reason: SharedString },
}

#[derive(Debug, Clone)]
pub struct TaskPaletteEntry {
    pub task_id: Option<TaskId>,
    pub task_slug: String,
    pub task_title: String,
    pub task_state: TaskState,
    pub merge_readiness: MergeReadiness,
    pub branch_name: Option<String>,
    pub last_activity: Option<Timestamp>,
    pub navigation: TaskPaletteNavigation,
}

impl TaskPaletteEntry {
    fn primary_label(&self) -> String {
        if self.task_title.trim().is_empty() {
            self.task_slug.clone()
        } else {
            self.task_title.clone()
        }
    }

    fn subtitle(&self) -> String {
        let mut parts = vec![
            self.task_slug.clone(),
            task_state_label(self.task_state).to_string(),
        ];

        if self.merge_readiness != MergeReadiness::Unknown {
            parts.push(format!(
                "merge {}",
                merge_readiness_label(self.merge_readiness)
            ));
        }
        if let Some(branch_name) = self
            .branch_name
            .as_ref()
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            parts.push(branch_name.to_string());
        }

        parts.join(" • ")
    }

    fn search_matches(&self, query: &str) -> bool {
        if query.is_empty() {
            return true;
        }

        let mut haystacks = vec![
            self.task_slug.clone(),
            self.task_title.clone(),
            task_state_label(self.task_state).to_string(),
            merge_readiness_label(self.merge_readiness).to_string(),
            self.subtitle(),
        ];

        if let Some(branch_name) = self.branch_name.as_ref() {
            haystacks.push(branch_name.clone());
        }
        if let Some(task_id) = self.task_id {
            haystacks.push(task_id.to_string());
        }

        haystacks
            .into_iter()
            .any(|value| value.to_ascii_lowercase().contains(query))
    }
}

#[derive(Debug, Default)]
struct DisplayRows {
    ordered_indices: Vec<usize>,
    in_progress_indices: Vec<usize>,
    recent_indices: Vec<usize>,
    remaining_indices: Vec<usize>,
}

pub struct TaskPaletteOverlay {
    root_focus_handle: FocusHandle,
    open: bool,
    input: Entity<TextInput>,
    scroll: ScrollHandle,
    selected_index: usize,
    loading: bool,
    action_in_flight: bool,
    error: Option<SharedString>,
    epic_slug: Option<String>,
    epic_title: Option<String>,
    entries: Vec<TaskPaletteEntry>,
}

impl TaskPaletteOverlay {
    pub fn new(root_focus_handle: FocusHandle, cx: &mut Context<RootView>) -> Self {
        let input = cx.new(|cx| TextInput::new(cx).placeholder("Filter tasks…"));
        Self {
            root_focus_handle,
            open: false,
            input,
            scroll: ScrollHandle::new(),
            selected_index: 0,
            loading: false,
            action_in_flight: false,
            error: None,
            epic_slug: None,
            epic_title: None,
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

    pub fn start_loading_for_epic(
        &mut self,
        epic_slug: String,
        epic_title: Option<String>,
        cx: &mut Context<RootView>,
    ) {
        self.epic_slug = Some(epic_slug);
        self.epic_title = epic_title;
        self.loading = true;
        self.action_in_flight = false;
        self.error = None;
        self.selected_index = 0;
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub fn finish_loading_for_epic(
        &mut self,
        epic_slug: String,
        epic_title: Option<String>,
        entries: Vec<TaskPaletteEntry>,
        cx: &mut Context<RootView>,
    ) {
        self.epic_slug = Some(epic_slug);
        self.epic_title = epic_title;
        self.entries = entries;
        self.loading = false;
        self.action_in_flight = false;
        self.error = None;
        self.selected_index = 0;
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub fn fail_loading(&mut self, message: impl Into<SharedString>, cx: &mut Context<RootView>) {
        self.loading = false;
        self.action_in_flight = false;
        self.error = Some(message.into());
        self.selected_index = 0;
        cx.notify();
    }

    pub fn handle_text_input_event(
        &mut self,
        event: TextInputEvent,
        cx: &mut Context<RootView>,
    ) -> Option<TaskPaletteEntry> {
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

    pub fn activate_selected_entry(
        &mut self,
        cx: &mut Context<RootView>,
    ) -> Option<TaskPaletteEntry> {
        self.activate_selected(cx)
    }

    pub fn complete_activation_success(&mut self, cx: &mut Context<RootView>) {
        self.action_in_flight = false;
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
        self.error = None;
        self.selected_index = 0;
        self.input.update(cx, |input, cx| input.set_text("", cx));
        cx.notify();
    }

    pub fn render(&self, window: &mut Window, cx: &mut Context<RootView>) -> impl IntoElement {
        self.render_overlay(window, cx)
    }

    fn activate_selected(&mut self, cx: &mut Context<RootView>) -> Option<TaskPaletteEntry> {
        if !self.open || self.is_blocking_loading() || self.action_in_flight {
            return None;
        }

        let display = self.display_rows(cx);
        let selected_index = self
            .selected_index
            .min(display.ordered_indices.len().saturating_sub(1));
        let Some(entry_index) = display.ordered_indices.get(selected_index).copied() else {
            self.error = Some("No matching tasks.".into());
            cx.notify();
            return None;
        };

        let Some(entry) = self.entries.get(entry_index).cloned() else {
            self.error = Some("Unable to resolve selected task.".into());
            cx.notify();
            return None;
        };

        match entry.navigation {
            TaskPaletteNavigation::Disabled { ref reason } => {
                self.error = Some(reason.clone());
                cx.notify();
                None
            }
            TaskPaletteNavigation::Task { .. } => {
                self.error = None;
                self.action_in_flight = true;
                cx.notify();
                Some(entry)
            }
        }
    }

    fn open_palette(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        self.open = true;
        self.selected_index = 0;
        self.error = None;
        self.action_in_flight = false;
        self.input.update(cx, |input, cx| input.set_text("", cx));
        self.scroll.set_offset(point(px(0.0), px(0.0)));
        window.focus(&self.input.focus_handle(cx));
        cx.notify();
    }

    fn close_palette(&mut self, window: &mut Window, cx: &mut Context<RootView>) {
        self.open = false;
        self.selected_index = 0;
        self.error = None;
        self.action_in_flight = false;
        self.input.update(cx, |input, cx| input.set_text("", cx));
        window.focus(&self.root_focus_handle);
        cx.notify();
    }

    fn is_blocking_loading(&self) -> bool {
        self.loading && self.entries.is_empty()
    }

    fn filtered_indices(&self, cx: &App) -> Vec<usize> {
        let query = self.input.read(cx).text().to_ascii_lowercase();
        let query = query.trim();

        self.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.search_matches(query))
            .map(|(index, _)| index)
            .collect()
    }

    fn display_rows(&self, cx: &App) -> DisplayRows {
        let filtered = self.filtered_indices(cx);

        let mut in_progress_indices: Vec<usize> = filtered
            .iter()
            .copied()
            .filter(|index| {
                self.entries
                    .get(*index)
                    .is_some_and(|entry| entry.task_state == TaskState::InProgress)
            })
            .collect();
        in_progress_indices.sort_by(|left, right| {
            cmp_last_activity_desc(
                self.entries
                    .get(*left)
                    .and_then(|entry| entry.last_activity),
                self.entries
                    .get(*right)
                    .and_then(|entry| entry.last_activity),
            )
            .then_with(|| cmp_task_entry_sort_key(&self.entries[*left], &self.entries[*right]))
        });

        let in_progress_set: HashSet<usize> = in_progress_indices.iter().copied().collect();
        let mut recent_indices: Vec<usize> = filtered
            .iter()
            .copied()
            .filter(|index| !in_progress_set.contains(index))
            .filter(|index| {
                self.entries
                    .get(*index)
                    .and_then(|entry| entry.last_activity)
                    .is_some()
            })
            .collect();
        recent_indices.sort_by(|left, right| {
            cmp_last_activity_desc(
                self.entries
                    .get(*left)
                    .and_then(|entry| entry.last_activity),
                self.entries
                    .get(*right)
                    .and_then(|entry| entry.last_activity),
            )
            .then_with(|| cmp_task_entry_sort_key(&self.entries[*left], &self.entries[*right]))
        });
        recent_indices.truncate(RECENT_TASKS_LIMIT);

        let recent_set: HashSet<usize> = recent_indices.iter().copied().collect();
        let mut remaining_indices: Vec<usize> = filtered
            .iter()
            .copied()
            .filter(|index| !in_progress_set.contains(index) && !recent_set.contains(index))
            .collect();
        remaining_indices.sort_by(|left, right| {
            cmp_task_entry_sort_key(&self.entries[*left], &self.entries[*right])
        });

        let mut ordered_indices = in_progress_indices.clone();
        ordered_indices.extend(recent_indices.iter().copied());
        ordered_indices.extend(remaining_indices.iter().copied());

        DisplayRows {
            ordered_indices,
            in_progress_indices,
            recent_indices,
            remaining_indices,
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

        if !display.in_progress_indices.is_empty() {
            child_index += 1;
            for _ in &display.in_progress_indices {
                if display_index == selected_display_index {
                    return Some(child_index);
                }
                display_index += 1;
                child_index += 1;
            }
        }

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

        if !display.remaining_indices.is_empty() {
            child_index += 1;
            for _ in &display.remaining_indices {
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

        let mut list = ScrollArea::new(("task_palette_list", cx.entity_id()), self.scroll.clone())
            .scrollbar_width(px(10.0));

        if self.loading && self.entries.is_empty() {
            list = list.child(
                div()
                    .py(theme.spacing.sm)
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("Loading tasks…"),
            );
        } else if display.ordered_indices.is_empty() {
            list = list.child(
                div()
                    .py(theme.spacing.sm)
                    .text_sm()
                    .text_color(theme.colors.foreground_muted)
                    .child("No tasks match."),
            );
        } else {
            if !display.in_progress_indices.is_empty() {
                list = list.child(
                    div()
                        .pt(theme.spacing.sm)
                        .flex()
                        .flex_row()
                        .items_center()
                        .gap(theme.spacing.xs)
                        .text_xs()
                        .text_color(theme.colors.foreground_muted)
                        .child(div().text_xs().child("▶"))
                        .child(div().text_xs().child("In progress")),
                );

                for entry_index in &display.in_progress_indices {
                    let Some(entry) = self.entries.get(*entry_index) else {
                        continue;
                    };
                    let row_display_index = display
                        .ordered_indices
                        .iter()
                        .position(|index| index == entry_index)
                        .unwrap_or(0);
                    list = list.child(render_task_row(
                        entry,
                        row_display_index == selected_index,
                        row_display_index,
                        self.is_blocking_loading() || self.action_in_flight,
                        &root,
                        cx,
                        &theme,
                    ));
                }
            }

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
                        .child(div().text_xs().child("Recent tasks")),
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
                    list = list.child(render_task_row(
                        entry,
                        row_display_index == selected_index,
                        row_display_index,
                        self.is_blocking_loading() || self.action_in_flight,
                        &root,
                        cx,
                        &theme,
                    ));
                }
            }

            if !display.remaining_indices.is_empty() {
                list = list.child(
                    div()
                        .pt(theme.spacing.sm)
                        .text_xs()
                        .text_color(theme.colors.foreground_muted)
                        .child("All tasks"),
                );

                for entry_index in &display.remaining_indices {
                    let Some(entry) = self.entries.get(*entry_index) else {
                        continue;
                    };
                    let row_display_index = display
                        .ordered_indices
                        .iter()
                        .position(|index| index == entry_index)
                        .unwrap_or(0);
                    list = list.child(render_task_row(
                        entry,
                        row_display_index == selected_index,
                        row_display_index,
                        self.is_blocking_loading() || self.action_in_flight,
                        &root,
                        cx,
                        &theme,
                    ));
                }
            }
        }

        let progress_label = if self.loading && self.entries.is_empty() {
            Some("Loading tasks…".into())
        } else if self.loading {
            Some("Refreshing tasks…".into())
        } else if self.action_in_flight {
            Some("Opening task…".into())
        } else {
            None
        };

        let title = self
            .epic_title
            .as_deref()
            .or(self.epic_slug.as_deref())
            .map(|label| format!("Tasks in {label}"))
            .unwrap_or_else(|| "Tasks".to_string());

        render_shared_palette_overlay(
            root.clone(),
            SharedPaletteOverlay {
                key_context: "TaskPalette",
                title: title.into(),
                subtitle: "Type to filter, ↑/↓ to select, Enter to open.".into(),
                progress_label,
                progress_kind: ProgressPillKind::Accent,
                error_title: "Task palette".into(),
                error: self.error.clone(),
                input: self.input.clone(),
                list: div().max_h(px(420.0)).child(list).into_any_element(),
                close_button_id: "task_palette_close",
                close_tooltip: "Close (Esc)",
                backdrop_id: "task_palette_backdrop",
            },
            window,
            cx,
            |this, cx| {
                this.task_palette.dismiss_overlay(cx);
                this.ui_updates.bump();
            },
        )
    }
}

fn render_task_row(
    entry: &TaskPaletteEntry,
    selected: bool,
    display_index: usize,
    globally_disabled: bool,
    root: &Entity<RootView>,
    cx: &mut Context<RootView>,
    theme: &UiTheme,
) -> impl IntoElement {
    let disabled_reason = match &entry.navigation {
        TaskPaletteNavigation::Disabled { reason } => Some(reason.clone()),
        TaskPaletteNavigation::Task { .. } => None,
    };
    let navigation_disabled = disabled_reason.is_some();
    let base_row_id = ElementId::from(("task_palette_row", cx.entity_id()));

    let mut row = div()
        .id((base_row_id, entry.task_slug.clone()))
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
                    div().min_w_0().flex_1().child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground)
                            .truncate()
                            .child(entry.primary_label()),
                    ),
                )
                .child(
                    div()
                        .text_xs()
                        .flex_none()
                        .text_color(theme.colors.foreground_muted)
                        .child(task_state_label(entry.task_state)),
                ),
        )
        .child(
            div()
                .truncate()
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child(entry.subtitle()),
        );

    if let Some(reason) = disabled_reason {
        row = row.child(
            div()
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child(reason),
        );
    }

    if navigation_disabled {
        row = row.opacity(0.55);
    }

    if globally_disabled {
        return row.cursor_not_allowed();
    }

    row = row.on_mouse_move(cx.listener(move |this, _, _, cx| {
        this.task_palette
            .set_selected_display_index(display_index, cx);
        this.ui_updates.bump();
    }));

    if navigation_disabled {
        return row.cursor_default();
    }

    let root = root.clone();
    row.cursor_pointer().on_click(move |_, window, cx| {
        let focus = root.read(cx).focus_handle.clone();
        root.update(cx, |this, cx| {
            this.task_palette
                .set_selected_display_index(display_index, cx);
            if let Some(selection) = this.task_palette.activate_selected_entry(cx) {
                this.activate_task_palette_entry(selection, cx);
            }
            this.ui_updates.bump();
        });
        if !root.read(cx).task_palette.is_open() {
            window.focus(&focus);
        }
    })
}

fn cmp_last_activity_desc(left: Option<Timestamp>, right: Option<Timestamp>) -> std::cmp::Ordering {
    right.cmp(&left)
}

fn cmp_task_entry_sort_key(
    left: &TaskPaletteEntry,
    right: &TaskPaletteEntry,
) -> std::cmp::Ordering {
    left.task_slug
        .to_ascii_lowercase()
        .cmp(&right.task_slug.to_ascii_lowercase())
        .then_with(|| {
            left.task_title
                .to_ascii_lowercase()
                .cmp(&right.task_title.to_ascii_lowercase())
        })
}

fn task_state_label(state: TaskState) -> &'static str {
    match state {
        TaskState::Unknown => "unknown",
        TaskState::Todo => "todo",
        TaskState::InProgress => "in progress",
        TaskState::Blocked => "blocked",
        TaskState::Done => "done",
    }
}

fn merge_readiness_label(readiness: MergeReadiness) -> &'static str {
    match readiness {
        MergeReadiness::Unknown => "unknown",
        MergeReadiness::Ready => "ready",
        MergeReadiness::Blocked => "blocked",
    }
}
