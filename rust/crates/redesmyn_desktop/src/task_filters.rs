use gpui::{App, KeyBinding, actions};
use redesmyn_ui_graph::GRAPH_SHORTCUTS_KEY_CONTEXT;

actions!(
    redesmyn_desktop_task_filters,
    [
        OpenTaskFilters,
        CloseTaskFilters,
        TaskFiltersActivate,
        TaskFiltersClearFocusedChip,
        TaskFiltersMoveUp,
        TaskFiltersMoveDown,
        TaskFiltersMoveLeft,
        TaskFiltersMoveRight,
        TaskFiltersToggleChipFocus,
    ]
);

pub fn bind_task_filter_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("f", OpenTaskFilters, Some(GRAPH_SHORTCUTS_KEY_CONTEXT)),
        KeyBinding::new("escape", CloseTaskFilters, Some("TaskFilters")),
        KeyBinding::new("escape", CloseTaskFilters, Some("TaskFilters > TextInput")),
        KeyBinding::new("tab", TaskFiltersToggleChipFocus, Some("TaskFilters")),
        KeyBinding::new(
            "tab",
            TaskFiltersToggleChipFocus,
            Some("TaskFilters > TextInput"),
        ),
        KeyBinding::new("shift-tab", TaskFiltersToggleChipFocus, Some("TaskFilters")),
        KeyBinding::new(
            "shift-tab",
            TaskFiltersToggleChipFocus,
            Some("TaskFilters > TextInput"),
        ),
        KeyBinding::new("up", TaskFiltersMoveUp, Some("TaskFilters")),
        KeyBinding::new("up", TaskFiltersMoveUp, Some("TaskFilters > TextInput")),
        KeyBinding::new("down", TaskFiltersMoveDown, Some("TaskFilters")),
        KeyBinding::new("down", TaskFiltersMoveDown, Some("TaskFilters > TextInput")),
        KeyBinding::new("left", TaskFiltersMoveLeft, Some("TaskFilters")),
        KeyBinding::new("left", TaskFiltersMoveLeft, Some("TaskFilters > TextInput")),
        KeyBinding::new("right", TaskFiltersMoveRight, Some("TaskFilters")),
        KeyBinding::new(
            "right",
            TaskFiltersMoveRight,
            Some("TaskFilters > TextInput"),
        ),
        KeyBinding::new("enter", TaskFiltersActivate, Some("TaskFilters")),
        KeyBinding::new(
            "backspace",
            TaskFiltersClearFocusedChip,
            Some("TaskFilters"),
        ),
        KeyBinding::new("delete", TaskFiltersClearFocusedChip, Some("TaskFilters")),
    ]);
}
