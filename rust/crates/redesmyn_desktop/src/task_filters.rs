use gpui::{App, KeyBinding, actions};

actions!(
    redesmyn_desktop_task_filters,
    [
        OpenTaskFilters,
        CloseTaskFilters,
        TaskFiltersMoveUp,
        TaskFiltersMoveDown,
        TaskFiltersMoveLeft,
        TaskFiltersMoveRight,
    ]
);

pub fn bind_task_filter_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("f", OpenTaskFilters, Some("Workspace")),
        KeyBinding::new("escape", CloseTaskFilters, Some("TaskFilters")),
        KeyBinding::new("escape", CloseTaskFilters, Some("TaskFilters > TextInput")),
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
    ]);
}
