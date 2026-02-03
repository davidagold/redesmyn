use gpui::{App, KeyBinding, actions};

actions!(
    redesmyn_desktop_task_filters,
    [OpenTaskFilters, CloseTaskFilters]
);

pub fn bind_task_filter_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("f", OpenTaskFilters, Some("Workspace")),
        KeyBinding::new("escape", CloseTaskFilters, Some("TaskFilters")),
        KeyBinding::new("escape", CloseTaskFilters, Some("TaskFilters > TextInput")),
    ]);
}
