use gpui::{App, KeyBinding, actions};

actions!(
    redesmyn_desktop_settings_dialog,
    [ToggleSettingsDialog, CloseSettingsDialog]
);

pub fn bind_settings_dialog_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("cmd-,", ToggleSettingsDialog, Some("Desktop")),
        KeyBinding::new("ctrl-,", ToggleSettingsDialog, Some("Desktop")),
        KeyBinding::new("cmd-,", ToggleSettingsDialog, Some("SettingsDialog")),
        KeyBinding::new(
            "cmd-,",
            ToggleSettingsDialog,
            Some("SettingsDialog > TextInput"),
        ),
        KeyBinding::new(
            "cmd-,",
            ToggleSettingsDialog,
            Some("SettingsDialog > TextArea"),
        ),
        KeyBinding::new("ctrl-,", ToggleSettingsDialog, Some("SettingsDialog")),
        KeyBinding::new(
            "ctrl-,",
            ToggleSettingsDialog,
            Some("SettingsDialog > TextInput"),
        ),
        KeyBinding::new(
            "ctrl-,",
            ToggleSettingsDialog,
            Some("SettingsDialog > TextArea"),
        ),
        KeyBinding::new("escape", CloseSettingsDialog, Some("SettingsDialog")),
        KeyBinding::new(
            "escape",
            CloseSettingsDialog,
            Some("SettingsDialog > TextInput"),
        ),
        KeyBinding::new(
            "escape",
            CloseSettingsDialog,
            Some("SettingsDialog > TextArea"),
        ),
    ]);
}
