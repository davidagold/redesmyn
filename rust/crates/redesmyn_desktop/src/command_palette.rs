use gpui::{App, KeyBinding, actions};

actions!(
    redesmyn_desktop_command_palette,
    [
        OpenEpicSelector,
        ToggleCommandPalette,
        ToggleAgentSessionsPalette,
        ToggleTaskPalette,
        CloseCommandPalette,
        CloseAgentSessionsPalette,
        CloseTaskPalette,
        SelectPreviousCommand,
        SelectNextCommand,
        SelectPreviousAgentSession,
        SelectNextAgentSession,
        SelectPreviousTaskPaletteItem,
        SelectNextTaskPaletteItem,
    ]
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CommandId {
    ToggleTheme,
    RefreshGraph,
    ToggleLeftSessionPane,
    OpenEpic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandGroup {
    Navigation,
    Actions,
}

impl CommandId {
    pub fn id_str(self) -> &'static str {
        match self {
            Self::ToggleTheme => "toggle_theme",
            Self::RefreshGraph => "refresh_graph",
            Self::ToggleLeftSessionPane => "toggle_left_session_pane",
            Self::OpenEpic => "open_epic",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::ToggleTheme => "Toggle theme",
            Self::RefreshGraph => "Refresh graph",
            Self::ToggleLeftSessionPane => "Toggle left session pane",
            Self::OpenEpic => "Open epic…",
        }
    }

    pub fn keywords(self) -> &'static [&'static str] {
        match self {
            Self::ToggleTheme => &["theme", "dark", "light", "appearance"],
            Self::RefreshGraph => &["refresh", "reload", "resync", "graph"],
            Self::ToggleLeftSessionPane => &["session", "pane", "left", "collapse"],
            Self::OpenEpic => &["open", "epic", "navigate", "switch"],
        }
    }

    pub fn group(self) -> CommandGroup {
        match self {
            Self::OpenEpic => CommandGroup::Navigation,
            Self::ToggleTheme | Self::RefreshGraph | Self::ToggleLeftSessionPane => {
                CommandGroup::Actions
            }
        }
    }

    pub fn matches_query(self, query: &str) -> bool {
        let query = query.trim();
        if query.is_empty() {
            return true;
        }

        let query = query.to_ascii_lowercase();
        if self.title().to_ascii_lowercase().contains(&query) {
            return true;
        }
        self.keywords()
            .iter()
            .any(|keyword| keyword.to_ascii_lowercase().contains(&query))
    }
}

#[derive(Debug, Clone)]
pub struct CommandRegistry {
    commands: Vec<CommandId>,
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self {
            commands: vec![
                CommandId::OpenEpic,
                CommandId::ToggleTheme,
                CommandId::RefreshGraph,
                CommandId::ToggleLeftSessionPane,
            ],
        }
    }
}

impl CommandRegistry {
    pub fn matching(&self, query: &str) -> Vec<CommandId> {
        self.commands
            .iter()
            .copied()
            .filter(|command| command.matches_query(query))
            .collect()
    }
}

pub fn bind_command_palette_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new(
            "e",
            OpenEpicSelector,
            Some("Desktop && !TextInput && !TextArea"),
        ),
        KeyBinding::new("cmd-k", ToggleCommandPalette, Some("Desktop")),
        KeyBinding::new("ctrl-k", ToggleCommandPalette, Some("Desktop")),
        KeyBinding::new("cmd-s", ToggleAgentSessionsPalette, Some("Desktop")),
        KeyBinding::new("ctrl-s", ToggleAgentSessionsPalette, Some("Desktop")),
        KeyBinding::new(
            "t",
            ToggleTaskPalette,
            Some("Desktop && !TextInput && !TextArea"),
        ),
        KeyBinding::new("escape", CloseCommandPalette, Some("CommandPalette")),
        KeyBinding::new(
            "escape",
            CloseCommandPalette,
            Some("CommandPalette > TextInput"),
        ),
        KeyBinding::new(
            "up",
            SelectPreviousCommand,
            Some("CommandPalette > TextInput"),
        ),
        KeyBinding::new(
            "down",
            SelectNextCommand,
            Some("CommandPalette > TextInput"),
        ),
        KeyBinding::new(
            "escape",
            CloseAgentSessionsPalette,
            Some("AgentSessionsPalette"),
        ),
        KeyBinding::new(
            "escape",
            CloseAgentSessionsPalette,
            Some("AgentSessionsPalette > TextInput"),
        ),
        KeyBinding::new(
            "up",
            SelectPreviousAgentSession,
            Some("AgentSessionsPalette > TextInput"),
        ),
        KeyBinding::new(
            "down",
            SelectNextAgentSession,
            Some("AgentSessionsPalette > TextInput"),
        ),
        KeyBinding::new("escape", CloseTaskPalette, Some("TaskPalette")),
        KeyBinding::new("escape", CloseTaskPalette, Some("TaskPalette > TextInput")),
        KeyBinding::new(
            "up",
            SelectPreviousTaskPaletteItem,
            Some("TaskPalette > TextInput"),
        ),
        KeyBinding::new(
            "down",
            SelectNextTaskPaletteItem,
            Some("TaskPalette > TextInput"),
        ),
    ]);
}
