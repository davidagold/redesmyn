use gpui::{App, Global, Window};

use crate::components::SplitPaneState;
use crate::settings::{ThemePreference, UiSettingsError, UiSettingsStore};
use crate::styles::{UiDensity, UiScale, UiTheme};

#[derive(Debug)]
pub struct UiContext {
    settings: UiSettingsStore,
    density: UiDensity,
    scale: UiScale,
}

impl Global for UiContext {}

impl UiContext {
    pub fn init(cx: &mut App) -> Result<(), UiSettingsError> {
        let settings = UiSettingsStore::load();
        cx.set_global(Self {
            settings,
            density: UiDensity::default(),
            scale: UiScale::default(),
        });
        Ok(())
    }

    pub fn theme_preference(&self) -> ThemePreference {
        self.settings.settings().theme
    }

    pub fn set_theme_preference(&mut self, theme: ThemePreference) -> Result<(), UiSettingsError> {
        self.settings.settings_mut().theme = theme;
        self.settings.save()
    }

    pub fn main_split_pane_state(&self) -> SplitPaneState {
        self.settings.settings().main_split_pane
    }

    pub fn set_main_split_pane_state(
        &mut self,
        state: SplitPaneState,
    ) -> Result<(), UiSettingsError> {
        self.settings.settings_mut().main_split_pane = state;
        self.settings.save()
    }

    pub fn theme_for_window(&self, window: &Window) -> UiTheme {
        UiTheme::for_window(
            self.settings.settings().theme,
            window.appearance(),
            self.density,
            self.scale,
        )
    }
}
