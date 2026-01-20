use gpui::{App, Window};

use crate::settings::ThemePreference;
use crate::styles::{UiDensity, UiScale, UiTheme};
use crate::UiContext;

pub fn theme_for_window(window: &Window, cx: &App) -> UiTheme {
    cx.try_global::<UiContext>()
        .map(|ui| ui.theme_for_window(window))
        .unwrap_or_else(|| {
            UiTheme::for_window(
                ThemePreference::System,
                window.appearance(),
                UiDensity::default(),
                UiScale::default(),
            )
        })
}

