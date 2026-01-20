use gpui::WindowAppearance;

use crate::settings::ThemePreference;
use crate::styles::{
    AnimationDurations, ColorTokens, RadiusTokens, SpacingTokens, TypographyTokens, UiDensity,
    UiScale,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeMode {
    Light,
    Dark,
}

impl ThemeMode {
    pub fn from_window_appearance(appearance: WindowAppearance) -> Self {
        match appearance {
            WindowAppearance::Light | WindowAppearance::VibrantLight => Self::Light,
            WindowAppearance::Dark | WindowAppearance::VibrantDark => Self::Dark,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UiTheme {
    pub mode: ThemeMode,
    pub colors: ColorTokens,
    pub spacing: SpacingTokens,
    pub radius: RadiusTokens,
    pub typography: TypographyTokens,
    pub animation: AnimationDurations,
}

impl UiTheme {
    pub fn new(mode: ThemeMode, density: UiDensity, scale: UiScale) -> Self {
        let colors = match mode {
            ThemeMode::Light => ColorTokens::rose_pine_dawn(),
            ThemeMode::Dark => ColorTokens::rose_pine(),
        };

        Self {
            mode,
            colors,
            spacing: SpacingTokens::new(density, scale),
            radius: RadiusTokens::default(),
            typography: TypographyTokens::default(),
            animation: AnimationDurations::default(),
        }
    }

    pub fn for_window(
        theme_preference: ThemePreference,
        window_appearance: WindowAppearance,
        density: UiDensity,
        scale: UiScale,
    ) -> Self {
        let mode = match theme_preference {
            ThemePreference::Dark => ThemeMode::Dark,
            ThemePreference::Light => ThemeMode::Light,
            ThemePreference::System => ThemeMode::from_window_appearance(window_appearance),
        };
        Self::new(mode, density, scale)
    }
}
