use std::sync::OnceLock;
use std::time::Duration;

use crate::settings::ThemePreference;

#[derive(Debug, Clone, Copy)]
struct UiTestMode {
    enabled: bool,
    theme: ThemePreference,
}

static UI_TEST_MODE: OnceLock<UiTestMode> = OnceLock::new();

pub fn ui_test_mode_enabled() -> bool {
    ui_test_mode().enabled
}

pub fn ui_test_theme_override() -> Option<ThemePreference> {
    let mode = ui_test_mode();
    mode.enabled.then_some(mode.theme)
}

pub fn ui_test_mode_animation_duration(duration: Duration) -> Duration {
    ui_test_mode_animation_duration_for_enabled(ui_test_mode_enabled(), duration)
}

fn ui_test_mode_animation_duration_for_enabled(enabled: bool, duration: Duration) -> Duration {
    enabled
        .then_some(Duration::from_millis(0))
        .unwrap_or(duration)
}

fn ui_test_mode() -> UiTestMode {
    *UI_TEST_MODE.get_or_init(|| {
        let enabled = std::env::var_os("REDESMYN_UI_TEST_MODE").is_some();
        let theme_env = std::env::var("REDESMYN_UI_TEST_THEME").ok();
        let theme_env = theme_env.as_deref().map(str::trim);
        ui_test_mode_from_env(enabled, theme_env)
    })
}

fn ui_test_mode_from_env(enabled: bool, theme_env: Option<&str>) -> UiTestMode {
    let theme = if enabled {
        match theme_env.map(str::to_ascii_lowercase).as_deref() {
            Some("light") => ThemePreference::Light,
            Some("dark") => ThemePreference::Dark,
            _ => ThemePreference::Dark,
        }
    } else {
        ThemePreference::System
    };

    UiTestMode { enabled, theme }
}

#[cfg(test)]
mod tests {
    use super::{
        ThemePreference, ui_test_mode_animation_duration_for_enabled, ui_test_mode_from_env,
    };
    use std::time::Duration;

    #[test]
    fn ui_test_mode_defaults_to_system_when_disabled() {
        let mode = ui_test_mode_from_env(false, None);
        assert!(!mode.enabled);
        assert_eq!(mode.theme, ThemePreference::System);
    }

    #[test]
    fn ui_test_mode_pins_dark_when_enabled_without_theme() {
        let mode = ui_test_mode_from_env(true, None);
        assert!(mode.enabled);
        assert_eq!(mode.theme, ThemePreference::Dark);
    }

    #[test]
    fn ui_test_mode_respects_light_theme() {
        let mode = ui_test_mode_from_env(true, Some("light"));
        assert!(mode.enabled);
        assert_eq!(mode.theme, ThemePreference::Light);
    }

    #[test]
    fn ui_test_mode_treats_unknown_theme_as_dark() {
        let mode = ui_test_mode_from_env(true, Some("nope"));
        assert!(mode.enabled);
        assert_eq!(mode.theme, ThemePreference::Dark);
    }

    #[test]
    fn ui_test_mode_animation_duration_is_zero_when_enabled() {
        assert_eq!(
            ui_test_mode_animation_duration_for_enabled(true, Duration::from_millis(123)),
            Duration::from_millis(0)
        );
        assert_eq!(
            ui_test_mode_animation_duration_for_enabled(false, Duration::from_millis(123)),
            Duration::from_millis(123)
        );
    }
}
