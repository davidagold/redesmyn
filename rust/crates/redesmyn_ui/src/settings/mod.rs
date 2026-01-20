//! UI settings (theme, density, persistence).

mod store;
mod theme;

pub use store::{UiSettings, UiSettingsError, UiSettingsStore};
pub use theme::ThemePreference;
