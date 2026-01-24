//! Small UI helpers.

mod idle;
mod test_mode;
pub(crate) mod text_editing;
mod theme;
mod user_action;

pub use idle::{UiActivityGuard, UiIdleTracker, ui_idle_tracker};
pub use test_mode::{
    ui_test_mode_animation_duration, ui_test_mode_enabled, ui_test_theme_override,
};
pub use theme::theme_for_window;
pub use user_action::UserActionState;
