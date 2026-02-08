//! Small UI helpers.

mod action_availability;
mod bounded_cache;
mod external_url;
mod idle;
mod test_mode;
pub(crate) mod text_editing;
mod theme;
mod transition;
mod user_action;

pub use action_availability::ActionAvailabilityProbe;
pub use bounded_cache::BoundedCache;
pub use external_url::{OpenExternalUrl, is_http_https_url};
pub use idle::{UiActivityGuard, UiIdleTracker, ui_idle_tracker};
pub use test_mode::{
    ui_test_mode_animation_duration, ui_test_mode_enabled, ui_test_theme_override,
};
pub use theme::theme_for_window;
pub use transition::{TransitionF32, TransitionMap};
pub use user_action::UserActionState;
