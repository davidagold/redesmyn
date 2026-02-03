//! Redesmyn UI foundations (theme, tokens, shared widgets) for GPUI.
//!
//! This crate is intentionally small and modular so UI work can be parallelized without merge
//! conflicts or a single "god module".

#![forbid(unsafe_code)]

pub mod components;
pub mod settings;
pub mod styles;
pub mod task_filters;
pub mod traits;
pub mod utils;

mod ui_context;

pub use ui_context::UiContext;

pub mod prelude {
    pub use gpui::prelude::*;
}
