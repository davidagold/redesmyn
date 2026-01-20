//! Redesmyn UI foundations (theme, tokens, shared widgets) for GPUI.
//!
//! This crate is intentionally small and modular so UI work can be parallelized without merge
//! conflicts or a single "god module".

#![forbid(unsafe_code)]

pub mod components;
pub mod settings;
pub mod styles;
pub mod traits;
pub mod utils;

pub mod prelude {
    pub use gpui::prelude::*;
}
