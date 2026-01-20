//! Shared GPUI UI foundations crate (expanded in T-44).

#![forbid(unsafe_code)]

pub mod components;
pub mod settings;
pub mod styles;
pub mod traits;
pub mod utils;

pub mod prelude {
    pub use gpui::prelude::*;
}

