use gpui::{Hsla, Pixels, div, prelude::*};

use crate::styles::UiTheme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlaySurfaceKind {
    /// Compact chrome that floats over content (e.g. filter chips row).
    Chrome,
    /// Larger popovers/menus (e.g. filter cascaded menus).
    Menu,
}

#[derive(Debug, Clone, Copy)]
struct OverlaySurfaceStyle {
    background: Hsla,
    border: Hsla,
}

impl OverlaySurfaceStyle {
    fn for_kind(theme: &UiTheme, kind: OverlaySurfaceKind) -> Self {
        let (background_alpha, border_alpha) = match kind {
            OverlaySurfaceKind::Chrome => (0.86_f32, 0.18_f32),
            OverlaySurfaceKind::Menu => (0.92_f32, 0.22_f32),
        };

        Self {
            background: theme.colors.surface.opacity(background_alpha),
            border: theme.colors.border.opacity(border_alpha),
        }
    }
}

/// Creates a tinted, translucent overlay surface.
///
/// GPUI currently doesn't expose a public API for per-element backdrop blur / custom fragment
/// shaders. Keep this implementation simple and stable: a translucent tint + a subtle border.
pub fn overlay_surface(theme: &UiTheme, kind: OverlaySurfaceKind, radius: Pixels) -> gpui::Div {
    let style = OverlaySurfaceStyle::for_kind(theme, kind);

    div()
        .relative()
        .overflow_hidden()
        .rounded(radius)
        .bg(style.background)
        .border_1()
        .border_color(style.border)
}
