use gpui::{Background, Hsla, Pixels, div, linear_color_stop, linear_gradient, prelude::*};

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
    background: Background,
    highlight: Background,
    inner_highlight: Hsla,
}

impl OverlaySurfaceStyle {
    fn for_kind(theme: &UiTheme, kind: OverlaySurfaceKind) -> Self {
        let (base, top_alpha, bottom_alpha, highlight_alpha, inner_highlight_alpha) = match kind {
            OverlaySurfaceKind::Chrome => (
                theme.colors.surface_elevated,
                0.84_f32,
                0.76_f32,
                0.10_f32,
                0.10_f32,
            ),
            OverlaySurfaceKind::Menu => (
                theme.colors.surface,
                0.97_f32,
                0.92_f32,
                0.18_f32,
                0.14_f32,
            ),
        };

        // Subtle, tinted "glass" background.
        let background = linear_gradient(
            180.0,
            linear_color_stop(base.opacity(top_alpha), 0.0),
            linear_color_stop(base.opacity(bottom_alpha), 1.0),
        );

        // Gentle top highlight to separate overlay from content.
        let highlight_color = theme.colors.surface_elevated;
        let highlight = linear_gradient(
            180.0,
            linear_color_stop(highlight_color.opacity(highlight_alpha), 0.0),
            linear_color_stop(highlight_color.opacity(0.0), 1.0),
        );

        // Hairline inner highlight (very subtle). This should read as a sheen, not a border.
        let inner_highlight = theme.colors.surface_elevated.opacity(inner_highlight_alpha);

        Self {
            background,
            highlight,
            inner_highlight,
        }
    }
}

/// Creates a tinted, translucent overlay surface.
///
/// This is intentionally an "80/20 glass" emulation: GPUI currently doesn't expose a public API
/// for per-element backdrop blur / custom fragment shaders, so we approximate with translucency,
/// a subtle highlight gradient, and a hairline inner sheen.
pub fn overlay_surface(theme: &UiTheme, kind: OverlaySurfaceKind, radius: Pixels) -> gpui::Div {
    let style = OverlaySurfaceStyle::for_kind(theme, kind);

    div()
        .relative()
        .overflow_hidden()
        .rounded(radius)
        .bg(style.background)
        .child(div().absolute().inset_0().rounded(radius).bg(style.highlight))
        .child(
            div()
                .absolute()
                .inset_0()
                .rounded(radius)
                .border_1()
                .border_color(style.inner_highlight),
        )
}

