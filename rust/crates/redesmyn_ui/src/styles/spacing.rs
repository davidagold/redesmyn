use gpui::{Pixels, px};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiScale(pub f32);

impl Default for UiScale {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiDensity {
    Compact,
    Comfortable,
}

impl Default for UiDensity {
    fn default() -> Self {
        Self::Comfortable
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpacingTokens {
    pub xs: Pixels,
    pub sm: Pixels,
    pub md: Pixels,
    pub lg: Pixels,
    pub xl: Pixels,
    pub xxl: Pixels,
}

impl SpacingTokens {
    pub fn new(density: UiDensity, scale: UiScale) -> Self {
        let density_scale = match density {
            UiDensity::Compact => 0.9,
            UiDensity::Comfortable => 1.0,
        };
        let s = density_scale * scale.0;

        Self {
            xs: px(4.0 * s),
            sm: px(8.0 * s),
            md: px(12.0 * s),
            lg: px(16.0 * s),
            xl: px(24.0 * s),
            xxl: px(32.0 * s),
        }
    }
}

