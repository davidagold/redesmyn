use gpui::{Pixels, px};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiusTokens {
    pub sm: Pixels,
    pub md: Pixels,
    pub lg: Pixels,
    pub xl: Pixels,
}

impl Default for RadiusTokens {
    fn default() -> Self {
        Self {
            sm: px(6.0),
            md: px(8.0),
            lg: px(10.0),
            xl: px(14.0),
        }
    }
}

