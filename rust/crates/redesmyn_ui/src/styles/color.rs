use gpui::{Hsla, Rgba, rgb};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorTokens {
    pub background: Hsla,
    pub surface: Hsla,
    pub surface_elevated: Hsla,
    pub foreground: Hsla,
    pub foreground_muted: Hsla,
    pub border: Hsla,
    pub ring: Hsla,
    pub accent: Hsla,
    pub accent_foreground: Hsla,
    pub danger: Hsla,
    pub warning: Hsla,
}

impl ColorTokens {
    pub fn rose_pine_dawn() -> Self {
        // `dashboard/src/index.css` (Rose Pine Dawn).
        Self {
            background: hsla(rgb(0xfefdfb)),
            surface: hsla(rgb(0xf7f6f4)),
            surface_elevated: hsla(rgb(0xfefdfb)),
            foreground: hsla(rgb(0x575279)),
            foreground_muted: hsla(rgb(0x797593)),
            border: hsla(rgb(0xdfdad9)),
            ring: hsla(rgb(0x907aa9)),
            accent: hsla(rgb(0xf2e9e1)),
            accent_foreground: hsla(rgb(0x575279)),
            danger: hsla(rgb(0xb4637a)),
            warning: hsla(rgb(0xea9d34)),
        }
    }

    pub fn rose_pine() -> Self {
        // `dashboard/src/index.css` (Rose Pine).
        Self {
            background: hsla(rgb(0x161420)),
            surface: hsla(rgb(0x100e17)),
            surface_elevated: hsla(rgb(0x1f1d2e)),
            foreground: hsla(rgb(0xe0def4)),
            foreground_muted: hsla(rgb(0x908caa)),
            border: hsla(rgb(0x403d52)),
            ring: hsla(rgb(0xc4a7e7)),
            accent: hsla(rgb(0x26233a)),
            accent_foreground: hsla(rgb(0xe0def4)),
            danger: hsla(rgb(0xeb6f92)),
            warning: hsla(rgb(0xf6c177)),
        }
    }
}

fn hsla(rgb: Rgba) -> Hsla {
    rgb.into()
}
