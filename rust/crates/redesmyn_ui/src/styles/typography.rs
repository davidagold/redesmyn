use gpui::{AbsoluteLength, Font, FontFallbacks, FontWeight, font, rems};

#[derive(Debug, Clone, PartialEq)]
pub struct TextToken {
    pub font: Font,
    pub size: AbsoluteLength,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypographyTokens {
    pub body: TextToken,
    pub caption: TextToken,
    pub mono: TextToken,
}

impl Default for TypographyTokens {
    fn default() -> Self {
        let system_ui = font(".SystemUIFont");

        let mono_fallbacks = FontFallbacks::from_fonts(vec![
            "Menlo".into(),
            "Monaco".into(),
            "Consolas".into(),
            "DejaVu Sans Mono".into(),
        ]);
        let mono_font = {
            // `SF Mono` isn't installed by default on every macOS system; `Menlo` is.
            let mut font = font("Menlo");
            font.weight = FontWeight::NORMAL;
            font.fallbacks = Some(mono_fallbacks);
            font
        };

        Self {
            body: TextToken {
                font: system_ui.clone(),
                size: rems(1.0).into(),
            },
            caption: TextToken {
                font: system_ui,
                size: rems(0.875).into(),
            },
            mono: TextToken {
                font: mono_font,
                size: rems(0.95).into(),
            },
        }
    }
}
