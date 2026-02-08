use std::time::Duration;

use gpui::{
    App, AsyncApp, Context, Render, RenderOnce, SharedString, Task, Window, div, prelude::*,
};

use crate::utils::{theme_for_window, ui_test_mode_enabled};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPillKind {
    Neutral,
    Accent,
}

impl Default for ProgressPillKind {
    fn default() -> Self {
        Self::Neutral
    }
}

#[derive(IntoElement)]
pub struct ProgressPill {
    label: SharedString,
    kind: ProgressPillKind,
}

impl ProgressPill {
    pub fn new(label: impl Into<SharedString>) -> Self {
        Self {
            label: label.into(),
            kind: ProgressPillKind::default(),
        }
    }

    pub fn kind(mut self, kind: ProgressPillKind) -> Self {
        self.kind = kind;
        self
    }
}

impl RenderOnce for ProgressPill {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let (bg, fg) = match self.kind {
            ProgressPillKind::Neutral => (theme.colors.accent, theme.colors.foreground),
            ProgressPillKind::Accent => (theme.colors.ring, theme.colors.background),
        };

        div()
            .flex()
            .items_center()
            .gap(theme.spacing.xs)
            .px(theme.spacing.sm)
            .py(theme.spacing.xs)
            .rounded(theme.radius.md)
            .bg(bg)
            .text_sm()
            .text_color(fg)
            .child(self.label)
            .child(cx.new(|_| ProgressDots::new()))
    }
}

struct ProgressDots {
    phase: u8,
    task: Option<Task<()>>,
}

impl ProgressDots {
    fn new() -> Self {
        Self {
            phase: 0,
            task: None,
        }
    }
}

impl Render for ProgressDots {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let ui_test_mode = ui_test_mode_enabled();
        let dots = progress_dots_text(ui_test_mode, self.phase);
        if ui_test_mode {
            return div().child(dots);
        }

        if self.task.is_none() {
            self.task = Some(cx.spawn(
                |weak: gpui::WeakEntity<ProgressDots>, cx: &mut AsyncApp| {
                    let cx = cx.clone();
                    async move {
                        loop {
                            gpui::Timer::after(Duration::from_millis(420)).await;
                            let Some(entity) = weak.upgrade() else {
                                break;
                            };
                            if cx
                                .update(|cx| {
                                    entity.update(cx, |this, cx| {
                                        this.phase = (this.phase + 1) % 4;
                                        cx.notify();
                                    })
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                    }
                },
            ));
        }

        div().child(dots)
    }
}

fn progress_dots_text(ui_test_mode: bool, phase: u8) -> &'static str {
    if ui_test_mode {
        return "...";
    }

    match phase {
        0 => "",
        1 => ".",
        2 => "..",
        _ => "...",
    }
}

#[cfg(test)]
mod tests {
    use super::progress_dots_text;

    #[test]
    fn progress_dots_text_is_frozen_in_test_mode() {
        assert_eq!(progress_dots_text(true, 0), "...");
        assert_eq!(progress_dots_text(true, 1), "...");
        assert_eq!(progress_dots_text(true, 2), "...");
        assert_eq!(progress_dots_text(true, 3), "...");
        assert_eq!(progress_dots_text(true, 255), "...");
    }

    #[test]
    fn progress_dots_text_cycles_in_normal_mode() {
        assert_eq!(progress_dots_text(false, 0), "");
        assert_eq!(progress_dots_text(false, 1), ".");
        assert_eq!(progress_dots_text(false, 2), "..");
        assert_eq!(progress_dots_text(false, 3), "...");
        assert_eq!(progress_dots_text(false, 4), "...");
    }
}
