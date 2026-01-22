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
        if ui_test_mode_enabled() {
            return div().child("...");
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

        let dots = match self.phase {
            0 => "",
            1 => ".",
            2 => "..",
            _ => "...",
        };

        div().child(dots)
    }
}
