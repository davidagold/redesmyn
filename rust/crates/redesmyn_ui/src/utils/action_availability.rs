use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use gpui::{Action, Context, EntityId, Window};

/// Queries `Window::is_action_available` safely during startup.
///
/// GPUI can panic if `Window::is_action_available` is called before the first rendered dispatch
/// tree is populated. This helper defers the first call until the next rendered frame.
///
/// This allows UI to reflect whether a key-mapped action is currently available based on focus and
/// key contexts, without manually tracking focus state across every possible target widget.
#[derive(Clone, Debug)]
pub struct ActionAvailabilityProbe {
    ready: Arc<AtomicBool>,
    ready_requested: bool,
}

impl Default for ActionAvailabilityProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl ActionAvailabilityProbe {
    pub fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            ready_requested: false,
        }
    }

    /// Returns whether the action is available along the dispatch path to the currently-focused
    /// element.
    ///
    /// Until the probe is ready (after the next frame), this returns `fallback` and schedules a
    /// re-render.
    pub fn is_action_available_or<T: 'static>(
        &mut self,
        window: &mut Window,
        cx: &mut Context<T>,
        action: &dyn Action,
        fallback: bool,
    ) -> bool {
        if !self.ready.load(Ordering::Relaxed) {
            if !self.ready_requested {
                self.ready_requested = true;
                self.schedule_ready(window, cx.entity_id());
            }
            return fallback;
        }

        window.is_action_available(action, cx)
    }

    fn schedule_ready(&self, window: &Window, entity_id: EntityId) {
        let ready = self.ready.clone();
        window.on_next_frame(move |_, cx| {
            ready.store(true, Ordering::Relaxed);
            cx.notify(entity_id);
        });
    }
}
