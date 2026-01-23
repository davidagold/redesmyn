use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use gpui::App;
use tokio::sync::watch;

use crate::UiContext;

#[derive(Debug, Clone)]
pub struct UiIdleTracker {
    state: Arc<State>,
}

#[derive(Debug)]
pub struct UiActivityGuard {
    state: Arc<State>,
}

#[derive(Debug)]
struct State {
    active_transitions: AtomicUsize,
    tx: watch::Sender<usize>,
}

impl UiIdleTracker {
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(0);
        Self {
            state: Arc::new(State {
                active_transitions: AtomicUsize::new(0),
                tx,
            }),
        }
    }

    #[must_use]
    pub fn begin_transition(&self) -> UiActivityGuard {
        let next = self.state.active_transitions.fetch_add(1, Ordering::AcqRel) + 1;
        let _ = self.state.tx.send(next);
        UiActivityGuard {
            state: self.state.clone(),
        }
    }

    pub fn active_transitions(&self) -> usize {
        self.state.active_transitions.load(Ordering::Acquire)
    }

    pub fn subscribe(&self) -> watch::Receiver<usize> {
        self.state.tx.subscribe()
    }
}

impl Default for UiIdleTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for UiActivityGuard {
    fn drop(&mut self) {
        let result = self.state.active_transitions.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| current.checked_sub(1),
        );

        match result {
            Ok(prev) => {
                let next = prev.saturating_sub(1);
                let _ = self.state.tx.send(next);
            }
            Err(current) => {
                debug_assert!(
                    false,
                    "UiActivityGuard dropped with no active transitions (current={current})."
                );
            }
        }
    }
}

pub fn ui_idle_tracker(cx: &App) -> Option<UiIdleTracker> {
    cx.try_global::<UiContext>()
        .map(|ui| ui.idle_tracker().clone())
}

#[cfg(test)]
mod tests {
    use super::UiIdleTracker;

    #[test]
    fn transition_guard_increments_and_decrements() {
        let tracker = UiIdleTracker::new();
        assert_eq!(tracker.active_transitions(), 0);

        let guard = tracker.begin_transition();
        assert_eq!(tracker.active_transitions(), 1);

        drop(guard);
        assert_eq!(tracker.active_transitions(), 0);
    }

    #[test]
    fn subscribe_receives_updates() {
        let tracker = UiIdleTracker::new();
        let rx = tracker.subscribe();

        let guard = tracker.begin_transition();
        assert!(rx.has_changed().unwrap_or(false));
        assert_eq!(*rx.borrow(), 1);

        drop(guard);
        assert!(rx.has_changed().unwrap_or(false));
        assert_eq!(*rx.borrow(), 0);
    }
}
