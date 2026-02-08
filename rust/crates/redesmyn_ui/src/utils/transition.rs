use std::collections::HashMap;
use std::hash::Hash;
use std::time::{Duration, Instant};

use gpui::Window;

#[derive(Debug, Clone, Copy)]
pub struct TransitionF32 {
    started_at: Instant,
    from: f32,
    to: f32,
    duration: Duration,
}

impl TransitionF32 {
    fn value(&self) -> f32 {
        if self.duration == Duration::from_millis(0) {
            return self.to;
        }

        let elapsed = self.started_at.elapsed().as_secs_f32();
        let total = self.duration.as_secs_f32();
        let t = ease_out_cubic((elapsed / total).clamp(0.0, 1.0));
        lerp_f32(self.from, self.to, t)
    }

    fn is_complete(&self) -> bool {
        if self.duration == Duration::from_millis(0) {
            return true;
        }

        self.started_at.elapsed() >= self.duration
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TransitionStateF32 {
    value: f32,
    transition: Option<TransitionF32>,
}

#[derive(Debug)]
pub struct TransitionMap<K> {
    states: HashMap<K, TransitionStateF32>,
}

impl<K> Default for TransitionMap<K> {
    fn default() -> Self {
        Self {
            states: HashMap::new(),
        }
    }
}

impl<K> TransitionMap<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retain(&mut self, mut f: impl FnMut(&K) -> bool) {
        self.states.retain(|key, _| f(key));
    }

    pub fn remove(&mut self, key: &K) {
        self.states.remove(key);
    }

    pub fn opacity_for_render(
        &mut self,
        key: K,
        visible: bool,
        duration: Duration,
        window: &Window,
    ) -> f32 {
        let target = if visible { 1.0 } else { 0.0 };
        // Avoid allocating and immediately removing transition state for keys that are stably
        // hidden. This is hot in graph views where we compute hover affordance opacity for many
        // nodes per frame.
        if !visible && !self.states.contains_key(&key) {
            return 0.0;
        }

        let value = self.value_for_render(key.clone(), target, duration, window);

        if !visible && value.abs() < 1e-3 {
            self.states.remove(&key);
            0.0
        } else {
            value
        }
    }

    pub fn value_for_render(
        &mut self,
        key: K,
        target: f32,
        duration: Duration,
        window: &Window,
    ) -> f32 {
        let state = self.states.entry(key).or_default();

        if let Some(transition) = state.transition {
            state.value = transition.value();
            if transition.is_complete() {
                state.transition = None;
            } else {
                window.request_animation_frame();
            }
        }

        let active_target = state.transition.map(|transition| transition.to);
        if active_target != Some(target) && (state.value - target).abs() > 1e-3 {
            if duration == Duration::from_millis(0) {
                state.value = target;
                state.transition = None;
            } else {
                state.transition = Some(TransitionF32 {
                    started_at: Instant::now(),
                    from: state.value,
                    to: target,
                    duration,
                });
                window.request_animation_frame();
            }
        }

        state.value
    }
}

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
