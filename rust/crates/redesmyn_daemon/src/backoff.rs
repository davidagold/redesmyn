use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Backoff {
    initial: Duration,
    max: Duration,
    factor: f64,
    next: Duration,
}

impl Backoff {
    pub fn new(initial: Duration, max: Duration, factor: f64) -> Self {
        let factor = if factor < 1.0 { 1.0 } else { factor };
        Self {
            initial,
            max,
            factor,
            next: initial,
        }
    }

    pub fn reset(&mut self) {
        self.next = self.initial;
    }

    pub fn next_delay(&mut self) -> Duration {
        let delay = self.next;
        let increased = self.next.mul_f64(self.factor);
        self.next = std::cmp::min(self.max, increased);
        delay
    }
}
