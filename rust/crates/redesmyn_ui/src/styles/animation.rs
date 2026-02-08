use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationCurve {
    Linear,
    EaseInOut,
    EaseOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationDurations {
    pub fast: Duration,
    pub normal: Duration,
    pub slow: Duration,
}

impl Default for AnimationDurations {
    fn default() -> Self {
        Self {
            fast: Duration::from_millis(120),
            normal: Duration::from_millis(180),
            slow: Duration::from_millis(260),
        }
    }
}
