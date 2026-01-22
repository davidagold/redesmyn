use gpui::{point, px, Pixels, Point};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphCameraLimits {
    pub min_zoom: f32,
    pub max_zoom: f32,
}

impl Default for GraphCameraLimits {
    fn default() -> Self {
        Self {
            min_zoom: 0.55,
            max_zoom: 1.2,
        }
    }
}

/// A simple 2D camera for pan/zoom with explicit world↔screen transforms.
///
/// World space is represented in logical pixels (f32). Screen space is local to the graph canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphCamera {
    /// World-space origin (top-left) that maps to screen-space (0, 0).
    origin_world: Point<f32>,
    zoom: f32,
    limits: GraphCameraLimits,
}

impl GraphCamera {
    #[must_use]
    pub fn new(limits: GraphCameraLimits) -> Self {
        Self {
            origin_world: point(0.0, 0.0),
            zoom: 1.0,
            limits,
        }
    }

    #[must_use]
    pub fn origin_world(&self) -> Point<f32> {
        self.origin_world
    }

    #[must_use]
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    pub fn set_origin_world(&mut self, origin_world: Point<f32>) {
        self.origin_world = origin_world;
    }

    pub fn set_zoom(&mut self, zoom: f32) {
        self.zoom = clamp_zoom(zoom, self.limits);
    }

    #[must_use]
    pub fn world_to_screen(&self, world: Point<f32>) -> Point<Pixels> {
        point(
            px((world.x - self.origin_world.x) * self.zoom),
            px((world.y - self.origin_world.y) * self.zoom),
        )
    }

    #[must_use]
    pub fn screen_to_world(&self, screen: Point<Pixels>) -> Point<f32> {
        point(
            self.origin_world.x + (f32::from(screen.x) / self.zoom),
            self.origin_world.y + (f32::from(screen.y) / self.zoom),
        )
    }

    /// Pan the camera by a screen-space delta (e.g. mouse drag or trackpad scroll).
    ///
    /// Positive deltas move the content in the same direction as the gesture.
    pub fn pan_by_screen_delta(&mut self, delta: Point<Pixels>) {
        self.origin_world.x -= f32::from(delta.x) / self.zoom;
        self.origin_world.y -= f32::from(delta.y) / self.zoom;
    }

    /// Zoom by a multiplicative factor, anchored at a screen-space point (canvas-local).
    pub fn zoom_by_factor_at(&mut self, factor: f32, anchor_screen: Point<Pixels>) {
        if factor <= 0.0 {
            return;
        }

        let anchor_world = self.screen_to_world(anchor_screen);
        let next_zoom = clamp_zoom(self.zoom * factor, self.limits);
        if next_zoom == self.zoom {
            return;
        }

        self.zoom = next_zoom;
        // Preserve the world point under the cursor after the zoom.
        self.origin_world.x = anchor_world.x - (f32::from(anchor_screen.x) / self.zoom);
        self.origin_world.y = anchor_world.y - (f32::from(anchor_screen.y) / self.zoom);
    }
}

fn clamp_zoom(zoom: f32, limits: GraphCameraLimits) -> f32 {
    zoom.clamp(limits.min_zoom, limits.max_zoom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(a: f32, b: f32) {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "expected {a} ~= {b} (diff {diff})");
    }

    #[test]
    fn world_screen_roundtrip() {
        let mut camera = GraphCamera::new(GraphCameraLimits::default());
        camera.set_origin_world(point(12.5, -3.75));
        camera.set_zoom(1.25);

        let world = point(100.0, 200.0);
        let screen = camera.world_to_screen(world);
        let roundtrip = camera.screen_to_world(screen);

        assert_approx(roundtrip.x, world.x);
        assert_approx(roundtrip.y, world.y);
    }

    #[test]
    fn zoom_keeps_anchor_world_point_stable() {
        let mut camera = GraphCamera::new(GraphCameraLimits {
            min_zoom: 0.1,
            max_zoom: 10.0,
        });
        camera.set_origin_world(point(10.0, 20.0));
        camera.set_zoom(1.0);

        let anchor = point(px(240.0), px(180.0));
        let world_before = camera.screen_to_world(anchor);

        camera.zoom_by_factor_at(2.0, anchor);

        let world_after = camera.screen_to_world(anchor);
        assert_approx(world_after.x, world_before.x);
        assert_approx(world_after.y, world_before.y);
    }
}
