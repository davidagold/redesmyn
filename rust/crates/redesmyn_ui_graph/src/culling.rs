use gpui::{Bounds, Pixels, Point, Size};

pub const DEFAULT_NODE_CULLING_OVERSCAN_PX: f32 = 200.0;

pub fn viewport_bounds_with_overscan(
    canvas_bounds: Bounds<Pixels>,
    overscan: Pixels,
) -> Bounds<Pixels> {
    let overscan_2x = overscan + overscan;
    Bounds {
        origin: Point {
            x: canvas_bounds.origin.x - overscan,
            y: canvas_bounds.origin.y - overscan,
        },
        size: Size {
            width: canvas_bounds.size.width + overscan_2x,
            height: canvas_bounds.size.height + overscan_2x,
        },
    }
}

