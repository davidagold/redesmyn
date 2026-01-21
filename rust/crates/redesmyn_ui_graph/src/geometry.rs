use gpui::{px, size, Bounds, Pixels, Point};

use crate::scene::GraphSceneNode;
use crate::{GraphCamera, GraphScene};

pub(crate) const DEFAULT_EDGE_THICKNESS_PX: Pixels = px(2.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EdgeSegments {
    pub(crate) segments: [Option<Bounds<Pixels>>; 3],
}

pub(crate) fn node_bounds_in_window(
    scene: &GraphScene,
    camera: &GraphCamera,
    node: &GraphSceneNode,
    canvas_bounds: Bounds<Pixels>,
) -> Bounds<Pixels> {
    let world_origin = scene.node_world_origin(node.id);
    let world_size = scene.node_world_size(node.id);

    let local_origin = camera.world_to_screen(world_origin);
    let local_size = size(
        px(world_size.width as f32 * camera.zoom()),
        px(world_size.height as f32 * camera.zoom()),
    );

    Bounds {
        origin: local_origin,
        size: local_size,
    } + canvas_bounds.origin
}

pub(crate) fn edge_segments_in_window(
    from_bounds: Bounds<Pixels>,
    to_bounds: Bounds<Pixels>,
    thickness: Pixels,
) -> EdgeSegments {
    let from = anchor_right_center(from_bounds);
    let to = anchor_left_center(to_bounds);

    let mid_x = px((f32::from(from.x) + f32::from(to.x)) / 2.0);

    let seg1 = horizontal_segment(from.x, mid_x, from.y, thickness);
    let seg2 = vertical_segment(mid_x, from.y, to.y, thickness);
    let seg3 = horizontal_segment(mid_x, to.x, to.y, thickness);

    EdgeSegments {
        segments: [seg1, seg2, seg3],
    }
}

fn anchor_right_center(bounds: Bounds<Pixels>) -> Point<Pixels> {
    let y = bounds.top() + bounds.size.height / 2.0;
    gpui::point(bounds.right(), y)
}

fn anchor_left_center(bounds: Bounds<Pixels>) -> Point<Pixels> {
    let y = bounds.top() + bounds.size.height / 2.0;
    gpui::point(bounds.left(), y)
}

fn horizontal_segment(
    x1: Pixels,
    x2: Pixels,
    y: Pixels,
    thickness: Pixels,
) -> Option<Bounds<Pixels>> {
    let (left, right) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
    let width = right - left;
    if width <= px(0.0) {
        return None;
    }

    Some(Bounds {
        origin: gpui::point(left, y - thickness / 2.0),
        size: size(width, thickness),
    })
}

fn vertical_segment(
    x: Pixels,
    y1: Pixels,
    y2: Pixels,
    thickness: Pixels,
) -> Option<Bounds<Pixels>> {
    let (top, bottom) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
    let height = bottom - top;
    if height <= px(0.0) {
        return None;
    }

    Some(Bounds {
        origin: gpui::point(x - thickness / 2.0, top),
        size: size(thickness, height),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphCameraLimits, GraphNodeId};
    use redesmyn_ids::TaskId;

    #[test]
    fn edge_segments_include_midpoint_route() {
        let from = Bounds {
            origin: gpui::point(px(0.0), px(0.0)),
            size: size(px(10.0), px(10.0)),
        };
        let to = Bounds {
            origin: gpui::point(px(50.0), px(30.0)),
            size: size(px(10.0), px(10.0)),
        };

        let segments = edge_segments_in_window(from, to, px(2.0));
        assert!(segments.segments.iter().flatten().count() >= 2);
    }

    #[test]
    fn node_bounds_scales_with_zoom() {
        let mut scene = GraphScene::empty_demo();
        let id = GraphNodeId::Task(TaskId::from_bytes([1; 16]));
        scene.insert_demo_node(id);

        let mut camera = GraphCamera::new(GraphCameraLimits::default());
        camera.set_zoom(2.0);

        let bounds =
            node_bounds_in_window(&scene, &camera, scene.node(id).unwrap(), Bounds::default());
        assert!(bounds.size.width > px(0.0));
        assert!(bounds.size.height > px(0.0));
    }
}
