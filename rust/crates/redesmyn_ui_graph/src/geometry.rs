use gpui::{Bounds, Pixels, Point, px, size};

use crate::scene::GraphSceneNode;
use crate::{GraphCamera, GraphScene};

pub(crate) const DEFAULT_EDGE_STROKE_PX: Pixels = px(2.0);
pub(crate) const DEFAULT_EDGE_INTERACTION_WIDTH_PX: Pixels = px(24.0);

pub(crate) const EDGE_LABEL_ZOOM: f32 = 0.8;
pub(crate) const EDGE_TICKS_ZOOM: f32 = 1.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EdgeLodBand {
    Low,
    Labels,
    Ticks,
}

pub(crate) fn edge_lod_band(zoom: f32) -> EdgeLodBand {
    if zoom < EDGE_LABEL_ZOOM {
        return EdgeLodBand::Low;
    }
    if zoom < EDGE_TICKS_ZOOM {
        return EdgeLodBand::Labels;
    }
    EdgeLodBand::Ticks
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EdgeRoute {
    pub(crate) points: [Point<Pixels>; 4],
}

impl EdgeRoute {
    pub(crate) fn segments(&self) -> [(Point<Pixels>, Point<Pixels>); 3] {
        let [a, b, c, d] = self.points;
        [(a, b), (b, c), (c, d)]
    }

    pub(crate) fn segment_bounds(&self, thickness: Pixels) -> [Option<Bounds<Pixels>>; 3] {
        let [a, b, c, d] = self.points;
        [
            horizontal_segment(a.x, b.x, a.y, thickness),
            vertical_segment(b.x, b.y, c.y, thickness),
            horizontal_segment(c.x, d.x, d.y, thickness),
        ]
    }

    #[must_use]
    pub(crate) fn label_center(&self) -> Point<Pixels> {
        let [_, b, c, _] = self.points;
        let y = px((f32::from(b.y) + f32::from(c.y)) / 2.0);
        gpui::point(b.x, y)
    }
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

pub(crate) fn edge_route_in_window(
    from_bounds: Bounds<Pixels>,
    to_bounds: Bounds<Pixels>,
) -> EdgeRoute {
    let from = anchor_right_center(from_bounds);
    let to = anchor_left_center(to_bounds);

    let mid_x = px((f32::from(from.x) + f32::from(to.x)) / 2.0);

    EdgeRoute {
        points: [
            from,
            gpui::point(mid_x, from.y),
            gpui::point(mid_x, to.y),
            to,
        ],
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

        let route = edge_route_in_window(from, to);
        let segments = route.segment_bounds(px(2.0));
        assert!(segments.iter().flatten().count() >= 2);
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

    #[test]
    fn edge_lod_matches_reference_thresholds() {
        assert_eq!(edge_lod_band(0.79), EdgeLodBand::Low);
        assert_eq!(edge_lod_band(0.8), EdgeLodBand::Labels);
        assert_eq!(edge_lod_band(1.04), EdgeLodBand::Labels);
        assert_eq!(edge_lod_band(1.05), EdgeLodBand::Ticks);
    }
}
