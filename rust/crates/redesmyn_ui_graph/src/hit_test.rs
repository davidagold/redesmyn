use gpui::{Bounds, Pixels, Point, px};

use crate::geometry::{DEFAULT_EDGE_THICKNESS_PX, edge_segments_in_window, node_bounds_in_window};
use crate::{GraphCamera, GraphEdgeId, GraphNodeId, GraphScene};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphHit {
    Node(GraphNodeId),
    Edge(GraphEdgeId),
}

pub fn hit_test(
    scene: &GraphScene,
    camera: &GraphCamera,
    canvas_bounds: Bounds<Pixels>,
    window_point: Point<Pixels>,
) -> Option<GraphHit> {
    for node in scene.nodes() {
        let bounds = node_bounds_in_window(scene, camera, node, canvas_bounds);
        if bounds.contains(&window_point) {
            return Some(GraphHit::Node(node.id));
        }
    }

    let tolerance = px(3.0);
    for edge in scene.edges() {
        let Some(from_bounds) = scene
            .node(edge.id.from)
            .map(|node| node_bounds_in_window(scene, camera, node, canvas_bounds))
        else {
            continue;
        };
        let Some(to_bounds) = scene
            .node(edge.id.to)
            .map(|node| node_bounds_in_window(scene, camera, node, canvas_bounds))
        else {
            continue;
        };

        let segments = edge_segments_in_window(from_bounds, to_bounds, DEFAULT_EDGE_THICKNESS_PX);
        for segment in segments.segments.iter().flatten() {
            if segment.dilate(tolerance).contains(&window_point) {
                return Some(GraphHit::Edge(edge.id));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphCameraLimits, GraphScene};

    #[test]
    fn hit_test_finds_demo_node() {
        let scene = GraphScene::demo();
        let camera = GraphCamera::new(GraphCameraLimits::default());
        let canvas = Bounds {
            origin: gpui::point(px(0.0), px(0.0)),
            size: gpui::size(px(800.0), px(600.0)),
        };

        let first_node = scene.nodes().next().unwrap();
        let bounds = node_bounds_in_window(&scene, &camera, first_node, canvas);
        let point = gpui::point(bounds.left() + px(4.0), bounds.top() + px(4.0));

        assert_eq!(
            hit_test(&scene, &camera, canvas, point),
            Some(GraphHit::Node(first_node.id))
        );
    }
}
