use gpui::{Bounds, Pixels, Point};

use crate::geometry::{
    DEFAULT_EDGE_INTERACTION_WIDTH_PX, edge_route_in_window, node_bounds_in_window,
};
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
        if node.id == GraphNodeId::Trunk {
            continue;
        }
        let bounds = node_bounds_in_window(scene, camera, node, canvas_bounds);
        if bounds.contains(&window_point) {
            return Some(GraphHit::Node(node.id));
        }
    }

    let hit_radius_sq = {
        let r = f32::from(DEFAULT_EDGE_INTERACTION_WIDTH_PX) / 2.0;
        r * r
    };
    for edge in scene.edges() {
        if edge.id.from == GraphNodeId::Trunk || edge.id.to == GraphNodeId::Trunk {
            continue;
        }
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

        let route = edge_route_in_window(from_bounds, to_bounds);
        for (a, b) in route.segments() {
            if point_distance_sq_to_segment(window_point, a, b) <= hit_radius_sq {
                return Some(GraphHit::Edge(edge.id));
            }
        }
    }

    None
}

fn point_distance_sq_to_segment(p: Point<Pixels>, a: Point<Pixels>, b: Point<Pixels>) -> f32 {
    let ax = f32::from(a.x);
    let ay = f32::from(a.y);
    let bx = f32::from(b.x);
    let by = f32::from(b.y);
    let px = f32::from(p.x);
    let py = f32::from(p.y);

    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;

    let ab_len_sq = abx * abx + aby * aby;
    if ab_len_sq == 0.0 {
        return apx * apx + apy * apy;
    }

    let t = ((apx * abx) + (apy * aby)) / ab_len_sq;
    let t = t.clamp(0.0, 1.0);
    let closest_x = ax + (t * abx);
    let closest_y = ay + (t * aby);

    let dx = px - closest_x;
    let dy = py - closest_y;
    (dx * dx) + (dy * dy)
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
            origin: gpui::point(gpui::px(0.0), gpui::px(0.0)),
            size: gpui::size(gpui::px(800.0), gpui::px(600.0)),
        };

        let first_node = scene.nodes().next().unwrap();
        let bounds = node_bounds_in_window(&scene, &camera, first_node, canvas);
        let point = gpui::point(bounds.left() + gpui::px(4.0), bounds.top() + gpui::px(4.0));

        assert_eq!(
            hit_test(&scene, &camera, canvas, point),
            Some(GraphHit::Node(first_node.id))
        );
    }

    #[test]
    fn hit_test_finds_demo_edge() {
        let scene = GraphScene::demo();
        let camera = GraphCamera::new(GraphCameraLimits::default());
        let canvas = Bounds {
            origin: gpui::point(gpui::px(0.0), gpui::px(0.0)),
            size: gpui::size(gpui::px(800.0), gpui::px(600.0)),
        };

        let edge = scene.edges().next().unwrap();
        let from = scene.node(edge.id.from).unwrap();
        let to = scene.node(edge.id.to).unwrap();
        let from_bounds = node_bounds_in_window(&scene, &camera, from, canvas);
        let to_bounds = node_bounds_in_window(&scene, &camera, to, canvas);
        let route = edge_route_in_window(from_bounds, to_bounds);

        let point = route.label_center();
        assert_eq!(
            hit_test(&scene, &camera, canvas, point),
            Some(GraphHit::Edge(edge.id))
        );
    }

    #[test]
    fn hit_test_ignores_trunk_node_and_edges() {
        let scene = GraphScene::demo();
        let camera = GraphCamera::new(GraphCameraLimits::default());
        let canvas = Bounds {
            origin: gpui::point(gpui::px(0.0), gpui::px(0.0)),
            size: gpui::size(gpui::px(800.0), gpui::px(600.0)),
        };

        // The demo trunk is at (40, 40) and task nodes are offset to the right, so this point is
        // safely inside the trunk column but outside task cards.
        let point = gpui::point(gpui::px(60.0), gpui::px(60.0));
        assert_eq!(hit_test(&scene, &camera, canvas, point), None);
    }
}
