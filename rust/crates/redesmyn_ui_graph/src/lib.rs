//! Graph view scaffolding for GPUI (scene model + camera + hit-testing + renderer).
//!
//! This crate intentionally keeps the scene/camera/hit-testing primitives explicit and
//! testable so downstream work (layout, node views, edge routing) can iterate independently.

#![forbid(unsafe_code)]

mod camera;
mod geometry;
mod hit_test;
mod scene;
mod view;

pub use camera::{GraphCamera, GraphCameraLimits};
pub use hit_test::GraphHit;
pub use scene::{GraphEdgeId, GraphNodeId, GraphScene, GraphSelection};
pub use view::GraphView;
