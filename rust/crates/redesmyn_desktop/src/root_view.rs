use std::sync::Arc;

use gpui::{div, prelude::*, Entity, Window};

#[derive(Debug)]
pub struct DesktopModel {
    config: Arc<redesmyn_config::RustConfig>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
}

impl DesktopModel {
    #[must_use]
    pub fn new(config: Arc<redesmyn_config::RustConfig>, daemon_host_id: Option<redesmyn_ids::HostId>) -> Self {
        Self {
            config,
            daemon_host_id,
        }
    }
}

pub struct RootView {
    model: Entity<DesktopModel>,
}

impl RootView {
    #[must_use]
    pub fn new(model: Entity<DesktopModel>) -> Self {
        Self { model }
    }
}

impl Render for RootView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let model = self.model.read(cx);

        div().child("Redesmyn Desktop (bootstrap)").child(
            div()
                .child(format!(
                    "Control plane: {}",
                    if model.config.desktop.embed_control_plane {
                        "embedded"
                    } else {
                        "external"
                    }
                ))
                .child(format!(
                    "Daemon: {}",
                    if model.config.desktop.embed_daemon {
                        "embedded"
                    } else {
                        "external"
                    }
                ))
                .child(format!(
                    "Daemon host id: {}",
                    model.daemon_host_id
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "<none>".to_string())
                ))
                .child(format!(
                    "Control plane DB: {}",
                    model.config.control_plane.db.path.display()
                ))
                .child(format!(
                    "Client socket: {}",
                    model.config.control_plane.api.client_socket_path.display()
                )),
        )
    }
}
