use std::sync::Arc;

use gpui::{Entity, Window, div, prelude::*};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;

#[derive(Debug)]
pub struct DesktopModel {
    config: Arc<redesmyn_config::RustConfig>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
    control_plane_client: Option<ClientInProcEndpoint>,
}

impl DesktopModel {
    #[must_use]
    pub fn new(
        config: Arc<redesmyn_config::RustConfig>,
        daemon_host_id: Option<redesmyn_ids::HostId>,
        control_plane_client: Option<ClientInProcEndpoint>,
    ) -> Self {
        Self {
            config,
            daemon_host_id,
            control_plane_client,
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
                    model
                        .daemon_host_id
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "<none>".to_string())
                ))
                .child(format!(
                    "Client transport: {}",
                    if model.control_plane_client.is_some() {
                        "in-proc"
                    } else {
                        "uds"
                    }
                ))
                .child(format!(
                    "Control plane DB: {}",
                    model.config.control_plane.db.path.display()
                ))
                .child(format!(
                    "Client socket: {}",
                    if model.control_plane_client.is_some() {
                        "<disabled (in-proc)>".to_string()
                    } else {
                        model
                            .config
                            .control_plane
                            .api
                            .client_socket_path
                            .display()
                            .to_string()
                    }
                )),
        )
    }
}
