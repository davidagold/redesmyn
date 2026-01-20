use std::sync::Arc;

use redesmyn_logging::tracing;
use redesmyn_transport::in_proc::InProcEndpoint;

pub struct DesktopApp;

pub struct DesktopHandle {
    config: Arc<redesmyn_config::RustConfig>,
    runtime: tokio::runtime::Runtime,
    control_plane: Option<redesmyn_control_plane::ControlPlaneHandle>,
    daemon: Option<redesmyn_daemon::service::DaemonHandle>,
    daemon_link: Option<crate::daemon_link::DaemonLinkHandle>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
}

#[derive(Debug, thiserror::Error)]
pub enum DesktopStartError {
    #[error("failed to build tokio runtime")]
    TokioRuntime(#[source] std::io::Error),
    #[error(transparent)]
    ControlPlane(#[from] redesmyn_control_plane::ControlPlaneStartError),
}

impl DesktopApp {
    pub fn start(config: redesmyn_config::RustConfig) -> Result<DesktopHandle, DesktopStartError> {
        let span = tracing::info_span!("desktop.start");
        let _enter = span.enter();

        let config = Arc::new(config);

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("redesmyn-desktop")
            .build()
            .map_err(DesktopStartError::TokioRuntime)?;

        let control_plane = if config.desktop.embed_control_plane {
            let options = redesmyn_control_plane::ControlPlaneStartOptions::from_config(&config.control_plane);
            Some(runtime.block_on(redesmyn_control_plane::ControlPlane::start(options))?)
        } else {
            None
        };

        let (daemon, daemon_link, daemon_host_id) = if config.desktop.embed_daemon {
            let (control_plane_conn, daemon_conn) = InProcEndpoint::pair(64);

            let daemon = runtime.block_on(async {
                redesmyn_daemon::service::Daemon::start(config.as_ref().clone(), Box::new(daemon_conn))
            });

            let daemon_host_id = Some(daemon.host_id());

            let daemon_link = Some(crate::daemon_link::DaemonLinkHandle::start(&runtime, control_plane_conn));
            (Some(daemon), daemon_link, daemon_host_id)
        } else {
            (None, None, None)
        };

        tracing::info!("desktop started");

        Ok(DesktopHandle {
            config,
            runtime,
            control_plane,
            daemon,
            daemon_link,
            daemon_host_id,
        })
    }
}

impl DesktopHandle {
    #[must_use]
    pub fn config(&self) -> &Arc<redesmyn_config::RustConfig> {
        &self.config
    }

    #[must_use]
    pub fn daemon_host_id(&self) -> Option<redesmyn_ids::HostId> {
        self.daemon_host_id
    }

    pub fn shutdown(mut self) {
        let span = tracing::info_span!("desktop.shutdown");
        let _enter = span.enter();

        if let Some(link) = self.daemon_link.take() {
            self.runtime.block_on(link.shutdown());
        }

        if let Some(daemon) = self.daemon.take() {
            self.runtime.block_on(daemon.shutdown());
        }

        if let Some(control_plane) = self.control_plane.take() {
            self.runtime.block_on(control_plane.shutdown());
        }

        tracing::info!("desktop shut down");
    }
}
