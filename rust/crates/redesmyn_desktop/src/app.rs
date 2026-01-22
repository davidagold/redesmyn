use std::sync::{Arc, Mutex};

use redesmyn_logging::tracing;
use redesmyn_transport::in_proc::InProcEndpoint;

pub struct DesktopApp;

pub struct DesktopHandle {
    config: Arc<redesmyn_config::RustConfig>,
    runtime: tokio::runtime::Runtime,
    control_plane: Option<redesmyn_control_plane::ControlPlaneHandle>,
    control_plane_client: Option<redesmyn_transport::client::in_proc::InProcEndpoint>,
    daemon: Option<redesmyn_daemon::DaemonHandle>,
    daemon_link: Option<redesmyn_control_plane::DaemonLinkHandle>,
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

        let mut control_plane = if config.desktop.embed_control_plane {
            let mut options = redesmyn_control_plane::ControlPlaneStartOptions::from_config(
                &config.control_plane,
            );
            options.client_api_socket_path = None;
            Some(runtime.block_on(redesmyn_control_plane::ControlPlane::start(options))?)
        } else {
            None
        };

        let control_plane_client = control_plane
            .as_mut()
            .map(|control_plane| control_plane.connect_in_proc_client(64));

        let (daemon, daemon_link, daemon_host_id) = if config.desktop.embed_daemon {
            let (control_plane_conn, daemon_conn) = InProcEndpoint::pair(64);

            let daemon_host_id = redesmyn_ids::HostId::new();
            let daemon_identity = redesmyn_daemon::HostIdentity::new(daemon_host_id);

            let daemon_config = redesmyn_daemon::DaemonRuntimeConfig::new(config.daemon.clone())
                .with_host_identity(daemon_identity);

            let connector_endpoint = Arc::new(Mutex::new(Some(daemon_conn)));
            let connector: Arc<dyn redesmyn_daemon::ControlPlaneConnector> = Arc::new(move || {
                let endpoint = Arc::clone(&connector_endpoint);
                async move {
                    let mut guard = endpoint.lock().expect("lock poisoned");
                    let endpoint = guard
                        .take()
                        .ok_or(redesmyn_transport::TransportError::ChannelClosed)?;
                    Ok(Box::new(endpoint) as Box<dyn redesmyn_transport::DaemonConnection>)
                }
            });

            let daemon_link = Some(redesmyn_control_plane::DaemonLinkHandle::start(
                &runtime,
                control_plane_conn,
            ));

            let daemon = runtime
                .block_on(async { redesmyn_daemon::Daemon::start(daemon_config, connector) });

            (Some(daemon), daemon_link, Some(daemon_host_id))
        } else {
            (None, None, None)
        };

        tracing::info!("desktop started");

        Ok(DesktopHandle {
            config,
            runtime,
            control_plane,
            control_plane_client,
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

    #[must_use]
    pub fn take_control_plane_client(
        &mut self,
    ) -> Option<redesmyn_transport::client::in_proc::InProcEndpoint> {
        self.control_plane_client.take()
    }

    #[must_use]
    pub fn connect_control_plane_client(
        &mut self,
        buffer: usize,
    ) -> Option<redesmyn_transport::client::in_proc::InProcEndpoint> {
        self.control_plane
            .as_mut()
            .map(|control_plane| control_plane.connect_in_proc_client(buffer))
    }

    #[must_use]
    pub fn tokio_handle(&self) -> tokio::runtime::Handle {
        self.runtime.handle().clone()
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
