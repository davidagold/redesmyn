use std::sync::Arc;
use std::time::Duration;

use redesmyn_config::DaemonConfig;
use redesmyn_git::{GitBackend, GitCliBackend};
use redesmyn_logging::tracing;
use redesmyn_protocol::ProtocolVersion;
use redesmyn_protocol::RepoScope;
use redesmyn_protocol::daemon::{CommandDispatch, DaemonFrame};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::DaemonCapabilities;
use crate::backoff::Backoff;
use crate::command_router::run_command_router;
use crate::control_plane::{
    ConnectionState, ControlPlaneConnector, run_control_plane_connection_manager,
};
use crate::host_identity::{HostIdentity, load_or_create_host_id};
use crate::repo::{
    RepoAttachError, RepoAttachmentManager, RepoDetachError, RepoRegistry, UnconfiguredRepoRegistry,
};

#[derive(Debug, Clone)]
pub struct BackoffConfig {
    pub initial: Duration,
    pub max: Duration,
    pub factor: f64,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial: Duration::from_millis(250),
            max: Duration::from_secs(10),
            factor: 2.0,
        }
    }
}

#[derive(Clone)]
pub struct DaemonRuntimeConfig {
    pub daemon: DaemonConfig,
    pub host_identity: Option<HostIdentity>,
    pub capabilities: DaemonCapabilities,
    pub supported_protocol: ProtocolVersion,
    pub backoff: BackoffConfig,
    pub repo_registry: Arc<dyn RepoRegistry>,
    pub git_backend: Arc<dyn GitBackend>,
}

impl DaemonRuntimeConfig {
    #[must_use]
    pub fn new(daemon: DaemonConfig) -> Self {
        Self {
            daemon,
            host_identity: None,
            capabilities: DaemonCapabilities {
                supports_repo_execution: true,
                supports_worktrees: false,
                supports_git_observation: false,
                supports_session_exec: true,
                supports_session_attach_tmux: false,
                supports_artifacts: false,
            },
            supported_protocol: ProtocolVersion::CURRENT,
            backoff: BackoffConfig::default(),
            repo_registry: Arc::new(UnconfiguredRepoRegistry),
            git_backend: Arc::new(GitCliBackend::new()),
        }
    }

    #[must_use]
    pub fn with_repo_registry(mut self, repo_registry: Arc<dyn RepoRegistry>) -> Self {
        self.repo_registry = repo_registry;
        self
    }

    #[must_use]
    pub fn with_host_identity(mut self, identity: HostIdentity) -> Self {
        self.host_identity = Some(identity);
        self
    }

    #[must_use]
    pub fn with_git_backend(mut self, git_backend: Arc<dyn GitBackend>) -> Self {
        self.git_backend = git_backend;
        self
    }
}

pub struct Daemon;

impl Daemon {
    #[must_use]
    pub fn start(
        config: DaemonRuntimeConfig,
        connector: Arc<dyn ControlPlaneConnector>,
    ) -> DaemonHandle {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (connection_state_tx, connection_state_rx) =
            watch::channel(ConnectionState::Disconnected);

        let host_identity = config.host_identity.unwrap_or_else(|| {
            let state_dir = config
                .daemon
                .repo_registry_dir
                .parent()
                .unwrap_or(&config.daemon.repo_registry_dir);
            match load_or_create_host_id(state_dir) {
                Ok(host_id) => HostIdentity::new(host_id),
                Err(err) => {
                    tracing::error!(error = %err, "failed to load host identity; using ephemeral id");
                    HostIdentity::new(redesmyn_ids::HostId::new())
                }
            }
        });

        let (command_dispatch_tx, command_dispatch_rx) = mpsc::channel::<CommandDispatch>(32);
        let (frames_tx, frames_rx) = mpsc::channel::<DaemonFrame>(256);

        let control_plane_task = tokio::spawn(run_control_plane_connection_manager(
            connector,
            host_identity,
            config.capabilities,
            config.supported_protocol,
            Backoff::new(
                config.backoff.initial,
                config.backoff.max,
                config.backoff.factor,
            ),
            command_dispatch_tx,
            frames_rx,
            shutdown_tx.clone(),
            shutdown_rx.clone(),
            connection_state_tx,
        ));

        let command_router_task = tokio::spawn(run_command_router(
            host_identity,
            config.daemon.clone(),
            config.repo_registry.clone(),
            config.git_backend.clone(),
            command_dispatch_rx,
            frames_tx,
            shutdown_rx.clone(),
        ));

        let (repo_cmd_tx, repo_cmd_rx) = mpsc::channel::<RepoCommand>(32);
        let repo_manager_task = tokio::spawn(run_repo_manager(
            host_identity,
            config.repo_registry,
            config.git_backend,
            repo_cmd_rx,
            shutdown_rx,
        ));

        DaemonHandle {
            shutdown_tx,
            tasks: vec![control_plane_task, command_router_task, repo_manager_task],
            repo_cmd_tx,
            connection_state_rx,
        }
    }
}

pub struct DaemonHandle {
    shutdown_tx: watch::Sender<bool>,
    tasks: Vec<JoinHandle<()>>,
    repo_cmd_tx: mpsc::Sender<RepoCommand>,
    connection_state_rx: watch::Receiver<ConnectionState>,
}

impl DaemonHandle {
    pub fn connection_state(&self) -> watch::Receiver<ConnectionState> {
        self.connection_state_rx.clone()
    }

    pub async fn attach_repo(&self, scope: RepoScope) -> Result<(), RepoAttachError> {
        let (tx, rx) = oneshot::channel();
        self.repo_cmd_tx
            .send(RepoCommand::Attach { scope, reply: tx })
            .await
            .map_err(|_| RepoAttachError::DaemonUnavailable)?;
        rx.await.map_err(|_| RepoAttachError::DaemonUnavailable)?
    }

    pub async fn detach_repo(&self, scope: RepoScope) -> Result<(), RepoDetachError> {
        let (tx, rx) = oneshot::channel();
        self.repo_cmd_tx
            .send(RepoCommand::Detach { scope, reply: tx })
            .await
            .map_err(|_| RepoDetachError::DaemonUnavailable)?;
        rx.await.map_err(|_| RepoDetachError::DaemonUnavailable)?
    }

    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        for task in self.tasks {
            let _ = task.await;
        }
    }
}

enum RepoCommand {
    Attach {
        scope: RepoScope,
        reply: oneshot::Sender<Result<(), RepoAttachError>>,
    },
    Detach {
        scope: RepoScope,
        reply: oneshot::Sender<Result<(), RepoDetachError>>,
    },
}

async fn run_repo_manager(
    identity: HostIdentity,
    registry: Arc<dyn RepoRegistry>,
    git_backend: Arc<dyn GitBackend>,
    mut rx: mpsc::Receiver<RepoCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let span = redesmyn_logging::redesmyn_info_span!("daemon.repo.manager");
    redesmyn_logging::span::record_host_id(&span, identity.host_id);
    redesmyn_logging::span::record_host_instance_id(&span, identity.host_instance_id);
    let _enter = span.enter();

    let mut manager = RepoAttachmentManager::new(registry, git_backend);

    loop {
        if *shutdown_rx.borrow() {
            return;
        }

        tokio::select! {
            _ = shutdown_rx.changed() => {}
            cmd = rx.recv() => {
                let Some(cmd) = cmd else { return; };
                match cmd {
                    RepoCommand::Attach { scope, reply } => {
                        let result = manager.attach(scope, shutdown_rx.clone());
                        let _ = reply.send(result);
                    }
                    RepoCommand::Detach { scope, reply } => {
                        let result = manager.detach(scope);
                        let _ = reply.send(result);
                    }
                }
            }
        }
    }
}
