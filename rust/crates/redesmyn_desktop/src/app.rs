use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use redesmyn_config::DesktopFixtureMode;
use redesmyn_control_plane::session_events::SessionEvents;
use redesmyn_logging::tracing;
use redesmyn_protocol::session::{
    AssistantMessage, SessionEventKind, SessionScope, SessionStarted, UserMessage,
};
use redesmyn_protocol::{SessionEvent, Timestamp};
use redesmyn_storage::StorageError;
use redesmyn_transport::in_proc::InProcEndpoint;

pub struct DesktopApp;

const SESSION_VIEWER_FIXTURE_SESSION_ID: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
const SESSION_VIEWER_FIXTURE_BASE_MS: i64 = 1_700_000_000_000;
const SESSION_VIEWER_FIXTURE_TURN_COUNT: u64 = 65;

#[derive(Clone)]
pub struct SessionViewerFixtureEmitter {
    tokio: tokio::runtime::Handle,
    session_events: SessionEvents,
    session_id: redesmyn_ids::SessionId,
    demo_counter: Arc<AtomicU64>,
}

impl std::fmt::Debug for SessionViewerFixtureEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionViewerFixtureEmitter")
            .field("session_id", &self.session_id)
            .finish_non_exhaustive()
    }
}

impl SessionViewerFixtureEmitter {
    #[must_use]
    pub fn session_id(&self) -> redesmyn_ids::SessionId {
        self.session_id
    }

    #[must_use]
    pub fn tokio(&self) -> tokio::runtime::Handle {
        self.tokio.clone()
    }

    pub async fn emit_demo_message(&self) -> Result<(), StorageError> {
        let ix = self.demo_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let created_at = Timestamp::now_utc();
        let kind = if ix % 2 == 0 {
            SessionEventKind::AssistantMessage(AssistantMessage {
                text: format!("Demo assistant message #{ix}"),
                preview: format!("Demo assistant message #{ix}"),
                full_text_artifact: None,
            })
        } else {
            SessionEventKind::UserMessage(UserMessage {
                text: format!("Demo user message #{ix}"),
                preview: format!("Demo user message #{ix}"),
                full_text_artifact: None,
            })
        };

        let event = SessionEvent {
            session_event_id: redesmyn_ids::SessionEventId::new(),
            created_at,
            scope: SessionScope::Chat,
            session_id: self.session_id,
            turn_id: None,
            kind,
        };

        self.session_events.append_session_event(&event).await?;

        tracing::info!(
            session_id = %self.session_id,
            session_event_id = %event.session_event_id,
            "emitted session viewer demo event"
        );

        Ok(())
    }

    fn new(tokio: tokio::runtime::Handle, session_events: SessionEvents) -> Self {
        let session_id = session_viewer_fixture_session_id();
        Self {
            tokio,
            session_events,
            session_id,
            demo_counter: Arc::new(AtomicU64::new(0)),
        }
    }
}

pub struct DesktopHandle {
    config: Arc<redesmyn_config::RustConfig>,
    runtime: tokio::runtime::Runtime,
    control_plane: Option<redesmyn_control_plane::ControlPlaneHandle>,
    control_plane_client: Option<redesmyn_transport::client::in_proc::InProcEndpoint>,
    session_viewer_fixture: Option<SessionViewerFixtureEmitter>,
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
    #[error("desktop fixture mode requires an embedded control plane")]
    FixtureRequiresEmbeddedControlPlane,
    #[error("failed to reset fixture sqlite DB at {path}")]
    FixtureDbReset {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to seed session viewer fixture data")]
    FixtureSeed(#[from] StorageError),
}

impl DesktopApp {
    pub fn start(
        mut config: redesmyn_config::RustConfig,
    ) -> Result<DesktopHandle, DesktopStartError> {
        let span = tracing::info_span!("desktop.start");
        let _enter = span.enter();

        let fixture_mode = config.desktop.fixture;
        if fixture_mode == Some(DesktopFixtureMode::SessionViewer) && !config.desktop.embed_control_plane
        {
            return Err(DesktopStartError::FixtureRequiresEmbeddedControlPlane);
        }

        if fixture_mode == Some(DesktopFixtureMode::SessionViewer) {
            let fixture_db_path = session_viewer_fixture_db_path(&config.control_plane.db.path);
            reset_fixture_sqlite_db(&fixture_db_path).map_err(|source| {
                DesktopStartError::FixtureDbReset {
                    path: fixture_db_path.clone(),
                    source,
                }
            })?;

            tracing::info!(
                db_path = %fixture_db_path.display(),
                "using session viewer fixture DB"
            );
            config.control_plane.db.path = fixture_db_path;
        }

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

        let session_viewer_fixture = if fixture_mode == Some(DesktopFixtureMode::SessionViewer) {
            let session_events = control_plane
                .as_ref()
                .expect("fixture mode requires an embedded control plane")
                .session_events();
            let session_id = session_viewer_fixture_session_id();

            let seeded = runtime.block_on(seed_session_viewer_fixture(
                &session_events,
                session_id,
            ))?;

            tracing::info!(
                session_id = %session_id,
                events = seeded,
                "seeded session viewer fixture data"
            );

            Some(SessionViewerFixtureEmitter::new(
                runtime.handle().clone(),
                session_events,
            ))
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
            session_viewer_fixture,
            daemon,
            daemon_link,
            daemon_host_id,
        })
    }
}

fn session_viewer_fixture_session_id() -> redesmyn_ids::SessionId {
    redesmyn_ids::SessionId::from_str(SESSION_VIEWER_FIXTURE_SESSION_ID)
        .expect("fixture session id should parse")
}

fn session_viewer_fixture_db_path(base_db_path: &Path) -> PathBuf {
    let Some(state_dir) = base_db_path.parent() else {
        return PathBuf::from(".redesmyn/fixtures/session_viewer/redesmyn_rust.sqlite3");
    };

    state_dir
        .join("fixtures")
        .join("session_viewer")
        .join("redesmyn_rust.sqlite3")
}

fn reset_fixture_sqlite_db(db_path: &Path) -> Result<(), std::io::Error> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    for suffix in ["", "-wal", "-shm"] {
        let mut path = db_path.as_os_str().to_os_string();
        path.push(suffix);
        let path = PathBuf::from(path);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }

    Ok(())
}

fn seeded_timestamp(index: u64) -> Timestamp {
    let offset_ms = i64::try_from(index)
        .unwrap_or(i64::MAX)
        .saturating_mul(1_000);
    let ms = SESSION_VIEWER_FIXTURE_BASE_MS.saturating_add(offset_ms);
    Timestamp::from_unix_millis(ms).expect("fixture timestamp should be in range")
}

fn seeded_ms(index: u64) -> u64 {
    u64::try_from(SESSION_VIEWER_FIXTURE_BASE_MS)
        .expect("fixture base ms should be non-negative")
        .saturating_add(index.saturating_mul(1_000))
}

fn seeded_session_event_id(created_at_ms: u64, counter: u64) -> redesmyn_ids::SessionEventId {
    let mut bytes = [0_u8; 16];

    let ts_bytes = created_at_ms.to_be_bytes();
    bytes[0..6].copy_from_slice(&ts_bytes[2..]);

    let rand_bytes = u128::from(counter).to_be_bytes();
    bytes[6..].copy_from_slice(&rand_bytes[6..]);

    redesmyn_ids::SessionEventId::from_bytes(bytes)
}

async fn seed_session_viewer_fixture(
    session_events: &SessionEvents,
    session_id: redesmyn_ids::SessionId,
) -> Result<usize, StorageError> {
    let mut seeded = 0_usize;
    let mut ix = 0_u64;

    let started = SessionEvent {
        session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
        created_at: seeded_timestamp(ix),
        scope: SessionScope::Chat,
        session_id,
        turn_id: None,
        kind: SessionEventKind::SessionStarted(SessionStarted {}),
    };
    session_events.append_session_event(&started).await?;
    seeded += 1;
    ix += 1;

    for turn in 0..SESSION_VIEWER_FIXTURE_TURN_COUNT {
        let user_preview = format!("Fixture user message #{turn}");
        let user = SessionEvent {
            session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
            created_at: seeded_timestamp(ix),
            scope: SessionScope::Chat,
            session_id,
            turn_id: None,
            kind: SessionEventKind::UserMessage(UserMessage {
                text: format!(
                    "{user_preview}\n\n- bullet one\n- bullet two\n\n`inline_code()`"
                ),
                preview: user_preview,
                full_text_artifact: None,
            }),
        };
        session_events.append_session_event(&user).await?;
        seeded += 1;
        ix += 1;

        let assistant_preview = format!("Fixture assistant reply #{turn}");
        let assistant = SessionEvent {
            session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
            created_at: seeded_timestamp(ix),
            scope: SessionScope::Chat,
            session_id,
            turn_id: None,
            kind: SessionEventKind::AssistantMessage(AssistantMessage {
                text: format!(
                    "{assistant_preview}\n\n```rust\nfn hello() {{\n    println!(\"hi\");\n}}\n```"
                ),
                preview: assistant_preview,
                full_text_artifact: None,
            }),
        };
        session_events.append_session_event(&assistant).await?;
        seeded += 1;
        ix += 1;
    }

    Ok(seeded)
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
    pub fn take_session_viewer_fixture(&mut self) -> Option<SessionViewerFixtureEmitter> {
        self.session_viewer_fixture.take()
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
