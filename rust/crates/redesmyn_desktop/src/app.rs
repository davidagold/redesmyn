use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use redesmyn_config::DesktopFixtureMode;
use redesmyn_control_plane::session_events::SessionEvents;
use redesmyn_logging::tracing;
use redesmyn_protocol::session::{AssistantMessage, SessionEventKind, SessionScope, UserMessage};
#[cfg(debug_assertions)]
use redesmyn_protocol::session::{
    ExternalSessionRef, InterfaceMode, SessionStarted, ToolInvocation, ToolResult, TurnCompleted,
    TurnStarted,
};
use redesmyn_protocol::{SessionEvent, Timestamp};
use redesmyn_storage::StorageError;
use redesmyn_transport::in_proc::InProcEndpoint;

#[cfg(debug_assertions)]
use std::path::Path;
#[cfg(debug_assertions)]
use std::str::FromStr as _;

pub struct DesktopApp;

#[cfg(debug_assertions)]
const SESSION_VIEWER_FIXTURE_SESSION_ID: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";
#[cfg(debug_assertions)]
const SESSION_VIEWER_FIXTURE_WORKSPACE_ID: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAW";
#[cfg(debug_assertions)]
const SESSION_VIEWER_FIXTURE_REPO_ID: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAX";
#[cfg(debug_assertions)]
const SESSION_VIEWER_FIXTURE_BASE_MS: i64 = 1_700_000_000_000;
#[cfg(debug_assertions)]
const SESSION_VIEWER_FIXTURE_TURN_COUNT: u64 = 30;

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
                image_attachments: Vec::new(),
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

    #[cfg(debug_assertions)]
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
    #[cfg_attr(debug_assertions, allow(dead_code))]
    #[error("desktop fixture mode is only supported in debug builds")]
    FixtureNotSupportedInRelease,
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

        #[cfg(not(debug_assertions))]
        if fixture_mode.is_some() {
            tracing::warn!(
                fixture = ?fixture_mode,
                "desktop fixture mode requested in a release build; refusing"
            );
            return Err(DesktopStartError::FixtureNotSupportedInRelease);
        }

        if fixture_mode == Some(DesktopFixtureMode::SessionViewer)
            && !config.desktop.embed_control_plane
        {
            return Err(DesktopStartError::FixtureRequiresEmbeddedControlPlane);
        }

        #[cfg(debug_assertions)]
        if fixture_mode == Some(DesktopFixtureMode::SessionViewer) {
            tracing::warn!("SESSION VIEWER FIXTURE MODE ENABLED: fixture DB is reset on startup");

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
            let options = redesmyn_control_plane::ControlPlaneStartOptions::from_config(
                &config.control_plane,
            );
            Some(runtime.block_on(redesmyn_control_plane::ControlPlane::start(options))?)
        } else {
            None
        };

        let session_viewer_fixture = {
            #[cfg(debug_assertions)]
            {
                if fixture_mode == Some(DesktopFixtureMode::SessionViewer) {
                    let session_events = control_plane
                        .as_ref()
                        .expect("fixture mode requires an embedded control plane")
                        .session_events();
                    let session_id = session_viewer_fixture_session_id();

                    let seeded = runtime
                        .block_on(seed_session_viewer_fixture(&session_events, session_id))?;

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
                }
            }

            #[cfg(not(debug_assertions))]
            {
                None
            }
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
                runtime.handle(),
                control_plane
                    .as_ref()
                    .expect("embedded daemon requires embedded control plane")
                    .control_plane(),
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

#[cfg(debug_assertions)]
fn session_viewer_fixture_session_id() -> redesmyn_ids::SessionId {
    redesmyn_ids::SessionId::from_str(SESSION_VIEWER_FIXTURE_SESSION_ID)
        .expect("fixture session id should parse")
}

#[cfg(debug_assertions)]
fn session_viewer_fixture_workspace_id() -> redesmyn_ids::WorkspaceId {
    redesmyn_ids::WorkspaceId::from_str(SESSION_VIEWER_FIXTURE_WORKSPACE_ID)
        .expect("fixture workspace id should parse")
}

#[cfg(debug_assertions)]
fn session_viewer_fixture_repo_id() -> redesmyn_ids::RepoId {
    redesmyn_ids::RepoId::from_str(SESSION_VIEWER_FIXTURE_REPO_ID)
        .expect("fixture repo id should parse")
}

#[cfg(debug_assertions)]
fn session_viewer_fixture_db_path(base_db_path: &Path) -> PathBuf {
    let Some(state_dir) = base_db_path.parent() else {
        return PathBuf::from(".redesmyn/fixtures/session_viewer/redesmyn_rust.sqlite3");
    };

    state_dir
        .join("fixtures")
        .join("session_viewer")
        .join("redesmyn_rust.sqlite3")
}

#[cfg(debug_assertions)]
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

#[cfg(debug_assertions)]
fn seeded_timestamp(index: u64) -> Timestamp {
    let offset_ms = i64::try_from(index)
        .unwrap_or(i64::MAX)
        .saturating_mul(1_000);
    let ms = SESSION_VIEWER_FIXTURE_BASE_MS.saturating_add(offset_ms);
    Timestamp::from_unix_millis(ms).expect("fixture timestamp should be in range")
}

#[cfg(debug_assertions)]
fn seeded_ms(index: u64) -> u64 {
    u64::try_from(SESSION_VIEWER_FIXTURE_BASE_MS)
        .expect("fixture base ms should be non-negative")
        .saturating_add(index.saturating_mul(1_000))
}

#[cfg(debug_assertions)]
fn seeded_session_event_id(created_at_ms: u64, counter: u64) -> redesmyn_ids::SessionEventId {
    let mut bytes = [0_u8; 16];

    let ts_bytes = created_at_ms.to_be_bytes();
    bytes[0..6].copy_from_slice(&ts_bytes[2..]);

    let rand_bytes = u128::from(counter).to_be_bytes();
    bytes[6..].copy_from_slice(&rand_bytes[6..]);

    redesmyn_ids::SessionEventId::from_bytes(bytes)
}

#[cfg(debug_assertions)]
async fn seed_session_viewer_fixture(
    session_events: &SessionEvents,
    session_id: redesmyn_ids::SessionId,
) -> Result<usize, StorageError> {
    ensure_session_viewer_fixture_agent_session(session_events.pool(), session_id).await?;

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
        let turn_id = format!("fixture_turn_{turn:03}");

        let turn_started = SessionEvent {
            session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
            created_at: seeded_timestamp(ix),
            scope: SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id.clone()),
            kind: SessionEventKind::TurnStarted(TurnStarted {
                interface_mode: InterfaceMode::Structured,
                external_session_ref: Some(ExternalSessionRef::CodexThread {
                    thread_id: "fixture_thread".to_string(),
                    turn_id: Some(turn_id.clone()),
                }),
                idempotency_key: Some(format!("fixture_turn_start_{turn:03}")),
                log_offset_bytes: None,
            }),
        };
        session_events.append_session_event(&turn_started).await?;
        seeded += 1;
        ix += 1;

        let user_preview = format!("Fixture user message #{turn}");
        let user = SessionEvent {
            session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
            created_at: seeded_timestamp(ix),
            scope: SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id.clone()),
            kind: SessionEventKind::UserMessage(UserMessage {
                text: format!("{user_preview}\n\n- bullet one\n- bullet two\n\n`inline_code()`"),
                preview: user_preview,
                full_text_artifact: None,
                image_attachments: Vec::new(),
            }),
        };
        session_events.append_session_event(&user).await?;
        seeded += 1;
        ix += 1;

        if turn % 10 == 0 {
            let tool_call_id = format!("fixture_tool_{turn:03}");

            let invocation = SessionEvent {
                session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
                created_at: seeded_timestamp(ix),
                scope: SessionScope::Chat,
                session_id,
                turn_id: Some(turn_id.clone()),
                kind: SessionEventKind::ToolInvocation(ToolInvocation {
                    tool_name: "functions.exec_command".to_string(),
                    tool_call_id: Some(tool_call_id.clone()),
                    input_preview: "echo \"hello from fixture\"".to_string(),
                    input_artifact: None,
                }),
            };
            session_events.append_session_event(&invocation).await?;
            seeded += 1;
            ix += 1;

            let result = SessionEvent {
                session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
                created_at: seeded_timestamp(ix),
                scope: SessionScope::Chat,
                session_id,
                turn_id: Some(turn_id.clone()),
                kind: SessionEventKind::ToolResult(ToolResult {
                    tool_name: "functions.exec_command".to_string(),
                    tool_call_id: Some(tool_call_id),
                    output_preview: "hello from fixture".to_string(),
                    output_artifact: None,
                    error: None,
                }),
            };
            session_events.append_session_event(&result).await?;
            seeded += 1;
            ix += 1;
        }

        let assistant_preview = format!("Fixture assistant reply #{turn}");
        let assistant = SessionEvent {
            session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
            created_at: seeded_timestamp(ix),
            scope: SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id.clone()),
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

        let turn_completed = SessionEvent {
            session_event_id: seeded_session_event_id(seeded_ms(ix), ix),
            created_at: seeded_timestamp(ix),
            scope: SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id),
            kind: SessionEventKind::TurnCompleted(TurnCompleted {
                interface_mode: InterfaceMode::Structured,
                external_session_ref: None,
                exit_code: Some(0),
                error: None,
            }),
        };
        session_events.append_session_event(&turn_completed).await?;
        seeded += 1;
        ix += 1;
    }

    Ok(seeded)
}

#[cfg(debug_assertions)]
async fn ensure_session_viewer_fixture_agent_session(
    pool: &sqlx::SqlitePool,
    session_id: redesmyn_ids::SessionId,
) -> Result<(), StorageError> {
    use redesmyn_storage::schema::{AgentKind, AgentSessionScopeKind, AgentSessionStatus};
    use redesmyn_storage::sessions::{AgentSessionRecord, get_agent_session, insert_agent_session};

    if get_agent_session(pool, session_id).await?.is_some() {
        return Ok(());
    }

    let workspace_id = session_viewer_fixture_workspace_id();
    let repo_id = session_viewer_fixture_repo_id();
    let now_ms = SESSION_VIEWER_FIXTURE_BASE_MS;

    sqlx::query(
        r#"
        INSERT OR IGNORE INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("fixture")
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        INSERT OR IGNORE INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("fixture")
    .bind("Fixture repo")
    .execute(pool)
    .await?;

    let session = AgentSessionRecord {
        session_id,
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
        scope_workspace_id: workspace_id,
        scope_repo_id: repo_id,
        scope_kind: AgentSessionScopeKind::Chat,
        task_id: None,
        epic_id: None,
        agent_kind: AgentKind::Codex,
        status: AgentSessionStatus::Stopped,
        external_session_ref: r#"{"type":"none"}"#.to_owned(),
        title: Some("Fixture session".to_owned()),
        started_at_ms: Some(now_ms),
        ended_at_ms: None,
        archived_at_ms: None,
        repo_name: None,
    };

    insert_agent_session(pool, &session).await?;

    Ok(())
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
