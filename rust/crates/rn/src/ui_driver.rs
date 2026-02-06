use std::{
    env,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use clap::{Args, Subcommand, ValueEnum};
use prost::Message as _;
use redesmyn_ids::{RequestId, TaskId};
use redesmyn_protocol::pb::redesmyn::protocol::v1 as pbv1;
use redesmyn_protocol::ui_driver::{
    CaptureScreenshotRequest, CaptureScreenshotResponse, CreateChatSessionRequest,
    CreateChatSessionResponse, OpenEpicRequest, SelectGraphNodeRequest, SelectTaskRequest,
    SessionSettingsMenuSendKeyRequest, SessionSettingsMenuSetOpenRequest,
    SetSettingsDialogOpenRequest, SetSettingsDialogSectionRequest, SettingsDialogSection,
    TaskFiltersMenuSetOpenRequest, TriggerRefreshRequest, UiDriverFrame, UiDriverMessage,
    UiDriverRequest, UiDriverRequestPayload, UiDriverResponseResult, UiPrimaryView,
    UiScreenshotWindow, UiSnapshotPredicate, WaitForUiIdleRequest, WaitForUiSnapshotRequest,
    WaitForUiSnapshotResponse,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, ProtocolEnvelope};
use serde::Serialize;

use crate::{CommandOutcome, Output, OutputFormat};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

const DEFAULT_MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

#[derive(Debug, Subcommand)]
pub(crate) enum UiDriverCommands {
    /// Runs a minimal end-to-end driver smoke flow:
    /// open epic → create chat → wait for idle → capture artifacts.
    Smoke(UiDriverSmokeArgs),
    /// Opens the settings dialog and captures artifacts.
    ///
    /// Prefer `--launch` (sets `REDESMYN_UI_TEST_MODE=1`) unless you are connecting to an
    /// already-running test instance.
    Settings(UiDriverSettingsArgs),
    /// Runs a deterministic graph smoke flow:
    /// open epic → select node → capture artifacts.
    ///
    /// Prefer `--launch` (sets `REDESMYN_UI_TEST_MODE=1`) unless you are connecting to an
    /// already-running test instance.
    GraphSmoke(UiDriverGraphSmokeArgs),
    /// Opens the session Settings menu and exercises basic keyboard navigation.
    SettingsMenuSmoke(UiDriverSettingsMenuSmokeArgs),
    /// Opens the workspace task filters menu and captures artifacts.
    TaskFiltersMenuSmoke(UiDriverTaskFiltersMenuSmokeArgs),
}

#[derive(Debug, Args)]
pub(crate) struct UiDriverSmokeArgs {
    /// Unix domain socket path for the desktop UI driver.
    ///
    /// If omitted, uses `REDESMYN_UI_DRIVER_SOCKET_PATH` when present.
    #[arg(long)]
    uds: Option<PathBuf>,

    /// Epic slug to select (required).
    #[arg(long)]
    epic: String,

    /// Artifact label for `CaptureScreenshot`.
    #[arg(long, default_value = "smoke")]
    label: String,

    /// Optional chat title hint (defaults to empty).
    #[arg(long)]
    chat_title: Option<String>,

    /// Timeout for wait operations (milliseconds).
    #[arg(long, default_value_t = 20_000)]
    timeout_ms: u64,

    /// Quiescence window for `WaitForIdle` (milliseconds).
    #[arg(long, default_value_t = 50)]
    quiescence_ms: u64,

    /// Launch the desktop app automatically (builds `redesmyn_desktop` if needed).
    #[arg(long, default_value_t = false)]
    launch: bool,

    /// Override `rust.control_plane.db.path` in `--launch` mode.
    ///
    /// Equivalent to setting `REDESMYN_RUST__CONTROL_PLANE__DB__PATH`.
    #[arg(long, alias = "dev-db")]
    rust_db_path: Option<PathBuf>,

    /// Artifacts directory for `--launch` mode (defaults to a temp dir).
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,

    /// Timeout waiting for the UI driver socket to appear (milliseconds, `--launch` mode).
    #[arg(long, default_value_t = 10_000)]
    startup_timeout_ms: u64,

    /// Keep the desktop app running after the smoke flow completes (`--launch` mode).
    #[arg(long, default_value_t = false)]
    keep_open: bool,
}

#[derive(Debug, Args)]
pub(crate) struct UiDriverGraphSmokeArgs {
    /// Unix domain socket path for the desktop UI driver.
    ///
    /// If omitted, uses `REDESMYN_UI_DRIVER_SOCKET_PATH` when present.
    #[arg(long)]
    uds: Option<PathBuf>,

    /// Epic slug to select (required).
    #[arg(long)]
    epic: String,

    /// TaskId to select in the graph.
    ///
    /// If omitted, uses a deterministic demo TaskId (bytes: [1; 16]) from UI test mode.
    /// Prefer `--launch`; otherwise pass `--task-id`.
    #[arg(long)]
    task_id: Option<TaskId>,

    /// Task slug to select in the graph (e.g. `T-58`).
    ///
    /// Mutually exclusive with `--task-id`. Useful when graph node ids are protocol-abstracted.
    #[arg(long)]
    task_slug: Option<String>,

    /// Artifact label for `CaptureScreenshot`.
    #[arg(long, default_value = "graph_smoke")]
    label: String,

    /// Timeout for wait operations (milliseconds).
    #[arg(long, default_value_t = 20_000)]
    timeout_ms: u64,

    /// Quiescence window for `WaitForIdle` (milliseconds).
    #[arg(long, default_value_t = 50)]
    quiescence_ms: u64,

    /// Launch the desktop app automatically (builds `redesmyn_desktop` if needed).
    #[arg(long, default_value_t = false)]
    launch: bool,

    /// Override `rust.control_plane.db.path` in `--launch` mode.
    ///
    /// Equivalent to setting `REDESMYN_RUST__CONTROL_PLANE__DB__PATH`.
    #[arg(long, alias = "dev-db")]
    rust_db_path: Option<PathBuf>,

    /// Artifacts directory for `--launch` mode (defaults to a temp dir).
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,

    /// Timeout waiting for the UI driver socket to appear (milliseconds, `--launch` mode).
    #[arg(long, default_value_t = 10_000)]
    startup_timeout_ms: u64,

    /// Keep the desktop app running after the smoke flow completes (`--launch` mode).
    #[arg(long, default_value_t = false)]
    keep_open: bool,
}

#[derive(Debug, Args)]
pub(crate) struct UiDriverSettingsArgs {
    /// Unix domain socket path for the desktop UI driver.
    ///
    /// If omitted, uses `REDESMYN_UI_DRIVER_SOCKET_PATH` when present.
    #[arg(long)]
    uds: Option<PathBuf>,

    /// Artifact label for `CaptureScreenshot`.
    #[arg(long, default_value = "settings")]
    label: String,

    /// Settings section to show before capturing artifacts.
    #[arg(long, value_enum)]
    section: Option<UiDriverSettingsSection>,
    /// Timeout for wait operations (milliseconds).
    #[arg(long, default_value_t = 20_000)]
    timeout_ms: u64,

    /// Quiescence window for `WaitForIdle` (milliseconds).
    #[arg(long, default_value_t = 50)]
    quiescence_ms: u64,

    /// Launch the desktop app automatically (builds `redesmyn_desktop` if needed).
    #[arg(long, default_value_t = false)]
    launch: bool,

    /// Override `rust.control_plane.db.path` in `--launch` mode.
    ///
    /// Equivalent to setting `REDESMYN_RUST__CONTROL_PLANE__DB__PATH`.
    #[arg(long, alias = "dev-db")]
    rust_db_path: Option<PathBuf>,

    /// Artifacts directory for `--launch` mode (defaults to a temp dir).
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,

    /// Timeout waiting for the UI driver socket to appear (milliseconds, `--launch` mode).
    #[arg(long, default_value_t = 10_000)]
    startup_timeout_ms: u64,

    /// Keep the desktop app running after the flow completes (`--launch` mode).
    #[arg(long, default_value_t = false)]
    keep_open: bool,
}

#[derive(Debug, Args)]
pub(crate) struct UiDriverSettingsMenuSmokeArgs {
    /// Unix domain socket path for the desktop UI driver.
    ///
    /// If omitted, uses `REDESMYN_UI_DRIVER_SOCKET_PATH` when present.
    #[arg(long)]
    uds: Option<PathBuf>,

    /// Epic slug to select (required).
    #[arg(long)]
    epic: String,

    /// Artifact label for `CaptureScreenshot`.
    #[arg(long, default_value = "settings_menu_smoke")]
    label: String,

    /// Optional chat title hint (defaults to empty).
    #[arg(long)]
    chat_title: Option<String>,

    /// Timeout for wait operations (milliseconds).
    #[arg(long, default_value_t = 20_000)]
    timeout_ms: u64,

    /// Quiescence window for `WaitForIdle` (milliseconds).
    #[arg(long, default_value_t = 50)]
    quiescence_ms: u64,

    /// Launch the desktop app automatically (builds `redesmyn_desktop` if needed).
    #[arg(long, default_value_t = false)]
    launch: bool,

    /// Artifacts directory for `--launch` mode (defaults to a temp dir).
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,

    /// Timeout waiting for the UI driver socket to appear (milliseconds, `--launch` mode).
    #[arg(long, default_value_t = 10_000)]
    startup_timeout_ms: u64,

    /// Keep the desktop app running after the smoke flow completes (`--launch` mode).
    #[arg(long, default_value_t = false)]
    keep_open: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
enum UiDriverSettingsSection {
    Appearance,
    Agents,
}

impl UiDriverSettingsSection {
    const fn to_protocol(self) -> SettingsDialogSection {
        match self {
            Self::Appearance => SettingsDialogSection::Appearance,
            Self::Agents => SettingsDialogSection::Agents,
        }
    }
}

#[derive(Debug, Args)]
pub(crate) struct UiDriverTaskFiltersMenuSmokeArgs {
    /// Unix domain socket path for the desktop UI driver.
    ///
    /// If omitted, uses `REDESMYN_UI_DRIVER_SOCKET_PATH` when present.
    #[arg(long)]
    uds: Option<PathBuf>,

    /// Epic slug to select (required).
    #[arg(long)]
    epic: String,

    /// Artifact label for `CaptureScreenshot`.
    #[arg(long, default_value = "task_filters_menu_smoke")]
    label: String,

    /// Timeout for wait operations (milliseconds).
    #[arg(long, default_value_t = 20_000)]
    timeout_ms: u64,

    /// Quiescence window for `WaitForIdle` (milliseconds).
    #[arg(long, default_value_t = 50)]
    quiescence_ms: u64,

    /// Launch the desktop app automatically (builds `redesmyn_desktop` if needed).
    #[arg(long, default_value_t = false)]
    launch: bool,

    /// Artifacts directory for `--launch` mode (defaults to a temp dir).
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,

    /// Timeout waiting for the UI driver socket to appear (milliseconds, `--launch` mode).
    #[arg(long, default_value_t = 10_000)]
    startup_timeout_ms: u64,

    /// Keep the desktop app running after the smoke flow completes (`--launch` mode).
    #[arg(long, default_value_t = false)]
    keep_open: bool,
}

#[derive(Debug, Serialize)]
struct SmokeReport {
    epic_slug: String,
    socket_path: String,
    chat_session_id: String,
    screenshot_path: Option<String>,
    ui_snapshot_path: Option<String>,
    artifacts_dir: Option<String>,
}

#[derive(Debug, Serialize)]
struct GraphSmokeReport {
    epic_slug: String,
    task_id: String,
    socket_path: String,
    screenshot_path: Option<String>,
    ui_snapshot_path: Option<String>,
    artifacts_dir: Option<String>,
}

#[derive(Debug, Serialize)]
struct SettingsReport {
    socket_path: String,
    screenshot_path: Option<String>,
    ui_snapshot_path: Option<String>,
    artifacts_dir: Option<String>,
}

#[derive(Debug, Serialize)]
struct SettingsMenuSmokeReport {
    epic_slug: String,
    socket_path: String,
    chat_session_id: String,
    screenshot_path: Option<String>,
    ui_snapshot_path: Option<String>,
    artifacts_dir: Option<String>,
}

#[derive(Debug, Serialize)]
struct TaskFiltersMenuSmokeReport {
    epic_slug: String,
    socket_path: String,
    screenshot_path: Option<String>,
    ui_snapshot_path: Option<String>,
    artifacts_dir: Option<String>,
}

pub(crate) fn ui_driver(cmd: UiDriverCommands, output: &Output) -> CommandOutcome {
    match cmd {
        UiDriverCommands::Smoke(args) => ui_driver_smoke(args, output),
        UiDriverCommands::Settings(args) => ui_driver_settings(args, output),
        UiDriverCommands::GraphSmoke(args) => ui_driver_graph_smoke(args, output),
        UiDriverCommands::SettingsMenuSmoke(args) => ui_driver_settings_menu_smoke(args, output),
        UiDriverCommands::TaskFiltersMenuSmoke(args) => {
            ui_driver_task_filters_menu_smoke(args, output)
        }
    }
}

// --- Smoke harness helpers ----------------------------------------------------

#[cfg(unix)]
struct UiDriverSmokeHarness {
    socket_path: PathBuf,
    artifacts_dir: Option<PathBuf>,
    keep_open: bool,
    desktop_child: Option<Child>,
    conn: UnixStream,
}

#[cfg(unix)]
impl UiDriverSmokeHarness {
    fn start(
        uds: Option<PathBuf>,
        launch: bool,
        artifacts_dir: Option<PathBuf>,
        startup_timeout_ms: u64,
        keep_open: bool,
        rust_db_path: Option<&Path>,
    ) -> Result<Self, ErrorEnvelope> {
        let socket_path = resolve_smoke_socket_path(uds, launch)?;
        let artifacts_dir = if launch {
            Some(artifacts_dir.unwrap_or_else(default_smoke_artifacts_dir))
        } else {
            None
        };

        if rust_db_path.is_some() && !launch {
            return Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "--rust-db-path requires --launch.",
            ));
        }

        let mut desktop_child: Option<Child> = None;
        if launch {
            let workspace_root = find_rust_workspace_root()?;
            build_desktop_app(&workspace_root)?;

            let Some(artifacts_dir) = artifacts_dir.as_ref() else {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "Expected artifacts_dir to be set in --launch mode.",
                ));
            };

            std::fs::create_dir_all(artifacts_dir).map_err(|err| {
                ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!(
                        "Failed to create artifacts directory {}: {err}",
                        artifacts_dir.display()
                    ),
                )
            })?;

            desktop_child = Some(spawn_desktop_app(
                &workspace_root,
                &socket_path,
                artifacts_dir,
                rust_db_path,
            )?);

            wait_for_path(&socket_path, Duration::from_millis(startup_timeout_ms)).map_err(
                |err| {
                    shutdown_desktop(&mut desktop_child, keep_open);
                    err
                },
            )?;
        }

        let conn = UnixStream::connect(&socket_path).map_err(|err| {
            shutdown_desktop(&mut desktop_child, keep_open);
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                format!(
                    "Failed to connect to UI driver socket {}: {err}",
                    socket_path.display()
                ),
            )
        })?;

        Ok(Self {
            socket_path,
            artifacts_dir,
            keep_open,
            desktop_child,
            conn,
        })
    }

    fn send(
        &mut self,
        payload: UiDriverRequestPayload,
        read_timeout: Duration,
    ) -> Result<UiDriverResponseResult, ErrorEnvelope> {
        send_ui_driver_request(&mut self.conn, payload, read_timeout)
    }

    fn require_ok(
        &mut self,
        payload: UiDriverRequestPayload,
        op: &'static str,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        require_ok_response(self.send(payload, read_timeout), op)
    }

    fn open_epic(
        &mut self,
        epic_slug: String,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        self.require_ok(
            UiDriverRequestPayload::OpenEpic(OpenEpicRequest { epic_slug }),
            "open_epic",
            read_timeout,
        )
    }

    fn select_task(
        &mut self,
        task_slug: String,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        self.require_ok(
            UiDriverRequestPayload::SelectTask(SelectTaskRequest { task_slug }),
            "select_task",
            read_timeout,
        )
    }

    fn create_chat_session(
        &mut self,
        chat_title: String,
        read_timeout: Duration,
    ) -> Result<String, ErrorEnvelope> {
        match self.send(
            UiDriverRequestPayload::CreateChatSession(CreateChatSessionRequest {
                name_hint: chat_title,
            }),
            read_timeout,
        )? {
            UiDriverResponseResult::CreateChatSession(CreateChatSessionResponse { session_id }) => {
                Ok(session_id.to_string())
            }
            UiDriverResponseResult::Error(err) => Err(err),
            other => Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!("Unexpected create_chat_session response: {other:?}"),
            )),
        }
    }

    fn wait_for_idle(
        &mut self,
        timeout_ms: u64,
        quiescence_ms: u64,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        self.require_ok(
            UiDriverRequestPayload::WaitForIdle(WaitForUiIdleRequest {
                timeout_ms,
                quiescence_ms,
            }),
            "wait_for_idle",
            read_timeout,
        )
    }

    fn wait_for_snapshot(
        &mut self,
        timeout_ms: u64,
        predicate: UiSnapshotPredicate,
        read_timeout: Duration,
    ) -> Result<WaitForUiSnapshotResponse, ErrorEnvelope> {
        match self.send(
            UiDriverRequestPayload::WaitForSnapshot(WaitForUiSnapshotRequest {
                timeout_ms,
                predicate,
            }),
            read_timeout,
        )? {
            UiDriverResponseResult::WaitForSnapshot(resp) => Ok(resp),
            UiDriverResponseResult::Error(err) => Err(err),
            other => Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!("Unexpected wait_for_snapshot response: {other:?}"),
            )),
        }
    }

    fn capture_screenshot(
        &mut self,
        label: String,
        read_timeout: Duration,
    ) -> Result<CaptureScreenshotResponse, ErrorEnvelope> {
        match self.send(
            UiDriverRequestPayload::CaptureScreenshot(CaptureScreenshotRequest {
                name_hint: label,
                window: Some(UiScreenshotWindow::Primary),
                include_decorations: Some(false),
            }),
            read_timeout,
        )? {
            UiDriverResponseResult::CaptureScreenshot(resp) => Ok(resp),
            UiDriverResponseResult::Error(err) => Err(err),
            other => Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!("Unexpected capture_screenshot response: {other:?}"),
            )),
        }
    }

    fn set_task_filters_menu_open(
        &mut self,
        open: bool,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        self.require_ok(
            UiDriverRequestPayload::TaskFiltersMenuSetOpen(TaskFiltersMenuSetOpenRequest { open }),
            "task_filters_menu_set_open",
            read_timeout,
        )
    }

    fn set_session_settings_menu_open(
        &mut self,
        open: bool,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        self.require_ok(
            UiDriverRequestPayload::SessionSettingsMenuSetOpen(SessionSettingsMenuSetOpenRequest {
                open,
            }),
            "settings_menu_set_open",
            read_timeout,
        )
    }

    fn send_session_settings_menu_key(
        &mut self,
        key: String,
        read_timeout: Duration,
    ) -> Result<(), ErrorEnvelope> {
        self.require_ok(
            UiDriverRequestPayload::SessionSettingsMenuSendKey(SessionSettingsMenuSendKeyRequest {
                key,
            }),
            "settings_menu_send_key",
            read_timeout,
        )
    }
}

#[cfg(unix)]
impl Drop for UiDriverSmokeHarness {
    fn drop(&mut self) {
        shutdown_desktop(&mut self.desktop_child, self.keep_open);
    }
}

#[cfg(unix)]
fn resolve_smoke_socket_path(uds: Option<PathBuf>, launch: bool) -> Result<PathBuf, ErrorEnvelope> {
    match uds {
        Some(path) => Ok(path),
        None if launch => Ok(default_smoke_socket_path()),
        None => match env::var_os("REDESMYN_UI_DRIVER_SOCKET_PATH") {
            Some(path) => Ok(PathBuf::from(path)),
            None => Err(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Missing --uds (or set REDESMYN_UI_DRIVER_SOCKET_PATH).",
            )),
        },
    }
}

fn ui_driver_settings(args: UiDriverSettingsArgs, output: &Output) -> CommandOutcome {
    #[cfg(not(unix))]
    {
        let _ = output;
        let _ = args;
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "UI driver settings is only supported on unix platforms.",
        ));
    }

    #[cfg(unix)]
    {
        let mut desktop_child: Option<Child> = None;

        let socket_path = match args.uds.clone() {
            Some(path) => path,
            None if args.launch => default_smoke_socket_path(),
            None => match env::var_os("REDESMYN_UI_DRIVER_SOCKET_PATH") {
                Some(path) => PathBuf::from(path),
                None => {
                    return CommandOutcome::Failure(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        "Missing --uds (or set REDESMYN_UI_DRIVER_SOCKET_PATH).",
                    ));
                }
            },
        };

        let artifacts_dir = if args.launch {
            Some(
                args.artifacts_dir
                    .clone()
                    .unwrap_or_else(default_smoke_artifacts_dir),
            )
        } else {
            None
        };

        if args.rust_db_path.is_some() && !args.launch {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "--rust-db-path requires --launch.",
            ));
        }

        if args.launch {
            let workspace_root = match find_rust_workspace_root() {
                Ok(root) => root,
                Err(err) => return CommandOutcome::Failure(err),
            };

            if let Err(err) = build_desktop_app(&workspace_root) {
                return CommandOutcome::Failure(err);
            }

            let Some(artifacts_dir) = artifacts_dir.as_ref() else {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "Expected artifacts_dir to be set in --launch mode.",
                ));
            };

            if let Err(err) = std::fs::create_dir_all(artifacts_dir) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!(
                        "Failed to create artifacts directory {}: {err}",
                        artifacts_dir.display()
                    ),
                ));
            }

            match spawn_desktop_app(
                &workspace_root,
                &socket_path,
                artifacts_dir,
                args.rust_db_path.as_deref(),
            ) {
                Ok(child) => desktop_child = Some(child),
                Err(err) => return CommandOutcome::Failure(err),
            }

            if let Err(err) =
                wait_for_path(&socket_path, Duration::from_millis(args.startup_timeout_ms))
            {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
        }

        let mut conn = match UnixStream::connect(&socket_path) {
            Ok(conn) => conn,
            Err(err) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!(
                        "Failed to connect to ui driver socket {}: {err}",
                        socket_path.display()
                    ),
                ));
            }
        };

        let timeout = Duration::from_millis(args.timeout_ms);
        let wait_timeout = timeout + Duration::from_secs(2);

        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::SetSettingsDialogOpen(SetSettingsDialogOpenRequest {
                    open: true,
                }),
                Duration::from_secs(2),
            ),
            "set_settings_dialog_open",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        if let Some(section) = args.section {
            if let Err(err) = require_ok_response(
                send_ui_driver_request(
                    &mut conn,
                    UiDriverRequestPayload::SetSettingsDialogSection(
                        SetSettingsDialogSectionRequest {
                            section: section.to_protocol(),
                        },
                    ),
                    Duration::from_secs(2),
                ),
                "set_settings_dialog_section",
            ) {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
        }

        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::WaitForIdle(WaitForUiIdleRequest {
                    timeout_ms: args.timeout_ms,
                    quiescence_ms: args.quiescence_ms,
                }),
                wait_timeout,
            ),
            "wait_for_idle",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        let screenshot = match send_ui_driver_request(
            &mut conn,
            UiDriverRequestPayload::CaptureScreenshot(CaptureScreenshotRequest {
                name_hint: args.label.clone(),
                window: Some(UiScreenshotWindow::Primary),
                include_decorations: Some(false),
            }),
            wait_timeout,
        ) {
            Ok(UiDriverResponseResult::CaptureScreenshot(resp)) => resp,
            Ok(UiDriverResponseResult::Error(err)) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
            Ok(other) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    format!("Unexpected capture_screenshot response: {other:?}"),
                ));
            }
            Err(err) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
        };

        let (screenshot_path, ui_snapshot_path) =
            infer_artifact_paths(&args.label, &screenshot).unwrap_or((None, None));

        shutdown_desktop(&mut desktop_child, args.keep_open);

        let report = SettingsReport {
            socket_path: socket_path.to_string_lossy().to_string(),
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: artifacts_dir.map(|dir| dir.to_string_lossy().to_string()),
        };

        match output.format {
            OutputFormat::Human => {
                println!("ok: true");
                println!("socket_path: {}", report.socket_path);
                if let Some(path) = report.screenshot_path.as_deref() {
                    println!("screenshot_path: {path}");
                }
                if let Some(path) = report.ui_snapshot_path.as_deref() {
                    println!("ui_snapshot_path: {path}");
                }
                if let Some(dir) = report.artifacts_dir.as_deref() {
                    println!("artifacts_dir: {dir}");
                }
                CommandOutcome::Success
            }
            OutputFormat::Json => match output.print_json_stdout(&report) {
                Ok(()) => CommandOutcome::Success,
                Err(err) => CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                )),
            },
        }
    }
}

fn ui_driver_smoke(args: UiDriverSmokeArgs, output: &Output) -> CommandOutcome {
    #[cfg(not(unix))]
    {
        let _ = output;
        let _ = args;
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "UI driver smoke is only supported on unix platforms.",
        ));
    }

    #[cfg(unix)]
    {
        let mut harness = match UiDriverSmokeHarness::start(
            args.uds.clone(),
            args.launch,
            args.artifacts_dir.clone(),
            args.startup_timeout_ms,
            args.keep_open,
            args.rust_db_path.as_deref(),
        ) {
            Ok(harness) => harness,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let timeout = Duration::from_millis(args.timeout_ms);
        let wait_timeout = timeout + Duration::from_secs(2);

        if let Err(err) = harness.open_epic(args.epic.clone(), Duration::from_secs(2)) {
            return CommandOutcome::Failure(err);
        }

        let wait_predicate = UiSnapshotPredicate {
            primary_view: Some(UiPrimaryView::EpicWorkspace),
            epic_slug: args.epic.clone(),
            in_flight_empty: None,
            selected_task_id: None,
            graph_layout_settled: None,
            graph_selection_settled: None,
        };
        if let Err(err) = harness.wait_for_snapshot(args.timeout_ms, wait_predicate, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        let chat_title = args.chat_title.clone().unwrap_or_default();
        let chat_session_id = match harness.create_chat_session(chat_title, wait_timeout) {
            Ok(session_id) => session_id,
            Err(err) => return CommandOutcome::Failure(err),
        };

        if let Err(err) = harness.wait_for_idle(args.timeout_ms, args.quiescence_ms, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        let screenshot = match harness.capture_screenshot(args.label.clone(), wait_timeout) {
            Ok(screenshot) => screenshot,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let (screenshot_path, ui_snapshot_path) =
            infer_artifact_paths(&args.label, &screenshot).unwrap_or((None, None));

        let report = SmokeReport {
            epic_slug: args.epic,
            socket_path: harness.socket_path.to_string_lossy().to_string(),
            chat_session_id,
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: harness
                .artifacts_dir
                .as_ref()
                .map(|dir| dir.to_string_lossy().to_string()),
        };

        match output.format {
            OutputFormat::Human => {
                println!("ok: true");
                println!("epic_slug: {}", report.epic_slug);
                println!("socket_path: {}", report.socket_path);
                println!("chat_session_id: {}", report.chat_session_id);
                if let Some(path) = report.screenshot_path.as_deref() {
                    println!("screenshot_path: {path}");
                }
                if let Some(path) = report.ui_snapshot_path.as_deref() {
                    println!("ui_snapshot_path: {path}");
                }
                if let Some(dir) = report.artifacts_dir.as_deref() {
                    println!("artifacts_dir: {dir}");
                }
                CommandOutcome::Success
            }
            OutputFormat::Json => match output.print_json_stdout(&report) {
                Ok(()) => CommandOutcome::Success,
                Err(err) => CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                )),
            },
        }
    }
}

fn ui_driver_settings_menu_smoke(
    args: UiDriverSettingsMenuSmokeArgs,
    output: &Output,
) -> CommandOutcome {
    #[cfg(not(unix))]
    {
        let _ = output;
        let _ = args;
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "UI driver settings menu smoke is only supported on unix platforms.",
        ));
    }

    #[cfg(unix)]
    {
        let mut harness = match UiDriverSmokeHarness::start(
            args.uds.clone(),
            args.launch,
            args.artifacts_dir.clone(),
            args.startup_timeout_ms,
            args.keep_open,
            None,
        ) {
            Ok(harness) => harness,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let wait_timeout = Duration::from_millis(args.timeout_ms);

        if let Err(err) = harness.open_epic(args.epic.clone(), wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        let chat_title = args.chat_title.clone().unwrap_or_default();
        let chat_session_id = match harness.create_chat_session(chat_title, wait_timeout) {
            Ok(session_id) => session_id,
            Err(err) => return CommandOutcome::Failure(err),
        };

        if let Err(err) = harness.wait_for_idle(args.timeout_ms, args.quiescence_ms, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = harness.set_task_filters_menu_open(true, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = harness.set_session_settings_menu_open(true, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        for key in ["down", "right", "down"] {
            if let Err(err) = harness.send_session_settings_menu_key(key.to_string(), wait_timeout)
            {
                return CommandOutcome::Failure(err);
            }
        }

        let screenshot = match harness.capture_screenshot(args.label.clone(), wait_timeout) {
            Ok(screenshot) => screenshot,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let (screenshot_path, ui_snapshot_path) =
            infer_artifact_paths(&args.label, &screenshot).unwrap_or((None, None));

        let report = SettingsMenuSmokeReport {
            epic_slug: args.epic,
            socket_path: harness.socket_path.to_string_lossy().to_string(),
            chat_session_id,
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: harness
                .artifacts_dir
                .as_ref()
                .map(|dir| dir.to_string_lossy().to_string()),
        };

        match output.format {
            OutputFormat::Human => {
                println!("ok: true");
                println!("epic_slug: {}", report.epic_slug);
                println!("socket_path: {}", report.socket_path);
                println!("chat_session_id: {}", report.chat_session_id);
                if let Some(path) = report.screenshot_path.as_deref() {
                    println!("screenshot_path: {path}");
                }
                if let Some(path) = report.ui_snapshot_path.as_deref() {
                    println!("ui_snapshot_path: {path}");
                }
                if let Some(dir) = report.artifacts_dir.as_deref() {
                    println!("artifacts_dir: {dir}");
                }
                CommandOutcome::Success
            }
            OutputFormat::Json => match output.print_json_stdout(&report) {
                Ok(()) => CommandOutcome::Success,
                Err(err) => CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                )),
            },
        }
    }
}

fn ui_driver_task_filters_menu_smoke(
    args: UiDriverTaskFiltersMenuSmokeArgs,
    output: &Output,
) -> CommandOutcome {
    #[cfg(not(unix))]
    {
        let _ = output;
        let _ = args;
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "UI driver task filters menu smoke is only supported on unix platforms.",
        ));
    }

    #[cfg(unix)]
    {
        let mut harness = match UiDriverSmokeHarness::start(
            args.uds.clone(),
            args.launch,
            args.artifacts_dir.clone(),
            args.startup_timeout_ms,
            args.keep_open,
            None,
        ) {
            Ok(harness) => harness,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let wait_timeout = Duration::from_millis(args.timeout_ms);

        if let Err(err) = harness.open_epic(args.epic.clone(), wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = harness.wait_for_idle(args.timeout_ms, args.quiescence_ms, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = harness.set_task_filters_menu_open(true, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        let screenshot = match harness.capture_screenshot(args.label.clone(), wait_timeout) {
            Ok(screenshot) => screenshot,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let (screenshot_path, ui_snapshot_path) =
            infer_artifact_paths(&args.label, &screenshot).unwrap_or((None, None));

        let report = TaskFiltersMenuSmokeReport {
            epic_slug: args.epic,
            socket_path: harness.socket_path.to_string_lossy().to_string(),
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: harness
                .artifacts_dir
                .as_ref()
                .map(|dir| dir.to_string_lossy().to_string()),
        };

        match output.format {
            OutputFormat::Human => {
                println!("ok: true");
                println!("epic_slug: {}", report.epic_slug);
                println!("socket_path: {}", report.socket_path);
                if let Some(path) = report.screenshot_path.as_deref() {
                    println!("screenshot_path: {path}");
                }
                if let Some(path) = report.ui_snapshot_path.as_deref() {
                    println!("ui_snapshot_path: {path}");
                }
                if let Some(dir) = report.artifacts_dir.as_deref() {
                    println!("artifacts_dir: {dir}");
                }
                CommandOutcome::Success
            }
            OutputFormat::Json => match output.print_json_stdout(&report) {
                Ok(()) => CommandOutcome::Success,
                Err(err) => CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                )),
            },
        }
    }
}

fn ui_driver_graph_smoke(args: UiDriverGraphSmokeArgs, output: &Output) -> CommandOutcome {
    #[cfg(not(unix))]
    {
        let _ = output;
        let _ = args;
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "UI driver graph smoke is only supported on unix platforms.",
        ));
    }

    #[cfg(unix)]
    {
        if args.task_id.is_some() && args.task_slug.is_some() {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Pass either --task-id or --task-slug, not both.",
            ));
        }

        if args.task_id.is_none() && args.task_slug.is_none() && !args.launch {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Missing --task-id/--task-slug. The default demo TaskId ([1; 16]) requires `--launch` (sets REDESMYN_UI_TEST_MODE=1).",
            ));
        }

        let mut harness = match UiDriverSmokeHarness::start(
            args.uds.clone(),
            args.launch,
            args.artifacts_dir.clone(),
            args.startup_timeout_ms,
            args.keep_open,
            args.rust_db_path.as_deref(),
        ) {
            Ok(harness) => harness,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let task_id = args
            .task_id
            .unwrap_or_else(|| TaskId::from_bytes([1_u8; 16]));
        let task_slug = args.task_slug;

        let timeout = Duration::from_millis(args.timeout_ms);
        let wait_timeout = timeout + Duration::from_secs(2);

        if let Err(err) = harness.open_epic(args.epic.clone(), Duration::from_secs(2)) {
            return CommandOutcome::Failure(err);
        }

        let wait_workspace_predicate = UiSnapshotPredicate {
            primary_view: Some(UiPrimaryView::EpicWorkspace),
            epic_slug: args.epic.clone(),
            in_flight_empty: None,
            selected_task_id: None,
            graph_layout_settled: Some(true),
            graph_selection_settled: None,
        };
        if let Err(err) =
            harness.wait_for_snapshot(args.timeout_ms, wait_workspace_predicate, wait_timeout)
        {
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = harness.require_ok(
            UiDriverRequestPayload::TriggerRefresh(TriggerRefreshRequest {
                epic_slug: args.epic.clone(),
                name_hint: "graph_smoke".to_string(),
            }),
            "trigger_refresh",
            wait_timeout,
        ) {
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = harness.wait_for_idle(args.timeout_ms, args.quiescence_ms, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        if let Some(slug) = task_slug.clone() {
            if let Err(err) = harness.select_task(slug, wait_timeout) {
                return CommandOutcome::Failure(err);
            }
        } else if let Err(err) = harness.require_ok(
            UiDriverRequestPayload::GraphSelectNode(SelectGraphNodeRequest { task_id }),
            "graph_select_node",
            wait_timeout,
        ) {
            return CommandOutcome::Failure(err);
        }

        let wait_selection_predicate = UiSnapshotPredicate {
            primary_view: None,
            epic_slug: String::new(),
            in_flight_empty: None,
            selected_task_id: task_slug.as_ref().map(|_| None).unwrap_or(Some(task_id)),
            graph_layout_settled: None,
            graph_selection_settled: Some(true),
        };
        let snapshot = match harness.wait_for_snapshot(
            args.timeout_ms,
            wait_selection_predicate,
            wait_timeout,
        ) {
            Ok(resp) => resp.snapshot,
            Err(err) => return CommandOutcome::Failure(err),
        };

        if !snapshot.graph.expanded_task_card_open {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Expected expanded task card to be open after selecting node.",
            ));
        }

        if let Some(expected_slug) = task_slug {
            if snapshot.selection.task_slug != expected_slug {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!(
                        "Expected selected task slug {expected_slug:?}; got {:?}",
                        snapshot.selection.task_slug
                    ),
                ));
            }
        } else if snapshot.graph.expanded_task_id != Some(task_id) {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                format!(
                    "Expected expanded task card to be open for {task_id}; got task_id={:?}",
                    snapshot.graph.expanded_task_id
                ),
            ));
        }

        if let Err(err) = harness.wait_for_idle(args.timeout_ms, args.quiescence_ms, wait_timeout) {
            return CommandOutcome::Failure(err);
        }

        let screenshot = match harness.capture_screenshot(args.label.clone(), wait_timeout) {
            Ok(screenshot) => screenshot,
            Err(err) => return CommandOutcome::Failure(err),
        };

        let (screenshot_path, ui_snapshot_path) =
            infer_artifact_paths(&args.label, &screenshot).unwrap_or((None, None));

        let report = GraphSmokeReport {
            epic_slug: args.epic,
            task_id: task_id.to_string(),
            socket_path: harness.socket_path.to_string_lossy().to_string(),
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: harness
                .artifacts_dir
                .as_ref()
                .map(|dir| dir.to_string_lossy().to_string()),
        };

        match output.format {
            OutputFormat::Human => {
                println!("ok: true");
                println!("epic_slug: {}", report.epic_slug);
                println!("task_id: {}", report.task_id);
                println!("socket_path: {}", report.socket_path);
                if let Some(path) = report.screenshot_path.as_deref() {
                    println!("screenshot_path: {path}");
                }
                if let Some(path) = report.ui_snapshot_path.as_deref() {
                    println!("ui_snapshot_path: {path}");
                }
                if let Some(dir) = report.artifacts_dir.as_deref() {
                    println!("artifacts_dir: {dir}");
                }
                CommandOutcome::Success
            }
            OutputFormat::Json => match output.print_json_stdout(&report) {
                Ok(()) => CommandOutcome::Success,
                Err(err) => CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                )),
            },
        }
    }
}

#[cfg(unix)]
fn default_smoke_socket_path() -> PathBuf {
    let pid = std::process::id();
    let mut path = std::env::temp_dir();
    path.push(format!("redesmyn-ui-driver-smoke-{pid}.sock"));
    path
}

#[cfg(unix)]
fn default_smoke_artifacts_dir() -> PathBuf {
    let pid = std::process::id();
    let mut path = std::env::temp_dir();
    path.push(format!("redesmyn-artifacts-{pid}"));
    path
}

#[cfg(unix)]
fn find_rust_workspace_root() -> Result<PathBuf, ErrorEnvelope> {
    let mut dir = env::current_dir().map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("Failed to read current directory: {err}"),
        )
    })?;

    loop {
        if dir.join("Cargo.toml").exists() {
            return Ok(dir);
        }
        if !dir.pop() {
            break;
        }
    }

    Err(ErrorEnvelope::new(
        ErrorCategory::InvalidRequest,
        "Could not find rust workspace root (Cargo.toml) from current directory.",
    ))
}

#[cfg(unix)]
fn build_desktop_app(workspace_root: &Path) -> Result<(), ErrorEnvelope> {
    let status = Command::new("cargo")
        .current_dir(workspace_root)
        .args(["build", "-p", "redesmyn_desktop"])
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                format!("Failed to run cargo build: {err}"),
            )
        })?;

    if status.success() {
        return Ok(());
    }

    Err(ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        format!("cargo build failed with status {status}"),
    ))
}

#[cfg(unix)]
fn spawn_desktop_app(
    workspace_root: &Path,
    socket_path: &Path,
    artifacts_dir: &Path,
    rust_db_path: Option<&Path>,
) -> Result<Child, ErrorEnvelope> {
    let bin_path = workspace_root
        .join("target")
        .join("debug")
        .join("redesmyn_desktop");
    if !bin_path.exists() {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!(
                "Desktop binary not found at {}; run `cargo build -p redesmyn_desktop`.",
                bin_path.display()
            ),
        ));
    }

    let repo_root = workspace_root.parent().ok_or_else(|| {
        ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!(
                "Rust workspace root {} has no parent; cannot infer repo root for UI driver.",
                workspace_root.display()
            ),
        )
    })?;

    let mut cmd = Command::new(bin_path);
    // The desktop app discovers epics relative to the repository root (e.g. `epics/`).
    // `workspace_root` is the `rust/` directory, so use its parent.
    cmd.current_dir(repo_root)
        .env("REDESMYN_UI_DRIVER_SOCKET_PATH", socket_path)
        .env("REDESMYN_TEST_ARTIFACTS_DIR", artifacts_dir)
        .env("REDESMYN_UI_TEST_MODE", "1")
        .env("REDESMYN_UI_TEST_THEME", "dark")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if let Some(rust_db_path) = rust_db_path {
        cmd.env("REDESMYN_RUST__CONTROL_PLANE__DB__PATH", rust_db_path);
    }

    cmd.spawn().map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("Failed to start desktop app: {err}"),
        )
    })
}

#[cfg(unix)]
fn wait_for_path(path: &Path, timeout: Duration) -> Result<(), ErrorEnvelope> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if path.exists() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(20));
    }
    Err(ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        format!("Timed out waiting for {}", path.display()),
    ))
}

#[cfg(unix)]
fn shutdown_desktop(child: &mut Option<Child>, keep_open: bool) {
    if keep_open {
        return;
    }
    let Some(mut child) = child.take() else {
        return;
    };
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(unix)]
fn require_ok_response(
    result: Result<UiDriverResponseResult, ErrorEnvelope>,
    op: &'static str,
) -> Result<(), ErrorEnvelope> {
    match result? {
        UiDriverResponseResult::Error(err) => Err(err),
        UiDriverResponseResult::OpenEpic(_)
        | UiDriverResponseResult::SelectTask(_)
        | UiDriverResponseResult::TriggerRefresh(_)
        | UiDriverResponseResult::SetSettingsDialogOpen(_)
        | UiDriverResponseResult::SetSettingsDialogSection(_)
        | UiDriverResponseResult::WaitForSnapshot(_)
        | UiDriverResponseResult::WaitForIdle(_)
        | UiDriverResponseResult::GraphSelectNode(_)
        | UiDriverResponseResult::GraphClearSelection(_)
        | UiDriverResponseResult::GraphToggleExpandedTaskCard(_)
        | UiDriverResponseResult::GraphMultiSelectAddNode(_)
        | UiDriverResponseResult::GraphMultiSelectRemoveNode(_)
        | UiDriverResponseResult::SessionSettingsMenuSetOpen(_)
        | UiDriverResponseResult::SessionSettingsMenuSendKey(_)
        | UiDriverResponseResult::TaskFiltersMenuSetOpen(_) => Ok(()),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!("Unexpected {op} response: {other:?}"),
        )),
    }
}

#[cfg(unix)]
fn send_ui_driver_request(
    stream: &mut UnixStream,
    payload: UiDriverRequestPayload,
    read_timeout: Duration,
) -> Result<UiDriverResponseResult, ErrorEnvelope> {
    let _ = stream.set_read_timeout(Some(read_timeout));

    let request_id = RequestId::new();
    let request = UiDriverRequest {
        request_id,
        payload,
    };
    let frame = UiDriverFrame::new(ProtocolEnvelope::new(), UiDriverMessage::Request(request));
    let out = frame.to_protobuf().encode_to_vec();

    write_framed_bytes(stream, &out, DEFAULT_MAX_FRAME_LEN).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("Failed to write ui driver frame: {err}"),
        )
    })?;

    let Some(bytes) = read_framed_bytes(stream, DEFAULT_MAX_FRAME_LEN).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("Failed to read ui driver frame: {err}"),
        )
    })?
    else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "UI driver connection closed.",
        ));
    };

    let proto = pbv1::UiDriverFrame::decode(bytes.as_slice()).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!("Invalid protobuf ui driver frame: {err}"),
        )
    })?;
    let frame = UiDriverFrame::try_from_protobuf(proto)?;
    let UiDriverMessage::Response(response) = frame.message else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Expected ui driver response frame.",
        ));
    };

    if response.request_id != request_id {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!(
                "Mismatched ui driver response request_id: expected {request_id}, got {}",
                response.request_id
            ),
        ));
    }

    Ok(response.result)
}

#[cfg(unix)]
fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut read_total = 0;
    while read_total < buf.len() {
        match reader.read(&mut buf[read_total..]) {
            Ok(0) => {
                if read_total == 0 {
                    return Ok(false);
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected end of file",
                ));
            }
            Ok(n) => read_total += n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(true)
}

#[cfg(unix)]
fn read_framed_bytes<R: Read>(
    reader: &mut R,
    max_frame_len: usize,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0_u8; 4];
    let has_more = read_exact_or_eof(reader, &mut len_buf)?;
    if !has_more {
        return Ok(None);
    }

    let len = u32::from_be_bytes(len_buf) as usize;
    if len > max_frame_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} > {max_frame_len} bytes"),
        ));
    }

    let mut payload = vec![0_u8; len];
    read_exact_or_eof(reader, &mut payload)?;
    Ok(Some(payload))
}

#[cfg(unix)]
fn write_framed_bytes<W: Write>(
    writer: &mut W,
    bytes: &[u8],
    max_frame_len: usize,
) -> std::io::Result<()> {
    let len = bytes.len();
    if len > max_frame_len || len > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} > {max_frame_len} bytes"),
        ));
    }

    writer.write_all(&(len as u32).to_be_bytes())?;
    writer.write_all(bytes)?;
    writer.flush()?;
    Ok(())
}

fn sanitize_label(label: &str) -> String {
    let label = label.trim();
    if label.is_empty() {
        return "checkpoint".to_string();
    }

    label
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

fn infer_artifact_paths(
    label: &str,
    screenshot: &CaptureScreenshotResponse,
) -> Result<(Option<String>, Option<String>), ErrorEnvelope> {
    if screenshot.png_path.trim().is_empty() {
        return Ok((None, None));
    }

    let screenshot_path = PathBuf::from(&screenshot.png_path);
    let Some(run_dir) = screenshot_path.parent() else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!("Invalid screenshot path: {}", screenshot.png_path),
        ));
    };

    let snapshot_path = run_dir.join(format!("ui_snapshot_{}.json", sanitize_label(label)));

    Ok((
        Some(screenshot_path.to_string_lossy().to_string()),
        Some(snapshot_path.to_string_lossy().to_string()),
    ))
}
