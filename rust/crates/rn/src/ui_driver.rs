use std::{
    env,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use clap::{Args, Subcommand};
use prost::Message as _;
use redesmyn_ids::{RequestId, TaskId};
use redesmyn_protocol::pb::redesmyn::protocol::v1 as pbv1;
use redesmyn_protocol::ui_driver::{
    CaptureScreenshotRequest, CaptureScreenshotResponse, CreateChatSessionRequest,
    CreateChatSessionResponse, OpenEpicRequest, SelectGraphNodeRequest,
    SessionSettingsMenuSendKeyRequest, SessionSettingsMenuSetOpenRequest,
    UiDriverFrame, UiDriverMessage, UiDriverRequest, UiDriverRequestPayload,
    UiDriverResponseResult, UiPrimaryView, UiScreenshotWindow, UiSnapshotPredicate,
    WaitForUiIdleRequest, WaitForUiSnapshotRequest,
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
    /// Runs a deterministic graph smoke flow:
    /// open epic → select node → capture artifacts.
    ///
    /// Prefer `--launch` (sets `REDESMYN_UI_TEST_MODE=1`) unless you are connecting to an
    /// already-running test instance.
    GraphSmoke(UiDriverGraphSmokeArgs),
    /// Opens the session Settings menu and exercises basic keyboard navigation.
    SettingsMenuSmoke(UiDriverSettingsMenuSmokeArgs),
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
struct SettingsMenuSmokeReport {
    epic_slug: String,
    socket_path: String,
    chat_session_id: String,
    screenshot_path: Option<String>,
    ui_snapshot_path: Option<String>,
    artifacts_dir: Option<String>,
}

pub(crate) fn ui_driver(cmd: UiDriverCommands, output: &Output) -> CommandOutcome {
    match cmd {
        UiDriverCommands::Smoke(args) => ui_driver_smoke(args, output),
        UiDriverCommands::GraphSmoke(args) => ui_driver_graph_smoke(args, output),
        UiDriverCommands::SettingsMenuSmoke(args) => ui_driver_settings_menu_smoke(args, output),
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

            match spawn_desktop_app(&workspace_root, &socket_path, artifacts_dir) {
                Ok(child) => desktop_child = Some(child),
                Err(err) => return CommandOutcome::Failure(err),
            }

            if let Err(err) =
                wait_for_path(&socket_path, Duration::from_millis(args.startup_timeout_ms))
            {
                if let Some(mut child) = desktop_child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return CommandOutcome::Failure(err);
            }
        }

        let mut conn = match UnixStream::connect(&socket_path) {
            Ok(conn) => conn,
            Err(err) => {
                if let Some(mut child) = desktop_child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!(
                        "Failed to connect to UI driver socket {}: {err}",
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
                UiDriverRequestPayload::OpenEpic(OpenEpicRequest {
                    epic_slug: args.epic.clone(),
                }),
                Duration::from_secs(2),
            ),
            "open_epic",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
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
        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::WaitForSnapshot(WaitForUiSnapshotRequest {
                    timeout_ms: args.timeout_ms,
                    predicate: wait_predicate,
                }),
                wait_timeout,
            ),
            "wait_for_snapshot",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        let chat_title = args.chat_title.clone().unwrap_or_default();
        let chat_session_id = match send_ui_driver_request(
            &mut conn,
            UiDriverRequestPayload::CreateChatSession(CreateChatSessionRequest {
                name_hint: chat_title,
            }),
            wait_timeout,
        ) {
            Ok(UiDriverResponseResult::CreateChatSession(CreateChatSessionResponse {
                session_id,
            })) => session_id.to_string(),
            Ok(UiDriverResponseResult::Error(err)) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
            Ok(other) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    format!("Unexpected create_chat_session response: {other:?}"),
                ));
            }
            Err(err) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
        };

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

        let report = SmokeReport {
            epic_slug: args.epic,
            socket_path: socket_path.to_string_lossy().to_string(),
            chat_session_id,
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: artifacts_dir.map(|dir| dir.to_string_lossy().to_string()),
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

            match spawn_desktop_app(&workspace_root, &socket_path, artifacts_dir) {
                Ok(child) => desktop_child = Some(child),
                Err(err) => return CommandOutcome::Failure(err),
            }

            if let Err(err) =
                wait_for_path(&socket_path, Duration::from_millis(args.startup_timeout_ms))
            {
                if let Some(mut child) = desktop_child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return CommandOutcome::Failure(err);
            }
        }

        let wait_timeout = Duration::from_millis(args.timeout_ms);

        let mut conn = match UnixStream::connect(&socket_path) {
            Ok(conn) => conn,
            Err(err) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("Failed to connect to UI driver socket: {err}"),
                ));
            }
        };

        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::OpenEpic(OpenEpicRequest { epic_slug: args.epic.clone() }),
                wait_timeout,
            ),
            "open_epic",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        let chat_title = args.chat_title.clone().unwrap_or_default();
        let chat_session_id = match send_ui_driver_request(
            &mut conn,
            UiDriverRequestPayload::CreateChatSession(CreateChatSessionRequest {
                name_hint: chat_title,
            }),
            wait_timeout,
        ) {
            Ok(UiDriverResponseResult::CreateChatSession(CreateChatSessionResponse {
                session_id,
            })) => session_id.to_string(),
            Ok(UiDriverResponseResult::Error(err)) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
            Ok(other) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    format!("Unexpected create_chat_session response: {other:?}"),
                ));
            }
            Err(err) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
        };

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

        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::SessionSettingsMenuSetOpen(SessionSettingsMenuSetOpenRequest {
                    open: true,
                }),
                wait_timeout,
            ),
            "settings_menu_set_open",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        for key in ["down", "right", "down"].iter().copied() {
            if let Err(err) = require_ok_response(
                send_ui_driver_request(
                    &mut conn,
                    UiDriverRequestPayload::SessionSettingsMenuSendKey(
                        SessionSettingsMenuSendKeyRequest { key: key.to_string() },
                    ),
                    wait_timeout,
                ),
                "settings_menu_send_key",
            ) {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
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

        let report = SettingsMenuSmokeReport {
            epic_slug: args.epic,
            socket_path: socket_path.to_string_lossy().to_string(),
            chat_session_id,
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: artifacts_dir.map(|dir| dir.to_string_lossy().to_string()),
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
        let mut desktop_child: Option<Child> = None;

        if args.task_id.is_none() && !args.launch {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "Missing --task-id. The default demo TaskId ([1; 16]) requires `--launch` (sets REDESMYN_UI_TEST_MODE=1).",
            ));
        }

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

            match spawn_desktop_app(&workspace_root, &socket_path, artifacts_dir) {
                Ok(child) => desktop_child = Some(child),
                Err(err) => return CommandOutcome::Failure(err),
            }

            if let Err(err) =
                wait_for_path(&socket_path, Duration::from_millis(args.startup_timeout_ms))
            {
                if let Some(mut child) = desktop_child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return CommandOutcome::Failure(err);
            }
        }

        let mut conn = match UnixStream::connect(&socket_path) {
            Ok(conn) => conn,
            Err(err) => {
                if let Some(mut child) = desktop_child.take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!(
                        "Failed to connect to UI driver socket {}: {err}",
                        socket_path.display()
                    ),
                ));
            }
        };

        let task_id = args
            .task_id
            .unwrap_or_else(|| TaskId::from_bytes([1_u8; 16]));

        let timeout = Duration::from_millis(args.timeout_ms);
        let wait_timeout = timeout + Duration::from_secs(2);

        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::OpenEpic(OpenEpicRequest {
                    epic_slug: args.epic.clone(),
                }),
                Duration::from_secs(2),
            ),
            "open_epic",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
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
        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::WaitForSnapshot(WaitForUiSnapshotRequest {
                    timeout_ms: args.timeout_ms,
                    predicate: wait_workspace_predicate,
                }),
                wait_timeout,
            ),
            "wait_for_snapshot",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        if let Err(err) = require_ok_response(
            send_ui_driver_request(
                &mut conn,
                UiDriverRequestPayload::GraphSelectNode(SelectGraphNodeRequest { task_id }),
                wait_timeout,
            ),
            "graph_select_node",
        ) {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(err);
        }

        let wait_selection_predicate = UiSnapshotPredicate {
            primary_view: None,
            epic_slug: String::new(),
            in_flight_empty: None,
            selected_task_id: Some(task_id),
            graph_layout_settled: None,
            graph_selection_settled: Some(true),
        };
        let snapshot = match send_ui_driver_request(
            &mut conn,
            UiDriverRequestPayload::WaitForSnapshot(WaitForUiSnapshotRequest {
                timeout_ms: args.timeout_ms,
                predicate: wait_selection_predicate,
            }),
            wait_timeout,
        ) {
            Ok(UiDriverResponseResult::WaitForSnapshot(resp)) => resp.snapshot,
            Ok(UiDriverResponseResult::Error(err)) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
            Ok(other) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    format!("Unexpected wait_for_snapshot response: {other:?}"),
                ));
            }
            Err(err) => {
                shutdown_desktop(&mut desktop_child, args.keep_open);
                return CommandOutcome::Failure(err);
            }
        };

        if !snapshot.graph.expanded_task_card_open
            || snapshot.graph.expanded_task_id != Some(task_id)
        {
            shutdown_desktop(&mut desktop_child, args.keep_open);
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                format!(
                    "Expected expanded task card to be open for {task_id}; got open={} task_id={:?}",
                    snapshot.graph.expanded_task_card_open, snapshot.graph.expanded_task_id
                ),
            ));
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

        let report = GraphSmokeReport {
            epic_slug: args.epic,
            task_id: task_id.to_string(),
            socket_path: socket_path.to_string_lossy().to_string(),
            screenshot_path,
            ui_snapshot_path,
            artifacts_dir: artifacts_dir.map(|dir| dir.to_string_lossy().to_string()),
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

    Command::new(bin_path)
        .current_dir(workspace_root)
        .env("REDESMYN_UI_DRIVER_SOCKET_PATH", socket_path)
        .env("REDESMYN_TEST_ARTIFACTS_DIR", artifacts_dir)
        .env("REDESMYN_UI_TEST_MODE", "1")
        .env("REDESMYN_UI_TEST_THEME", "dark")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| {
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
        | UiDriverResponseResult::WaitForSnapshot(_)
        | UiDriverResponseResult::WaitForIdle(_)
        | UiDriverResponseResult::GraphSelectNode(_)
        | UiDriverResponseResult::GraphClearSelection(_)
        | UiDriverResponseResult::GraphToggleExpandedTaskCard(_)
        | UiDriverResponseResult::GraphMultiSelectAddNode(_)
        | UiDriverResponseResult::GraphMultiSelectRemoveNode(_)
        | UiDriverResponseResult::SessionSettingsMenuSetOpen(_)
        | UiDriverResponseResult::SessionSettingsMenuSendKey(_) => Ok(()),
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
