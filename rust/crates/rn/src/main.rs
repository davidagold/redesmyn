use std::{
    env,
    ffi::OsString,
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::{Duration, Instant},
};

use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(unix)]
use redesmyn_client_api::uds::{UdsConnectOptions, connect_uds};
use redesmyn_config::{discover_repo_root_from, legacy_db_path, rust_db_path};
use redesmyn_protocol::client::{
    AgentKind, AgentMessageConflictAction, CommandState, CreateCommandRequest, GetEpicGraphRequest,
    RequestPayload, ResponseResult, StartAgentRequest, WaitForCommandRequest,
};
use redesmyn_protocol::sync_commands::{LOCAL_SYNC_FROM_DOCS_KIND, LocalSyncFromDocsCommand};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, ProtocolEnvelope, Scope};
use serde::Serialize;

mod protocol;
mod ui_driver;

fn exit_code_from_i32(code: i32) -> ExitCode {
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}

fn main() -> ExitCode {
    let output_for_clap_errors = OutputFormat::detect_from_env_args();
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => return exit_from_clap_error(err, output_for_clap_errors),
    };

    let output = Output::new(if cli.global.json {
        OutputFormat::Json
    } else {
        cli.global.output
    });

    // Keep `rn-rs --help` zippy: clap will handle `--help/--version` and exit before we reach this.
    redesmyn_logging::init();

    match run(cli, &output) {
        CommandOutcome::Success => ExitCode::SUCCESS,
        CommandOutcome::ExitCode(code) => exit_code_from_i32(code),
        CommandOutcome::Failure(err) => {
            output.print_error(&err);
            exit_code_from_i32(err.exit_code())
        }
    }
}

fn exit_from_clap_error(err: clap::Error, output_format: OutputFormat) -> ExitCode {
    let kind = err.kind();
    let exit_code = match kind {
        clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
            ExitCode::SUCCESS
        }
        _ => exit_code_from_i32(ErrorCategory::InvalidRequest.exit_code()),
    };

    // For `--help`/`--version`, respect clap’s default rendering/stream choice.
    if matches!(
        kind,
        clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
    ) {
        let _ = err.print();
        return exit_code;
    }

    match output_format {
        OutputFormat::Human => {
            let _ = err.print();
        }
        OutputFormat::Json => {
            let error = ErrorEnvelope::new(ErrorCategory::InvalidRequest, err.to_string());
            Output::new(OutputFormat::Json).print_error(&error);
        }
    }

    exit_code
}

fn run(cli: Cli, output: &Output) -> CommandOutcome {
    match cli.command {
        Commands::Doctor(args) => doctor(args, output),
        Commands::Sync(args) => sync(args, output),
        Commands::Task(cmd) => task(cmd, output),
        Commands::Protocol(cmd) => protocol::protocol(cmd, output),
        Commands::UiDriver(cmd) => ui_driver::ui_driver(cmd, output),
        Commands::Db(cmd) => db(cmd, output),
        Commands::Version(args) => version(args, output),
        Commands::Bench(cmd) => bench(cmd, output),
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "rn-rs",
    about = "Redesmyn CLI (Rust port)",
    version,
    arg_required_else_help = true
)]
struct Cli {
    #[command(flatten)]
    global: GlobalArgs,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Args)]
struct GlobalArgs {
    /// Output format for command results and errors.
    #[arg(long, value_enum, default_value_t = OutputFormat::Human, global = true)]
    output: OutputFormat,

    /// Shorthand for `--output json`.
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
enum OutputFormat {
    Human,
    Json,
}

impl OutputFormat {
    fn detect_from_env_args() -> Self {
        let mut args = env::args_os().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--json" {
                return Self::Json;
            }
            if arg == "--output" {
                if matches!(args.next(), Some(val) if val == "json") {
                    return Self::Json;
                }
                continue;
            }
            if arg == "--output=json" {
                return Self::Json;
            }
        }
        Self::Human
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Checks environment and configuration basics.
    Doctor(DoctorArgs),

    /// Synchronize local task docs into the running control plane.
    Sync(SyncArgs),

    /// Task orchestration helpers.
    #[command(subcommand)]
    Task(TaskCommands),

    /// Protocol tooling.
    #[command(subcommand)]
    Protocol(protocol::ProtocolCommands),

    /// Desktop UI automation driver tools.
    #[command(subcommand)]
    UiDriver(ui_driver::UiDriverCommands),

    /// DB tooling.
    #[command(subcommand)]
    Db(DbCommands),

    /// Prints version information.
    Version(VersionArgs),

    /// Simple benchmarking helpers for startup regression tracking.
    #[command(subcommand)]
    Bench(BenchCommands),
}

#[derive(Debug, Args)]
struct DoctorArgs {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "lower")]
enum SyncFrom {
    Local,
}

#[derive(Debug, Args)]
struct SyncArgs {
    /// Sync source (currently only `local` is supported in Rust CLI).
    #[arg(long, value_enum, default_value_t = SyncFrom::Local)]
    from: SyncFrom,

    /// Epic slug (defaults only if exactly one local epic exists).
    #[arg(long)]
    epic: Option<String>,

    /// Path within the repo to sync from (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Keep parity with legacy CLI surface; branch materialization is delegated to control-plane sync.
    #[arg(long, default_value_t = true)]
    create_branches: bool,

    /// Timeout waiting for sync command completion.
    #[arg(long, default_value_t = 30_000)]
    timeout_ms: u64,
}

#[derive(Debug, Subcommand)]
enum TaskCommands {
    /// Start a task agent session (same control-plane request path as the desktop UI).
    Start(TaskStartArgs),
}

#[derive(Debug, Args)]
struct TaskStartArgs {
    /// Epic slug (defaults only if exactly one local epic exists).
    #[arg(long)]
    epic: Option<String>,

    /// Task slug within the epic (e.g. `T-3`).
    #[arg(long)]
    task: String,

    /// Path within the repo to operate in (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Optional initial prompt to seed the new session.
    #[arg(long)]
    prompt: Option<String>,

    /// Timeout waiting for the start command to reach a terminal state.
    #[arg(long, default_value_t = 30_000)]
    timeout_ms: u64,
}

#[derive(Debug, Subcommand)]
enum DbCommands {
    /// Imports selected legacy Python DB state into the Rust DB (safe; idempotent).
    ImportLegacy(DbImportLegacyArgs),
}

#[derive(Debug, Args)]
struct DbImportLegacyArgs {
    /// Path within the repo to import from (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Legacy Python DB path (defaults to `<repo>/.redesmyn/redesmyn.sqlite3`).
    #[arg(long)]
    legacy_db_path: Option<PathBuf>,

    /// Rust control-plane DB path (defaults to `<repo>/.redesmyn/redesmyn_rust.sqlite3`).
    #[arg(long)]
    rust_db_path: Option<PathBuf>,

    /// Report what would be imported without writing.
    #[arg(long)]
    dry_run: bool,
}

#[derive(Debug, Args)]
struct VersionArgs {
    /// Print only the version string.
    #[arg(long)]
    short: bool,
}

#[derive(Debug, Subcommand)]
enum BenchCommands {
    /// Benchmarks `rn-rs --help` startup time (wall-clock, includes process spawn).
    Startup(BenchStartupArgs),
}

#[derive(Debug, Args)]
struct BenchStartupArgs {
    /// Number of warmup runs (not counted).
    #[arg(long, default_value_t = 5)]
    warmup: u32,

    /// Number of timed runs.
    #[arg(long, default_value_t = 50)]
    iterations: u32,
}

#[derive(Debug)]
enum CommandOutcome {
    Success,
    ExitCode(i32),
    Failure(ErrorEnvelope),
}

struct Output {
    format: OutputFormat,
}

impl Output {
    fn new(format: OutputFormat) -> Self {
        Self { format }
    }

    fn print_json_stdout<T: Serialize>(&self, value: &T) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        serde_json::to_writer(&mut stdout, value)?;
        writeln!(&mut stdout)
    }

    fn print_json_stderr<T: Serialize>(&self, value: &T) -> io::Result<()> {
        let mut stderr = io::stderr().lock();
        serde_json::to_writer(&mut stderr, value)?;
        writeln!(&mut stderr)
    }

    fn print_error(&self, err: &ErrorEnvelope) {
        match self.format {
            OutputFormat::Human => {
                let _ = writeln!(io::stderr(), "error: {}", err.message);
            }
            OutputFormat::Json => {
                let _ = self.print_json_stderr(err);
            }
        }
    }
}

fn doctor(_args: DoctorArgs, output: &Output) -> CommandOutcome {
    #[derive(Debug, Clone, Copy, Serialize)]
    #[serde(rename_all = "snake_case")]
    #[allow(dead_code)]
    enum CheckStatus {
        Ok,
        Warn,
        Error,
    }

    #[derive(Debug, Serialize)]
    struct Check {
        name: &'static str,
        status: CheckStatus,
        message: String,
    }

    #[derive(Debug, Serialize)]
    struct Report {
        ok: bool,
        checks: Vec<Check>,
    }

    let mut checks = Vec::new();
    let mut ok = true;

    let cwd = match env::current_dir() {
        Ok(cwd) => cwd,
        Err(err) => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to read cwd: {err}"),
            ));
        }
    };

    checks.push(Check {
        name: "cwd",
        status: CheckStatus::Ok,
        message: cwd.display().to_string(),
    });

    let repo_root = find_git_root(&cwd);
    match &repo_root {
        Some(root) => checks.push(Check {
            name: "git_repo",
            status: CheckStatus::Ok,
            message: root.display().to_string(),
        }),
        None => {
            ok = false;
            checks.push(Check {
                name: "git_repo",
                status: CheckStatus::Error,
                message: "no .git directory found (run from within a repo)".to_owned(),
            });
        }
    }

    if let Some(repo_root) = repo_root {
        let rust_workspace = repo_root.join("rust").join("Cargo.toml");
        if rust_workspace.exists() {
            checks.push(Check {
                name: "rust_workspace",
                status: CheckStatus::Ok,
                message: rust_workspace.display().to_string(),
            });
        } else {
            ok = false;
            checks.push(Check {
                name: "rust_workspace",
                status: CheckStatus::Error,
                message: "missing rust/Cargo.toml (expected Rust workspace)".to_owned(),
            });
        }
    }

    let report = Report { ok, checks };

    match output.format {
        OutputFormat::Human => {
            for check in &report.checks {
                let status = match check.status {
                    CheckStatus::Ok => "ok",
                    CheckStatus::Warn => "warn",
                    CheckStatus::Error => "error",
                };
                println!("{status:>5}  {:<14} {}", check.name, check.message);
            }
        }
        OutputFormat::Json => {
            if let Err(err) = output.print_json_stdout(&report) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }
    }

    if report.ok {
        CommandOutcome::Success
    } else {
        CommandOutcome::ExitCode(ErrorCategory::InvalidRequest.exit_code())
    }
}

fn sync(args: SyncArgs, output: &Output) -> CommandOutcome {
    let SyncFrom::Local = args.from;

    let start = match args.repo {
        Some(path) => path,
        None => match env::current_dir() {
            Ok(cwd) => cwd,
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to read cwd: {err}"),
                ));
            }
        },
    };

    let repo_root = match discover_repo_root_from(&start) {
        Some(repo_root) => repo_root,
        None => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!(
                    "no .git directory found from {} (pass --repo to specify a repo)",
                    start.display()
                ),
            ));
        }
    };

    let epic_slug = if let Some(epic) = args.epic {
        epic
    } else if let Some(inferred) = infer_single_epic_slug_from_fs(&repo_root) {
        inferred
    } else {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "multiple epics found; pass --epic <slug>",
        ));
    };

    #[derive(Debug, Serialize)]
    struct SyncReport {
        repo_root: String,
        epic_slug: String,
        command_id: String,
        state: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .enable_io()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to initialize async runtime: {err}"),
            ));
        }
    };

    let result = match runtime.block_on(sync_from_local_via_control_plane(
        &repo_root,
        &epic_slug,
        args.create_branches,
        args.timeout_ms,
    )) {
        Ok(result) => result,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let report = SyncReport {
        repo_root: repo_root.display().to_string(),
        epic_slug,
        command_id: result.command_id,
        state: result.state,
        message: result.message.clone(),
    };

    match output.format {
        OutputFormat::Human => {
            if let Some(message) = &report.message {
                println!("{message}");
            } else {
                println!("sync completed: state={}", report.state);
            }
        }
        OutputFormat::Json => {
            if let Err(err) = output.print_json_stdout(&report) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }
    }

    if report.state == "succeeded" {
        CommandOutcome::Success
    } else {
        CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Conflict,
            report
                .message
                .unwrap_or_else(|| format!("sync failed (state={})", report.state)),
        ))
    }
}

fn task(cmd: TaskCommands, output: &Output) -> CommandOutcome {
    match cmd {
        TaskCommands::Start(args) => task_start(args, output),
    }
}

fn task_start(args: TaskStartArgs, output: &Output) -> CommandOutcome {
    let start = match args.repo {
        Some(path) => path,
        None => match env::current_dir() {
            Ok(cwd) => cwd,
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to read cwd: {err}"),
                ));
            }
        },
    };

    let repo_root = match discover_repo_root_from(&start) {
        Some(repo_root) => repo_root,
        None => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!(
                    "no .git directory found from {} (pass --repo to specify a repo)",
                    start.display()
                ),
            ));
        }
    };

    let epic_slug = if let Some(epic) = args.epic {
        epic
    } else if let Some(inferred) = infer_single_epic_slug_from_fs(&repo_root) {
        inferred
    } else {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "multiple epics found; pass --epic <slug>",
        ));
    };

    #[derive(Debug, Serialize)]
    struct TaskStartReport {
        repo_root: String,
        epic_slug: String,
        task_slug: String,
        session_id: String,
        command_id: String,
        state: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .enable_io()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to initialize async runtime: {err}"),
            ));
        }
    };

    let result = match runtime.block_on(start_task_agent_via_control_plane(
        &epic_slug,
        &args.task,
        args.prompt.clone(),
        args.timeout_ms,
    )) {
        Ok(result) => result,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let report = TaskStartReport {
        repo_root: repo_root.display().to_string(),
        epic_slug,
        task_slug: result.task_slug,
        session_id: result.session_id,
        command_id: result.command_id,
        state: result.state,
        message: result.message.clone(),
    };

    match output.format {
        OutputFormat::Human => {
            if report.state == "succeeded" {
                println!("started session: {}", report.session_id);
            } else if let Some(message) = &report.message {
                println!("{message}");
            } else {
                println!("start failed: state={}", report.state);
            }
        }
        OutputFormat::Json => {
            if let Err(err) = output.print_json_stdout(&report) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }
    }

    if report.state == "succeeded" {
        CommandOutcome::Success
    } else {
        CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            report
                .message
                .unwrap_or_else(|| format!("failed to start task agent (state={})", report.state)),
        ))
    }
}

#[derive(Debug)]
struct SyncCommandResult {
    command_id: String,
    state: String,
    message: Option<String>,
}

async fn sync_from_local_via_control_plane(
    repo_root: &Path,
    epic_slug: &str,
    create_branches: bool,
    timeout_ms: u64,
) -> Result<SyncCommandResult, ErrorEnvelope> {
    #[cfg(not(unix))]
    {
        let _ = (repo_root, epic_slug, create_branches, timeout_ms);
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "client API socket is only supported on unix platforms",
        ));
    }

    #[cfg(unix)]
    {
        let socket_path = load_default_client_socket_path()?;
        let options = UdsConnectOptions::new(socket_path);
        let (client, task) = connect_uds(options, 64).await?;
        tokio::spawn(task.run());

        let graph = match client
            .request(RequestPayload::GetEpicGraph(GetEpicGraphRequest {
                epic_slug: epic_slug.to_string(),
            }))
            .await?
        {
            ResponseResult::GetEpicGraph(resp) => resp.graph,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for GetEpicGraph",
                ));
            }
        };

        let workspace_id = graph.workspace_id.ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                "GetEpicGraph response missing workspace_id",
            )
        })?;
        let repo_id = graph.repo_id.ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                "GetEpicGraph response missing repo_id",
            )
        })?;

        let envelope = ProtocolEnvelope::new().with_scope(Scope::from(
            redesmyn_protocol::RepoScope::new(workspace_id, repo_id),
        ));

        let payload = serde_json::to_vec(&LocalSyncFromDocsCommand {
            repo_root: repo_root.display().to_string(),
            epic_slug: epic_slug.to_string(),
            create_branches,
        })
        .map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to encode sync payload: {err}"),
            )
        })?;

        let command = match client
            .request_with_envelope(
                envelope.clone(),
                RequestPayload::CreateCommand(CreateCommandRequest {
                    kind: LOCAL_SYNC_FROM_DOCS_KIND.to_string(),
                    target_task_id: None,
                    idempotency_key: None,
                    created_by: Some("rn-rs sync".to_string()),
                    json_payload: payload,
                }),
            )
            .await?
        {
            ResponseResult::CreateCommand(resp) => resp.command,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for CreateCommand",
                ));
            }
        };

        let waited = match client
            .request_with_envelope(
                envelope,
                RequestPayload::WaitForCommand(WaitForCommandRequest {
                    command_id: command.command_id,
                    terminal_states: vec![
                        CommandState::Succeeded,
                        CommandState::Failed,
                        CommandState::Canceled,
                    ],
                    timeout_ms,
                }),
            )
            .await?
        {
            ResponseResult::WaitForCommand(resp) => resp.command,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for WaitForCommand",
                ));
            }
        };

        let state = command_state_label(waited.state).to_string();

        Ok(SyncCommandResult {
            command_id: waited.command_id.to_string(),
            state,
            message: waited.last_update.and_then(|update| update.message),
        })
    }
}

fn command_state_label(state: CommandState) -> &'static str {
    match state {
        CommandState::Queued => "queued",
        CommandState::Accepted => "accepted",
        CommandState::Running => "running",
        CommandState::Blocked => "blocked",
        CommandState::Resumable => "resumable",
        CommandState::Succeeded => "succeeded",
        CommandState::Failed => "failed",
        CommandState::Canceled => "canceled",
        CommandState::Unknown => "unknown",
    }
}

#[derive(Debug)]
struct StartAgentCommandResult {
    task_slug: String,
    session_id: String,
    command_id: String,
    state: String,
    message: Option<String>,
}

async fn start_task_agent_via_control_plane(
    epic_slug: &str,
    task_slug: &str,
    prompt: Option<String>,
    timeout_ms: u64,
) -> Result<StartAgentCommandResult, ErrorEnvelope> {
    #[cfg(not(unix))]
    {
        let _ = (epic_slug, task_slug, prompt, timeout_ms);
        return Err(ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            "client API socket is only supported on unix platforms",
        ));
    }

    #[cfg(unix)]
    {
        let socket_path = load_default_client_socket_path()?;
        let options = UdsConnectOptions::new(socket_path);
        let (client, task) = connect_uds(options, 64).await?;
        tokio::spawn(task.run());

        let graph = match client
            .request(RequestPayload::GetEpicGraph(GetEpicGraphRequest {
                epic_slug: epic_slug.to_string(),
            }))
            .await?
        {
            ResponseResult::GetEpicGraph(resp) => resp.graph,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for GetEpicGraph",
                ));
            }
        };

        let workspace_id = graph.workspace_id.ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                "GetEpicGraph response missing workspace_id",
            )
        })?;
        let repo_id = graph.repo_id.ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                "GetEpicGraph response missing repo_id",
            )
        })?;

        let desired_slug = task_slug.trim();
        let desired_slug_lower = desired_slug.to_ascii_lowercase();
        let (task_slug, task_id) = graph
            .nodes
            .iter()
            .find(|node| node.task_slug == desired_slug)
            .or_else(|| {
                graph
                    .nodes
                    .iter()
                    .find(|node| node.task_slug.to_ascii_lowercase() == desired_slug_lower)
            })
            .map(|node| (node.task_slug.clone(), node.task_id))
            .ok_or_else(|| {
                let mut known: Vec<String> =
                    graph.nodes.iter().map(|node| node.task_slug.clone()).collect();
                known.sort();
                ErrorEnvelope::new(
                    ErrorCategory::NotFound,
                    format!(
                        "task not found in epic {epic_slug}: {desired_slug} (known: {})",
                        known.join(", ")
                    ),
                )
            })?;

        let task_id = task_id.ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("epic graph node is missing task_id for task {task_slug}"),
            )
        })?;

        let envelope = ProtocolEnvelope::new().with_scope(Scope::from(
            redesmyn_protocol::RepoScope::new(workspace_id, repo_id),
        ));

        let started = match client
            .request_with_envelope(
                envelope.clone(),
                RequestPayload::StartAgent(StartAgentRequest {
                    task_id,
                    agent_kind: AgentKind::Codex,
                    initial_prompt: prompt,
                    on_conflict: AgentMessageConflictAction::Fail,
                    session_model_selection: None,
                }),
            )
            .await?
        {
            ResponseResult::StartAgent(resp) => resp,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for StartAgent",
                ));
            }
        };

        let waited = match client
            .request_with_envelope(
                envelope,
                RequestPayload::WaitForCommand(WaitForCommandRequest {
                    command_id: started.command.command_id,
                    terminal_states: vec![
                        CommandState::Succeeded,
                        CommandState::Failed,
                        CommandState::Canceled,
                    ],
                    timeout_ms,
                }),
            )
            .await?
        {
            ResponseResult::WaitForCommand(resp) => resp.command,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for WaitForCommand",
                ));
            }
        };

        Ok(StartAgentCommandResult {
            task_slug,
            session_id: started.session_id.to_string(),
            command_id: waited.command_id.to_string(),
            state: command_state_label(waited.state).to_string(),
            message: waited.last_update.and_then(|update| update.message),
        })
    }
}

fn infer_single_epic_slug_from_fs(repo_root: &Path) -> Option<String> {
    let epics_dir = repo_root.join("epics");
    let mut slugs: Vec<String> = std::fs::read_dir(epics_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') {
                return None;
            }
            Some(name)
        })
        .collect();
    slugs.sort();
    if slugs.len() == 1 { slugs.pop() } else { None }
}

#[cfg(unix)]
fn load_default_client_socket_path() -> Result<PathBuf, ErrorEnvelope> {
    let config = redesmyn_config::load_rust_config(Default::default()).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Unavailable,
            format!("failed to load config: {err}"),
        )
    })?;
    Ok(config.control_plane.api.client_socket_path)
}

fn find_git_root(start: &Path) -> Option<PathBuf> {
    for dir in start.ancestors() {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
    }
    None
}

fn db(cmd: DbCommands, output: &Output) -> CommandOutcome {
    match cmd {
        DbCommands::ImportLegacy(args) => db_import_legacy(args, output),
    }
}

fn db_import_legacy(args: DbImportLegacyArgs, output: &Output) -> CommandOutcome {
    #[derive(Debug, Serialize)]
    struct PlannedCounts {
        epics: usize,
        tasks: usize,
        session_previews: usize,
    }

    #[derive(Debug, Serialize)]
    struct AppliedCounts {
        workspace_id: String,
        repo_id: String,
        upserted_workspaces: u64,
        upserted_repositories: u64,
        upserted_epics: u64,
        upserted_tasks: u64,
        upserted_session_events: u64,
    }

    #[derive(Debug, Serialize)]
    struct Report {
        repo_root: String,
        legacy_db_path: String,
        rust_db_path: String,
        dry_run: bool,
        legacy_alembic_version: Option<String>,
        rust_sqlx_version_before: Option<i64>,
        rust_sqlx_version_after: Option<i64>,
        tables: [&'static str; 6],
        planned: PlannedCounts,
        #[serde(skip_serializing_if = "Option::is_none")]
        applied: Option<AppliedCounts>,
    }

    let start = match args.repo {
        Some(path) => path,
        None => match env::current_dir() {
            Ok(cwd) => cwd,
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to read cwd: {err}"),
                ));
            }
        },
    };

    let repo_root = match discover_repo_root_from(&start) {
        Some(repo_root) => repo_root,
        None => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                format!(
                    "no .git directory found from {} (pass --repo to specify a repo)",
                    start.display()
                ),
            ));
        }
    };

    let legacy_db_path = args
        .legacy_db_path
        .clone()
        .unwrap_or_else(|| legacy_db_path(&repo_root));
    let rust_db_path = args
        .rust_db_path
        .clone()
        .unwrap_or_else(|| rust_db_path(&repo_root));

    let options = redesmyn_storage::legacy_import::ImportLegacyOptions {
        repo_root: repo_root.clone(),
        legacy_db_path: legacy_db_path.clone(),
        rust_db_path: rust_db_path.clone(),
        dry_run: args.dry_run,
    };

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
    {
        Ok(runtime) => runtime,
        Err(err) => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to initialize async runtime: {err}"),
            ));
        }
    };

    let outcome = match runtime.block_on(redesmyn_storage::legacy_import::import_legacy(options)) {
        Ok(outcome) => outcome,
        Err(err) => return CommandOutcome::Failure(storage_error_to_envelope(err)),
    };

    let report = Report {
        repo_root: repo_root.display().to_string(),
        legacy_db_path: legacy_db_path.display().to_string(),
        rust_db_path: rust_db_path.display().to_string(),
        dry_run: args.dry_run,
        legacy_alembic_version: outcome.legacy_alembic_version.clone(),
        rust_sqlx_version_before: outcome.rust_sqlx_version_before,
        rust_sqlx_version_after: outcome.rust_sqlx_version_after,
        tables: [
            "workspaces",
            "repositories",
            "epics",
            "tasks",
            "session_events",
            "legacy_id_map",
        ],
        planned: PlannedCounts {
            epics: outcome.planned.epics,
            tasks: outcome.planned.tasks,
            session_previews: outcome.planned.session_previews,
        },
        applied: outcome.applied.as_ref().map(|applied| AppliedCounts {
            workspace_id: applied.workspace_id.to_string(),
            repo_id: applied.repo_id.to_string(),
            upserted_workspaces: applied.upserted_workspaces,
            upserted_repositories: applied.upserted_repositories,
            upserted_epics: applied.upserted_epics,
            upserted_tasks: applied.upserted_tasks,
            upserted_session_events: applied.upserted_session_events,
        }),
    };

    match output.format {
        OutputFormat::Human => {
            println!("repo:      {}", report.repo_root);
            println!("legacy db: {}", report.legacy_db_path);
            println!("rust db:   {}", report.rust_db_path);
            if let Some(version) = &report.legacy_alembic_version {
                println!("legacy alembic: {version}");
            }
            if let Some(version) = report.rust_sqlx_version_before {
                println!("rust sqlx (before): {version}");
            }
            if let Some(version) = report.rust_sqlx_version_after {
                println!("rust sqlx (after):  {version}");
            }
            if report.dry_run {
                println!(
                    "would import: epics={} tasks={} session_previews={}",
                    report.planned.epics, report.planned.tasks, report.planned.session_previews
                );
            } else if let Some(applied) = &report.applied {
                println!(
                    "upserted: workspaces={} repositories={} epics={} tasks={} session_events={}",
                    applied.upserted_workspaces,
                    applied.upserted_repositories,
                    applied.upserted_epics,
                    applied.upserted_tasks,
                    applied.upserted_session_events
                );
            }
        }
        OutputFormat::Json => {
            if let Err(err) = output.print_json_stdout(&report) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }
    }

    CommandOutcome::Success
}

fn storage_error_to_envelope(err: redesmyn_storage::StorageError) -> ErrorEnvelope {
    use redesmyn_storage::StorageError as E;

    match err {
        E::LegacyDbNotFound { path } => ErrorEnvelope::new(
            ErrorCategory::NotFound,
            format!("legacy DB not found: {}", path.display()),
        ),
        E::LegacyRepoNotFound { repo_root } => ErrorEnvelope::new(
            ErrorCategory::NotFound,
            format!(
                "legacy DB does not contain repository metadata for {}",
                repo_root.display()
            ),
        ),
        E::Conflict { message } => ErrorEnvelope::new(ErrorCategory::Conflict, message),
        E::InvalidData { message } => ErrorEnvelope::new(ErrorCategory::InvalidRequest, message),
        E::LegacyDbSnapshot { path, source } => ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("failed to snapshot legacy DB {}: {source}", path.display()),
        ),
        E::CreateDbDir { path, source } => ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("failed to create db directory {}: {source}", path.display()),
        ),
        E::Migrate(source) => ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("failed to apply migrations: {source}"),
        ),
        E::Sqlx(source) => {
            ErrorEnvelope::new(ErrorCategory::Internal, format!("db error: {source}"))
        }
    }
}

fn version(args: VersionArgs, output: &Output) -> CommandOutcome {
    #[derive(Serialize)]
    struct VersionInfo<'a> {
        name: &'a str,
        version: &'a str,
    }

    let name = "rn-rs";
    let version = env!("CARGO_PKG_VERSION");

    match output.format {
        OutputFormat::Human => {
            if args.short {
                println!("{version}");
            } else {
                println!("{name} {version}");
            }
        }
        OutputFormat::Json => {
            if let Err(err) = output.print_json_stdout(&VersionInfo { name, version }) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }
    }

    CommandOutcome::Success
}

fn bench(cmd: BenchCommands, output: &Output) -> CommandOutcome {
    match cmd {
        BenchCommands::Startup(args) => bench_startup(args, output),
    }
}

fn bench_startup(args: BenchStartupArgs, output: &Output) -> CommandOutcome {
    #[derive(Serialize)]
    struct ResultRow {
        warmup: u32,
        iterations: u32,
        min_ms: f64,
        p50_ms: f64,
        p90_ms: f64,
        mean_ms: f64,
        max_ms: f64,
    }

    let exe = match env::current_exe() {
        Ok(exe) => exe,
        Err(err) => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to locate current executable: {err}"),
            ));
        }
    };

    let help_args: [OsString; 1] = ["--help".into()];

    for _ in 0..args.warmup {
        let child = Command::new(&exe)
            .args(&help_args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();

        match child {
            Ok(mut child) => {
                if let Ok(status) = child.wait() {
                    if !status.success() {
                        return CommandOutcome::Failure(ErrorEnvelope::new(
                            ErrorCategory::Internal,
                            format!("warmup run exited non-zero: {status}"),
                        ));
                    }
                }
            }
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("failed to spawn child process: {err}"),
                ));
            }
        }
    }

    let mut samples = Vec::with_capacity(args.iterations as usize);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let status = Command::new(&exe)
            .args(&help_args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let elapsed = start.elapsed();

        match status {
            Ok(status) if status.success() => {
                samples.push(elapsed);
            }
            Ok(status) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("bench run exited non-zero: {status}"),
                ));
            }
            Err(err) => {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    format!("failed to spawn child process: {err}"),
                ));
            }
        }
    }

    let mut sorted = samples.clone();
    sorted.sort_unstable_by_key(Duration::as_nanos);

    let min = sorted.first().copied().unwrap_or_default();
    let max = sorted.last().copied().unwrap_or_default();
    let p50 = percentile(&sorted, 0.50);
    let p90 = percentile(&sorted, 0.90);
    let mean = mean(&samples);

    let row = ResultRow {
        warmup: args.warmup,
        iterations: args.iterations,
        min_ms: min.as_secs_f64() * 1000.0,
        p50_ms: p50.as_secs_f64() * 1000.0,
        p90_ms: p90.as_secs_f64() * 1000.0,
        mean_ms: mean.as_secs_f64() * 1000.0,
        max_ms: max.as_secs_f64() * 1000.0,
    };

    match output.format {
        OutputFormat::Human => {
            println!("warmup:     {}", row.warmup);
            println!("iterations: {}", row.iterations);
            println!("min:        {:.3} ms", row.min_ms);
            println!("p50:        {:.3} ms", row.p50_ms);
            println!("p90:        {:.3} ms", row.p90_ms);
            println!("mean:       {:.3} ms", row.mean_ms);
            println!("max:        {:.3} ms", row.max_ms);
        }
        OutputFormat::Json => {
            if let Err(err) = output.print_json_stdout(&row) {
                return CommandOutcome::Failure(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    format!("failed to write output: {err}"),
                ));
            }
        }
    }

    CommandOutcome::Success
}

fn mean(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::default();
    }
    let nanos_sum: u128 = samples.iter().map(|d| d.as_nanos()).sum();
    Duration::from_nanos((nanos_sum / samples.len() as u128) as u64)
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::default();
    }
    let clamped = p.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * clamped).round() as usize;
    sorted[idx]
}
