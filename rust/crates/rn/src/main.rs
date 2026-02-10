use std::{
    env,
    ffi::OsString,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::{Duration, Instant},
};

use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(unix)]
use redesmyn_client_api::uds::{UdsConnectOptions, connect_uds};
use redesmyn_config::{
    discover_repo_root_from, global_config_path, legacy_db_path, repo_config_path, rust_db_path,
};
use redesmyn_ids::{RepoId, TaskId, WorkspaceId};
use redesmyn_protocol::client::{
    AgentKind, AgentMessageConflictAction, CommandState, CreateCommandRequest, GetEpicGraphRequest,
    GetLatestTaskSessionRequest, GetSessionEventsRequest, ModelReasoningEffort, RequestPayload,
    ResponseResult, RestartAgentRequest, SendTaskAgentMessageRequest, SessionEventCursor,
    SessionEventKindFilter, SessionModelSelection, StartAgentRequest, StopAgentRequest,
    TaskAgentMessageConversationContinuity, TaskAgentMessageDelivery, WaitForCommandRequest,
};
use redesmyn_protocol::prelude::BUILT_IN_PRELUDE_TEMPLATE;
use redesmyn_protocol::sync_commands::{LOCAL_SYNC_FROM_DOCS_KIND, LocalSyncFromDocsCommand};
use redesmyn_protocol::{
    CodexApprovalPolicy, CodexSandboxPolicy, ErrorCategory, ErrorEnvelope, ProtocolEnvelope, Scope,
    SessionEvent, SessionEventKind, TurnState,
};
use serde::Serialize;
use toml_edit::DocumentMut;

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
    /// Restart a task agent session through the control plane.
    Restart(TaskRestartArgs),
    /// Stop active task agent sessions through the control plane.
    Stop(TaskStopArgs),
    /// Send a durable message to a task agent through the control plane.
    Send(TaskSendArgs),
    /// Read durable task session history through the control plane.
    History(TaskHistoryArgs),
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

#[derive(Debug, Args)]
struct TaskRestartArgs {
    /// Epic slug (defaults only if exactly one local epic exists).
    #[arg(long)]
    epic: Option<String>,

    /// Task slug within the epic (e.g. `T-3`).
    #[arg(long)]
    task: String,

    /// Path within the repo to operate in (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Optional initial prompt to seed the restarted session.
    #[arg(long)]
    prompt: Option<String>,

    /// Timeout waiting for the restart command to reach a terminal state.
    #[arg(long, default_value_t = 30_000)]
    timeout_ms: u64,
}

#[derive(Debug, Args)]
struct TaskStopArgs {
    /// Epic slug (defaults only if exactly one local epic exists).
    #[arg(long)]
    epic: Option<String>,

    /// Task slug within the epic (e.g. `T-3`).
    #[arg(long)]
    task: String,

    /// Path within the repo to operate in (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Timeout waiting for the stop command to reach a terminal state.
    #[arg(long, default_value_t = 30_000)]
    timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "snake_case")]
enum TaskSendConflictAction {
    Fail,
    InterruptTurn,
    StopSessionAndStartNew,
}

impl From<TaskSendConflictAction> for AgentMessageConflictAction {
    fn from(value: TaskSendConflictAction) -> Self {
        match value {
            TaskSendConflictAction::Fail => AgentMessageConflictAction::Fail,
            TaskSendConflictAction::InterruptTurn => AgentMessageConflictAction::InterruptTurn,
            TaskSendConflictAction::StopSessionAndStartNew => {
                AgentMessageConflictAction::StopSessionAndStartNew
            }
        }
    }
}

#[derive(Debug, Args)]
struct TaskSendArgs {
    /// Epic slug (defaults only if exactly one local epic exists).
    #[arg(long)]
    epic: Option<String>,

    /// Task slug within the epic (e.g. `T-3`).
    #[arg(long)]
    task: String,

    /// Message text to send.
    #[arg(long)]
    message: String,

    /// Intent metadata for orchestration/UI interpretation.
    #[arg(long, default_value = "unspecified")]
    intent: String,

    /// Conflict behavior if another turn/session is in flight.
    #[arg(long, value_enum, default_value_t = TaskSendConflictAction::Fail)]
    on_conflict: TaskSendConflictAction,

    /// Shorthand for `--on-conflict interrupt_turn`.
    #[arg(long)]
    interrupt: bool,

    /// Path within the repo to operate in (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,

    /// Timeout waiting for the resulting command to reach a terminal state.
    #[arg(long, default_value_t = 30_000)]
    timeout_ms: u64,
}

#[derive(Debug, Args)]
struct TaskHistoryArgs {
    /// Epic slug (defaults only if exactly one local epic exists).
    #[arg(long)]
    epic: Option<String>,

    /// Task slug within the epic (e.g. `T-3`).
    #[arg(long)]
    task: String,

    /// Number of events to read (newest N, rendered chronologically).
    #[arg(long, default_value_t = 100)]
    limit: u32,

    /// Include all durable event kinds instead of transcript-only kinds.
    #[arg(long)]
    all: bool,

    /// Path within the repo to operate in (defaults to current directory).
    #[arg(long)]
    repo: Option<PathBuf>,
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

    let runtime = match build_current_thread_runtime() {
        Ok(runtime) => runtime,
        Err(err) => return CommandOutcome::Failure(err),
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
        TaskCommands::Restart(args) => task_restart(args, output),
        TaskCommands::Stop(args) => task_stop(args, output),
        TaskCommands::Send(args) => task_send(args, output),
        TaskCommands::History(args) => task_history(args, output),
    }
}

#[derive(Debug)]
struct TaskCommandContext {
    repo_root: PathBuf,
    epic_slug: String,
}

fn resolve_task_command_context(
    repo: Option<&PathBuf>,
    epic: Option<&str>,
) -> Result<TaskCommandContext, ErrorEnvelope> {
    let start = match repo {
        Some(path) => path.clone(),
        None => env::current_dir().map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to read cwd: {err}"),
            )
        })?,
    };

    let repo_root = discover_repo_root_from(&start).ok_or_else(|| {
        ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!(
                "no .git directory found from {} (pass --repo to specify a repo)",
                start.display()
            ),
        )
    })?;

    let epic_slug = if let Some(epic) = epic {
        epic.to_string()
    } else if let Some(inferred) = infer_single_epic_slug_from_fs(&repo_root) {
        inferred
    } else {
        return Err(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "multiple epics found; pass --epic <slug>",
        ));
    };

    Ok(TaskCommandContext {
        repo_root,
        epic_slug,
    })
}

fn build_current_thread_runtime() -> Result<tokio::runtime::Runtime, ErrorEnvelope> {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .enable_io()
        .build()
        .map_err(|err| {
            ErrorEnvelope::new(
                ErrorCategory::Internal,
                format!("failed to initialize async runtime: {err}"),
            )
        })
}

fn build_task_initial_prompt(
    defaults: &TaskStartDefaults,
    prompt: Option<String>,
) -> Option<String> {
    let prelude_prompt = if defaults.send_prelude {
        defaults
            .prelude
            .clone()
            .or_else(|| Some(BUILT_IN_PRELUDE_TEMPLATE.to_string()))
    } else {
        None
    };
    let user_prompt = prompt
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    merge_task_start_prompts(prelude_prompt, user_prompt)
}

fn task_start(args: TaskStartArgs, output: &Output) -> CommandOutcome {
    let context = match resolve_task_command_context(args.repo.as_ref(), args.epic.as_deref()) {
        Ok(context) => context,
        Err(err) => return CommandOutcome::Failure(err),
    };
    let repo_root = context.repo_root;
    let epic_slug = context.epic_slug;

    let defaults = load_effective_task_start_defaults(&repo_root);
    let initial_prompt = build_task_initial_prompt(&defaults, args.prompt.clone());

    let session_model_selection =
        if defaults.model_id.is_none() && defaults.reasoning_effort.is_none() {
            None
        } else {
            Some(SessionModelSelection {
                model_id: defaults.model_id.clone(),
                reasoning_effort: defaults.reasoning_effort,
            })
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

    let runtime = match build_current_thread_runtime() {
        Ok(runtime) => runtime,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let result = match runtime.block_on(start_task_agent_via_control_plane(
        &epic_slug,
        &args.task,
        initial_prompt,
        session_model_selection,
        defaults.codex_approval_policy,
        defaults.codex_sandbox_policy.clone(),
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

fn task_restart(args: TaskRestartArgs, output: &Output) -> CommandOutcome {
    let context = match resolve_task_command_context(args.repo.as_ref(), args.epic.as_deref()) {
        Ok(context) => context,
        Err(err) => return CommandOutcome::Failure(err),
    };
    let repo_root = context.repo_root;
    let epic_slug = context.epic_slug;

    let defaults = load_effective_task_start_defaults(&repo_root);
    let initial_prompt = build_task_initial_prompt(&defaults, args.prompt.clone());

    #[derive(Debug, Serialize)]
    struct TaskRestartReport {
        repo_root: String,
        epic_slug: String,
        task_slug: String,
        session_id: String,
        command_id: String,
        state: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    let runtime = match build_current_thread_runtime() {
        Ok(runtime) => runtime,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let result = match runtime.block_on(restart_task_agent_via_control_plane(
        &epic_slug,
        &args.task,
        initial_prompt,
        args.timeout_ms,
    )) {
        Ok(result) => result,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let report = TaskRestartReport {
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
                println!("restarted session: {}", report.session_id);
            } else if let Some(message) = &report.message {
                println!("{message}");
            } else {
                println!("restart failed: state={}", report.state);
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
            report.message.unwrap_or_else(|| {
                format!("failed to restart task agent (state={})", report.state)
            }),
        ))
    }
}

fn task_stop(args: TaskStopArgs, output: &Output) -> CommandOutcome {
    let context = match resolve_task_command_context(args.repo.as_ref(), args.epic.as_deref()) {
        Ok(context) => context,
        Err(err) => return CommandOutcome::Failure(err),
    };
    let repo_root = context.repo_root;
    let epic_slug = context.epic_slug;

    #[derive(Debug, Serialize)]
    struct TaskStopReport {
        repo_root: String,
        epic_slug: String,
        task_slug: String,
        command_id: String,
        state: String,
        ended_session_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    let runtime = match build_current_thread_runtime() {
        Ok(runtime) => runtime,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let result = match runtime.block_on(stop_task_agent_via_control_plane(
        &epic_slug,
        &args.task,
        args.timeout_ms,
    )) {
        Ok(result) => result,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let report = TaskStopReport {
        repo_root: repo_root.display().to_string(),
        epic_slug,
        task_slug: result.task_slug,
        command_id: result.command_id,
        state: result.state,
        ended_session_ids: result.ended_session_ids,
        message: result.message.clone(),
    };

    match output.format {
        OutputFormat::Human => {
            if report.state == "succeeded" {
                if report.ended_session_ids.is_empty() {
                    println!("stop completed: no active sessions");
                } else {
                    println!("stopped {} session(s)", report.ended_session_ids.len());
                }
            } else if let Some(message) = &report.message {
                println!("{message}");
            } else {
                println!("stop failed: state={}", report.state);
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
                .unwrap_or_else(|| format!("failed to stop task agent (state={})", report.state)),
        ))
    }
}

fn task_send(args: TaskSendArgs, output: &Output) -> CommandOutcome {
    let context = match resolve_task_command_context(args.repo.as_ref(), args.epic.as_deref()) {
        Ok(context) => context,
        Err(err) => return CommandOutcome::Failure(err),
    };
    let repo_root = context.repo_root;
    let epic_slug = context.epic_slug;

    let intent = match normalize_task_message_intent(&args.intent) {
        Some(intent) => intent,
        None => {
            return CommandOutcome::Failure(ErrorEnvelope::new(
                ErrorCategory::InvalidRequest,
                "intent must contain only letters, numbers, '-' or '_'",
            ));
        }
    };
    if args.message.trim().is_empty() {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "message text is required",
        ));
    }

    if args.interrupt && args.on_conflict != TaskSendConflictAction::Fail {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "--interrupt cannot be combined with --on-conflict (set one or the other)",
        ));
    }

    #[derive(Debug, Serialize)]
    struct TaskSendReport {
        repo_root: String,
        epic_slug: String,
        task_slug: String,
        intent: String,
        session_id: String,
        command_id: String,
        state: String,
        delivery: String,
        conversation_continuity: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        warnings: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    }

    let on_conflict = if args.interrupt {
        AgentMessageConflictAction::InterruptTurn
    } else {
        args.on_conflict.into()
    };

    let runtime = match build_current_thread_runtime() {
        Ok(runtime) => runtime,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let result = match runtime.block_on(send_task_agent_message_via_control_plane(
        &epic_slug,
        &args.task,
        &args.message,
        &intent,
        on_conflict,
        args.timeout_ms,
    )) {
        Ok(result) => result,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let report = TaskSendReport {
        repo_root: repo_root.display().to_string(),
        epic_slug,
        task_slug: result.task_slug,
        intent,
        session_id: result.session_id,
        command_id: result.command_id,
        state: result.state,
        delivery: task_agent_message_delivery_label(result.delivery).to_string(),
        conversation_continuity: task_agent_message_conversation_continuity_label(
            result.conversation_continuity,
        )
        .to_string(),
        warnings: result.warnings,
        message: result.message.clone(),
    };

    match output.format {
        OutputFormat::Human => {
            if report.state == "succeeded" {
                println!(
                    "message delivered: session={} command={} delivery={}",
                    report.session_id, report.command_id, report.delivery
                );
            } else if let Some(message) = &report.message {
                println!("{message}");
            } else {
                println!("message delivery failed: state={}", report.state);
            }
            for warning in &report.warnings {
                println!("warning: {warning}");
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
                .unwrap_or_else(|| format!("failed to deliver message (state={})", report.state)),
        ))
    }
}

fn task_history(args: TaskHistoryArgs, output: &Output) -> CommandOutcome {
    if args.limit == 0 {
        return CommandOutcome::Failure(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "--limit must be greater than 0",
        ));
    }

    let context = match resolve_task_command_context(args.repo.as_ref(), args.epic.as_deref()) {
        Ok(context) => context,
        Err(err) => return CommandOutcome::Failure(err),
    };
    let repo_root = context.repo_root;
    let epic_slug = context.epic_slug;

    let runtime = match build_current_thread_runtime() {
        Ok(runtime) => runtime,
        Err(err) => return CommandOutcome::Failure(err),
    };

    let result = match runtime.block_on(load_task_history_via_control_plane(
        &epic_slug, &args.task, args.limit, args.all,
    )) {
        Ok(result) => result,
        Err(err) => return CommandOutcome::Failure(err),
    };

    #[derive(Debug, Serialize)]
    struct TaskHistoryEventReport {
        session_event_id: String,
        created_at: String,
        kind: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
    }

    #[derive(Debug, Serialize)]
    struct TaskHistoryReport {
        repo_root: String,
        epic_slug: String,
        task_slug: String,
        session_id: String,
        limit: u32,
        returned: usize,
        has_more: bool,
        events: Vec<TaskHistoryEventReport>,
    }

    let events: Vec<TaskHistoryEventReport> = result
        .events
        .iter()
        .map(|event| TaskHistoryEventReport {
            session_event_id: event.session_event_id.to_string(),
            created_at: format_timestamp(event.created_at),
            kind: session_event_kind_label(&event.kind).to_string(),
            role: transcript_role_for_event(event),
            text: transcript_text_for_event(event),
        })
        .collect();

    let report = TaskHistoryReport {
        repo_root: repo_root.display().to_string(),
        epic_slug,
        task_slug: result.task_slug,
        session_id: result.session_id,
        limit: args.limit,
        returned: events.len(),
        has_more: result.has_more,
        events,
    };

    match output.format {
        OutputFormat::Human => {
            println!("session: {}", report.session_id);
            for event in &result.events {
                if let Some(line) = render_history_line(event) {
                    println!("{line}");
                }
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

fn normalize_task_message_intent(raw: &str) -> Option<String> {
    let normalized = raw.trim().to_ascii_lowercase().replace('-', "_");
    if normalized.is_empty() {
        return None;
    }
    if normalized
        .chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
    {
        Some(normalized)
    } else {
        None
    }
}

#[derive(Debug)]
struct ResolvedTaskTarget {
    task_slug: String,
    task_id: TaskId,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
}

fn resolve_task_target_from_epic_graph(
    graph: &redesmyn_protocol::client::EpicGraph,
    epic_slug: &str,
    task_slug: &str,
) -> Result<ResolvedTaskTarget, ErrorEnvelope> {
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
    let (resolved_slug, task_id) = graph
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
            let mut known: Vec<String> = graph
                .nodes
                .iter()
                .map(|node| node.task_slug.clone())
                .collect();
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
            format!("epic graph node is missing task_id for task {resolved_slug}"),
        )
    })?;

    Ok(ResolvedTaskTarget {
        task_slug: resolved_slug,
        task_id,
        workspace_id,
        repo_id,
    })
}

fn session_event_kind_filters_for_history(all: bool) -> Vec<SessionEventKindFilter> {
    if all {
        Vec::new()
    } else {
        vec![
            SessionEventKindFilter::UserMessage,
            SessionEventKindFilter::AssistantMessage,
            SessionEventKindFilter::StatusUpdate,
        ]
    }
}

#[derive(Debug)]
struct TaskSendCommandResult {
    task_slug: String,
    session_id: String,
    command_id: String,
    state: String,
    message: Option<String>,
    delivery: TaskAgentMessageDelivery,
    conversation_continuity: TaskAgentMessageConversationContinuity,
    warnings: Vec<String>,
}

#[derive(Debug)]
struct TaskStopCommandResult {
    task_slug: String,
    command_id: String,
    state: String,
    message: Option<String>,
    ended_session_ids: Vec<String>,
}

#[derive(Debug)]
struct TaskRestartCommandResult {
    task_slug: String,
    session_id: String,
    command_id: String,
    state: String,
    message: Option<String>,
}

async fn restart_task_agent_via_control_plane(
    epic_slug: &str,
    task_slug: &str,
    initial_prompt: Option<String>,
    timeout_ms: u64,
) -> Result<TaskRestartCommandResult, ErrorEnvelope> {
    #[cfg(not(unix))]
    {
        let _ = (epic_slug, task_slug, initial_prompt, timeout_ms);
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

        let resolved = resolve_task_target_from_epic_graph(&graph, epic_slug, task_slug)?;
        let envelope = ProtocolEnvelope::new().with_scope(Scope::from(
            redesmyn_protocol::RepoScope::new(resolved.workspace_id, resolved.repo_id),
        ));

        let restarted = match client
            .request_with_envelope(
                envelope.clone(),
                RequestPayload::RestartAgent(RestartAgentRequest {
                    task_id: resolved.task_id,
                    agent_kind: AgentKind::Codex,
                    initial_prompt,
                    // Preserve sticky session settings across restarts by default.
                    session_model_selection: None,
                    codex_approval_policy: None,
                    codex_sandbox_policy: None,
                }),
            )
            .await?
        {
            ResponseResult::RestartAgent(resp) => resp,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for RestartAgent",
                ));
            }
        };

        let waited = match client
            .request_with_envelope(
                envelope,
                RequestPayload::WaitForCommand(WaitForCommandRequest {
                    command_id: restarted.command.command_id,
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

        Ok(TaskRestartCommandResult {
            task_slug: resolved.task_slug,
            session_id: restarted.session_id.to_string(),
            command_id: waited.command_id.to_string(),
            state: command_state_label(waited.state).to_string(),
            message: waited.last_update.and_then(|update| update.message),
        })
    }
}

async fn stop_task_agent_via_control_plane(
    epic_slug: &str,
    task_slug: &str,
    timeout_ms: u64,
) -> Result<TaskStopCommandResult, ErrorEnvelope> {
    #[cfg(not(unix))]
    {
        let _ = (epic_slug, task_slug, timeout_ms);
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

        let resolved = resolve_task_target_from_epic_graph(&graph, epic_slug, task_slug)?;

        let envelope = ProtocolEnvelope::new().with_scope(Scope::from(
            redesmyn_protocol::RepoScope::new(resolved.workspace_id, resolved.repo_id),
        ));

        let stopped = match client
            .request_with_envelope(
                envelope.clone(),
                RequestPayload::StopAgent(StopAgentRequest {
                    task_id: resolved.task_id,
                }),
            )
            .await?
        {
            ResponseResult::StopAgent(resp) => resp,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for StopAgent",
                ));
            }
        };

        let waited = match client
            .request_with_envelope(
                envelope,
                RequestPayload::WaitForCommand(WaitForCommandRequest {
                    command_id: stopped.command.command_id,
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

        Ok(TaskStopCommandResult {
            task_slug: resolved.task_slug,
            command_id: waited.command_id.to_string(),
            state: command_state_label(waited.state).to_string(),
            message: waited.last_update.and_then(|update| update.message),
            ended_session_ids: stopped
                .ended_session_ids
                .into_iter()
                .map(|session_id| session_id.to_string())
                .collect(),
        })
    }
}

async fn send_task_agent_message_via_control_plane(
    epic_slug: &str,
    task_slug: &str,
    message: &str,
    intent: &str,
    on_conflict: AgentMessageConflictAction,
    timeout_ms: u64,
) -> Result<TaskSendCommandResult, ErrorEnvelope> {
    #[cfg(not(unix))]
    {
        let _ = (
            epic_slug,
            task_slug,
            message,
            intent,
            on_conflict,
            timeout_ms,
        );
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

        let resolved = resolve_task_target_from_epic_graph(&graph, epic_slug, task_slug)?;

        let envelope = ProtocolEnvelope::new().with_scope(Scope::from(
            redesmyn_protocol::RepoScope::new(resolved.workspace_id, resolved.repo_id),
        ));

        let sent = match client
            .request_with_envelope(
                envelope.clone(),
                RequestPayload::SendTaskAgentMessage(SendTaskAgentMessageRequest {
                    task_id: resolved.task_id,
                    message: message.trim_end().to_string(),
                    intent: Some(intent.to_string()),
                    on_conflict,
                    interrupt: None,
                    agent_kind: AgentKind::Codex,
                }),
            )
            .await?
        {
            ResponseResult::SendTaskAgentMessage(resp) => resp,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for SendTaskAgentMessage",
                ));
            }
        };

        let waited = match client
            .request_with_envelope(
                envelope,
                RequestPayload::WaitForCommand(WaitForCommandRequest {
                    command_id: sent.command.command_id,
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

        Ok(TaskSendCommandResult {
            task_slug: resolved.task_slug,
            session_id: sent.session_id.to_string(),
            command_id: waited.command_id.to_string(),
            state: command_state_label(waited.state).to_string(),
            message: waited.last_update.and_then(|update| update.message),
            delivery: sent.delivery,
            conversation_continuity: sent.conversation_continuity,
            warnings: sent.warnings,
        })
    }
}

#[derive(Debug)]
struct TaskHistoryResult {
    task_slug: String,
    session_id: String,
    events: Vec<SessionEvent>,
    has_more: bool,
}

async fn load_task_history_via_control_plane(
    epic_slug: &str,
    task_slug: &str,
    limit: u32,
    all: bool,
) -> Result<TaskHistoryResult, ErrorEnvelope> {
    const MAX_SESSION_EVENTS_PAGE_LIMIT: u32 = 512;

    #[cfg(not(unix))]
    {
        let _ = (epic_slug, task_slug, limit, all);
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

        let resolved = resolve_task_target_from_epic_graph(&graph, epic_slug, task_slug)?;
        let envelope = ProtocolEnvelope::new().with_scope(Scope::from(
            redesmyn_protocol::RepoScope::new(resolved.workspace_id, resolved.repo_id),
        ));

        let session_id = match client
            .request_with_envelope(
                envelope.clone(),
                RequestPayload::GetLatestTaskSession(GetLatestTaskSessionRequest {
                    task_id: resolved.task_id,
                }),
            )
            .await?
        {
            ResponseResult::GetLatestTaskSession(resp) => resp.session_id,
            ResponseResult::Error(err) => return Err(err),
            _ => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Internal,
                    "unexpected response for GetLatestTaskSession",
                ));
            }
        }
        .ok_or_else(|| {
            ErrorEnvelope::new(
                ErrorCategory::NotFound,
                format!("no task session found for {}", resolved.task_slug),
            )
        })?;

        let kinds = session_event_kind_filters_for_history(all);
        let mut remaining = limit.max(1);
        let mut before: Option<SessionEventCursor> = None;
        let mut pages: Vec<Vec<SessionEvent>> = Vec::new();

        loop {
            if remaining == 0 {
                break;
            }
            let page_limit = remaining.min(MAX_SESSION_EVENTS_PAGE_LIMIT);
            let response = match client
                .request_with_envelope(
                    envelope.clone(),
                    RequestPayload::GetSessionEvents(GetSessionEventsRequest {
                        session_id,
                        before,
                        limit: page_limit,
                        kinds: kinds.clone(),
                    }),
                )
                .await?
            {
                ResponseResult::GetSessionEvents(resp) => resp,
                ResponseResult::Error(err) => return Err(err),
                _ => {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Internal,
                        "unexpected response for GetSessionEvents",
                    ));
                }
            };

            if response.events.is_empty() {
                before = response.next_cursor;
                break;
            }

            let fetched = response.events.len() as u32;
            remaining = remaining.saturating_sub(fetched);
            before = response.next_cursor;
            pages.push(response.events);

            if before.is_none() {
                break;
            }
        }

        let has_more = remaining == 0 && before.is_some();
        let mut events = Vec::new();
        for mut page in pages.into_iter().rev() {
            events.append(&mut page);
        }

        Ok(TaskHistoryResult {
            task_slug: resolved.task_slug,
            session_id: session_id.to_string(),
            events,
            has_more,
        })
    }
}

fn session_event_kind_label(kind: &SessionEventKind) -> &'static str {
    match kind {
        SessionEventKind::SessionStarted(_) => "session_started",
        SessionEventKind::SessionEnded(_) => "session_ended",
        SessionEventKind::TurnStarted(_) => "turn_started",
        SessionEventKind::TurnCompleted(_) => "turn_completed",
        SessionEventKind::UserMessage(_) => "user_message",
        SessionEventKind::AssistantMessage(_) => "assistant_message",
        SessionEventKind::AssistantReasoning(_) => "assistant_reasoning",
        SessionEventKind::ToolInvocation(_) => "tool_invocation",
        SessionEventKind::ToolResult(_) => "tool_result",
        SessionEventKind::StatusUpdate(_) => "status_update",
        SessionEventKind::TaskAgentMessageSent(_) => "task_agent_message_sent",
        SessionEventKind::PermissionsModeChanged(_) => "permissions_mode_changed",
        SessionEventKind::CodexApprovalPolicyChanged(_) => "codex_approval_policy_changed",
        SessionEventKind::CodexSandboxPolicyChanged(_) => "codex_sandbox_policy_changed",
        SessionEventKind::SessionModelChanged(_) => "session_model_changed",
        SessionEventKind::PermissionRequested(_) => "permission_requested",
        SessionEventKind::PermissionDecided(_) => "permission_decided",
        SessionEventKind::ArtifactEmitted(_) => "artifact_emitted",
        SessionEventKind::Unknown(_) => "unknown",
    }
}

fn transcript_role_for_event(event: &SessionEvent) -> Option<String> {
    match event.kind {
        SessionEventKind::UserMessage(_) => Some("user".to_string()),
        SessionEventKind::AssistantMessage(_) => Some("assistant".to_string()),
        SessionEventKind::StatusUpdate(_) => Some("status".to_string()),
        _ => None,
    }
}

fn transcript_text_for_event(event: &SessionEvent) -> Option<String> {
    match &event.kind {
        SessionEventKind::UserMessage(data) => Some(compact_text(&data.text, 240)),
        SessionEventKind::AssistantMessage(data) => Some(compact_text(&data.text, 240)),
        SessionEventKind::StatusUpdate(data) => Some(compact_status_update(data)),
        _ => None,
    }
}

fn render_history_line(event: &SessionEvent) -> Option<String> {
    let ts = format_timestamp(event.created_at);
    match &event.kind {
        SessionEventKind::UserMessage(data) => {
            Some(format!("[{ts}] user: {}", compact_text(&data.text, 240)))
        }
        SessionEventKind::AssistantMessage(data) => Some(format!(
            "[{ts}] assistant: {}",
            compact_text(&data.text, 240)
        )),
        SessionEventKind::StatusUpdate(data) => {
            Some(format!("[{ts}] status: {}", compact_status_update(data)))
        }
        SessionEventKind::TaskAgentMessageSent(data) => Some(format!(
            "[{ts}] task_agent_message_sent: intent={} {}",
            data.intent,
            compact_text(&data.message_preview, 200)
        )),
        SessionEventKind::AssistantReasoning(data) => Some(format!(
            "[{ts}] assistant_reasoning: {}",
            compact_text(&data.summary.preview, 240)
        )),
        SessionEventKind::ToolInvocation(data) => Some(format!(
            "[{ts}] tool_invocation:{} {}",
            data.tool_name,
            compact_text(&data.input_preview, 200)
        )),
        SessionEventKind::ToolResult(data) => Some(format!(
            "[{ts}] tool_result:{} {}",
            data.tool_name,
            compact_text(&data.output_preview, 200)
        )),
        SessionEventKind::SessionStarted(_) => Some(format!("[{ts}] session_started")),
        SessionEventKind::SessionEnded(_) => Some(format!("[{ts}] session_ended")),
        SessionEventKind::TurnStarted(_) => Some(format!("[{ts}] turn_started")),
        SessionEventKind::TurnCompleted(data) => {
            let mut line = String::from("turn_completed");
            if let Some(code) = data.exit_code {
                line.push_str(&format!(" exit_code={code}"));
            }
            if let Some(error) = &data.error {
                line.push_str(&format!(" error={}", compact_text(&error.message, 160)));
            }
            Some(format!("[{ts}] {line}"))
        }
        SessionEventKind::PermissionsModeChanged(data) => {
            Some(format!("[{ts}] permissions_mode_changed: {:?}", data.mode))
        }
        SessionEventKind::CodexApprovalPolicyChanged(data) => Some(format!(
            "[{ts}] codex_approval_policy_changed: {:?}",
            data.approval_policy
        )),
        SessionEventKind::CodexSandboxPolicyChanged(data) => Some(format!(
            "[{ts}] codex_sandbox_policy_changed: {:?}",
            data.sandbox_policy
        )),
        SessionEventKind::SessionModelChanged(data) => Some(format!(
            "[{ts}] session_model_changed: model_id={:?} reasoning_effort={:?}",
            data.model_id, data.reasoning_effort
        )),
        SessionEventKind::PermissionRequested(data) => Some(format!(
            "[{ts}] permission_requested: {}",
            compact_text(&data.summary, 200)
        )),
        SessionEventKind::PermissionDecided(data) => Some(format!(
            "[{ts}] permission_decided: request_id={} decision={:?}",
            data.request_id, data.decision
        )),
        SessionEventKind::ArtifactEmitted(data) => Some(format!(
            "[{ts}] artifact_emitted: {}",
            data.label.clone().unwrap_or_else(|| "artifact".to_string())
        )),
        SessionEventKind::Unknown(data) => Some(format!(
            "[{ts}] unknown_event: {}",
            compact_text(&data.event_type, 120)
        )),
    }
}

fn compact_status_update(data: &redesmyn_protocol::StatusUpdate) -> String {
    let mut parts = vec![turn_state_label(data.turn_state).to_string()];
    if let Some(progress) = data.progress_percent {
        parts.push(format!("{progress}%"));
    }
    if data.blocking == Some(true) {
        parts.push("blocking".to_string());
    }
    if let Some(message) = &data.message {
        parts.push(compact_text(message, 200));
    }
    parts.join(" | ")
}

fn turn_state_label(state: TurnState) -> &'static str {
    match state {
        TurnState::Running => "running",
        TurnState::Blocked => "blocked",
        TurnState::Completed => "completed",
        TurnState::Unknown => "unknown",
    }
}

fn compact_text(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut count = 0_usize;
    let mut last_was_space = false;
    let mut truncated = false;
    for ch in value.trim().chars() {
        if ch.is_whitespace() {
            if !last_was_space && !out.is_empty() {
                out.push(' ');
                count += 1;
                last_was_space = true;
            }
            continue;
        }
        if count >= max_chars {
            truncated = true;
            break;
        }

        out.push(ch);
        count += 1;
        last_was_space = false;
    }

    if truncated {
        out.push_str("...");
    }

    out
}

fn format_timestamp(value: redesmyn_protocol::Timestamp) -> String {
    serde_json::to_string(&value)
        .map(|value| value.trim_matches('"').to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn task_agent_message_delivery_label(value: TaskAgentMessageDelivery) -> &'static str {
    match value {
        TaskAgentMessageDelivery::StructuredStarted => "structured_started",
        TaskAgentMessageDelivery::StructuredResumed => "structured_resumed",
        TaskAgentMessageDelivery::InteractiveStarted => "interactive_started",
        TaskAgentMessageDelivery::InteractiveSent => "interactive_sent",
    }
}

fn task_agent_message_conversation_continuity_label(
    value: TaskAgentMessageConversationContinuity,
) -> &'static str {
    match value {
        TaskAgentMessageConversationContinuity::Kept => "kept",
        TaskAgentMessageConversationContinuity::Broken => "broken",
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
    initial_prompt: Option<String>,
    session_model_selection: Option<SessionModelSelection>,
    codex_approval_policy: Option<CodexApprovalPolicy>,
    codex_sandbox_policy: Option<CodexSandboxPolicy>,
    timeout_ms: u64,
) -> Result<StartAgentCommandResult, ErrorEnvelope> {
    #[cfg(not(unix))]
    {
        let _ = (
            epic_slug,
            task_slug,
            initial_prompt,
            session_model_selection,
            codex_approval_policy,
            codex_sandbox_policy,
            timeout_ms,
        );
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
                let mut known: Vec<String> = graph
                    .nodes
                    .iter()
                    .map(|node| node.task_slug.clone())
                    .collect();
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
                    initial_prompt,
                    on_conflict: AgentMessageConflictAction::Fail,
                    session_model_selection,
                    codex_approval_policy,
                    codex_sandbox_policy,
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

#[derive(Debug, Clone)]
struct TaskStartDefaults {
    send_prelude: bool,
    prelude: Option<String>,
    model_id: Option<String>,
    reasoning_effort: Option<ModelReasoningEffort>,
    codex_approval_policy: Option<CodexApprovalPolicy>,
    codex_sandbox_policy: Option<CodexSandboxPolicy>,
}

impl Default for TaskStartDefaults {
    fn default() -> Self {
        Self {
            send_prelude: true,
            prelude: None,
            model_id: None,
            reasoning_effort: None,
            codex_approval_policy: None,
            codex_sandbox_policy: None,
        }
    }
}

fn load_effective_task_start_defaults(repo_root: &Path) -> TaskStartDefaults {
    let mut defaults = TaskStartDefaults::default();

    if let Some(global_path) = global_config_path() {
        if global_path.exists() {
            apply_task_start_defaults_from_path(&mut defaults, &global_path);
        }
    }

    let repo_path = repo_config_path(repo_root);
    if repo_path.exists() {
        apply_task_start_defaults_from_path(&mut defaults, &repo_path);
    }

    defaults
}

fn apply_task_start_defaults_from_path(defaults: &mut TaskStartDefaults, path: &Path) {
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => {
            redesmyn_logging::tracing::warn!(
                error = %err,
                path = %path.display(),
                "unable to read config defaults"
            );
            return;
        }
    };

    let doc = match content.parse::<DocumentMut>() {
        Ok(doc) => doc,
        Err(err) => {
            redesmyn_logging::tracing::warn!(
                error = %err,
                path = %path.display(),
                "unable to parse config defaults"
            );
            return;
        }
    };

    if let Some(harness) = doc.get("harness").and_then(|item| item.as_table()) {
        if let Some(send_prelude) = harness.get("send_prelude").and_then(|item| item.as_bool()) {
            defaults.send_prelude = send_prelude;
        }

        if harness.contains_key("prelude") {
            defaults.prelude = harness
                .get("prelude")
                .and_then(|item| item.as_str())
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
        }
    }

    if let Some(session_defaults) = doc.get("session_defaults").and_then(|item| item.as_table()) {
        if let Some(codex) = session_defaults
            .get("codex")
            .and_then(|item| item.as_table())
        {
            if codex.contains_key("model") {
                defaults.model_id = codex
                    .get("model")
                    .and_then(|item| item.as_str())
                    .and_then(parse_model_id_override);
            }

            if let Some(value) = codex.get("reasoning_effort").and_then(|item| item.as_str()) {
                if let Some(override_value) = parse_reasoning_effort_override(value) {
                    defaults.reasoning_effort = override_value;
                }
            }

            if let Some(value) = codex.get("approval_policy").and_then(|item| item.as_str()) {
                if let Some(override_value) = parse_codex_approval_policy_override(value) {
                    defaults.codex_approval_policy = override_value;
                }
            }

            if let Some(value) = codex.get("sandbox_policy").and_then(|item| item.as_str()) {
                if let Some(override_value) = parse_codex_sandbox_policy_override(value) {
                    defaults.codex_sandbox_policy = override_value;
                }
            }
        }
    }
}

fn parse_model_id_override(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("default") {
        return None;
    }
    Some(trimmed.to_string())
}

fn parse_reasoning_effort_override(value: &str) -> Option<Option<ModelReasoningEffort>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "default" => Some(None),
        "minimal" => Some(Some(ModelReasoningEffort::Minimal)),
        "low" => Some(Some(ModelReasoningEffort::Low)),
        "medium" => Some(Some(ModelReasoningEffort::Medium)),
        "high" => Some(Some(ModelReasoningEffort::High)),
        "xhigh" | "x_high" | "x-high" => Some(Some(ModelReasoningEffort::Xhigh)),
        _ => None,
    }
}

fn parse_codex_approval_policy_override(value: &str) -> Option<Option<CodexApprovalPolicy>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "default" => Some(None),
        "untrusted" => Some(Some(CodexApprovalPolicy::UnlessTrusted)),
        "on_failure" | "on-failure" => Some(Some(CodexApprovalPolicy::OnFailure)),
        "on_request" | "on-request" => Some(Some(CodexApprovalPolicy::OnRequest)),
        "never" => Some(Some(CodexApprovalPolicy::Never)),
        _ => None,
    }
}

fn parse_codex_sandbox_policy_override(value: &str) -> Option<Option<CodexSandboxPolicy>> {
    match value.trim().to_ascii_lowercase().as_str() {
        "default" => Some(None),
        "read_only" | "read-only" => Some(Some(CodexSandboxPolicy::ReadOnly)),
        "workspace_write" | "workspace-write" => Some(Some(CodexSandboxPolicy::WorkspaceWrite {
            writable_roots: Vec::new(),
            network_access: false,
            exclude_tmpdir_env_var: false,
            exclude_slash_tmp: false,
        })),
        "danger_full_access" | "danger-full-access" => {
            Some(Some(CodexSandboxPolicy::DangerFullAccess))
        }
        _ => None,
    }
}

fn merge_task_start_prompts(prelude: Option<String>, prompt: Option<String>) -> Option<String> {
    match (prelude, prompt) {
        (None, None) => None,
        (Some(value), None) | (None, Some(value)) => Some(value),
        (Some(prelude), Some(prompt)) => Some(format!("{prelude}\n\n{prompt}")),
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
