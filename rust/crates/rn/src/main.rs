use std::{
    env,
    ffi::OsString,
    io::{self, Write as _},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::{Duration, Instant},
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope};
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

    let output = Output::new(cli.global.output);

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
        Commands::Protocol(cmd) => protocol::protocol(cmd, output),
        Commands::UiDriver(cmd) => ui_driver::ui_driver(cmd, output),
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

    /// Protocol tooling.
    #[command(subcommand)]
    Protocol(protocol::ProtocolCommands),

    /// Desktop UI automation driver tools.
    #[command(subcommand)]
    UiDriver(ui_driver::UiDriverCommands),

    /// Prints version information.
    Version(VersionArgs),

    /// Simple benchmarking helpers for startup regression tracking.
    #[command(subcommand)]
    Bench(BenchCommands),
}

#[derive(Debug, Args)]
struct DoctorArgs {}

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

fn find_git_root(start: &Path) -> Option<PathBuf> {
    for dir in start.ancestors() {
        if dir.join(".git").exists() {
            return Some(dir.to_path_buf());
        }
    }
    None
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
