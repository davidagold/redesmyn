use clap::{Parser, Subcommand};
use redesmyn_protocol::ErrorEnvelope;

#[cfg(unix)]
use std::path::PathBuf;

#[cfg(unix)]
use clap::ValueEnum;

#[cfg(unix)]
use redesmyn_control_plane::ControlPlaneDb;
#[cfg(unix)]
use redesmyn_control_plane::ControlPlaneStartOptions;
#[cfg(unix)]
use redesmyn_control_plane::client_api::ClientApiCodec;
#[cfg(unix)]
use redesmyn_control_plane::ControlPlane;

#[derive(Debug, Parser)]
#[command(name = "redesmyn-server")]
#[command(about = "Redesmyn control-plane server (Rust port).")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Tiny end-to-end example for the workspace error conventions (T-3).
    DemoError {
        /// Task id to look up (pass `T-404` to demo a NotFound error).
        task_id: Option<String>,
    },

    /// Runs the headless control plane service (T-16).
    #[cfg(unix)]
    Serve {
        /// Codec to use for the client API (Protobuf by default; JSON for debug/diagnostics).
        #[arg(long, value_enum, default_value_t = ClientApiCodecArg::Protobuf)]
        codec: ClientApiCodecArg,

        /// Override the configured socket path.
        #[arg(long)]
        socket_path: Option<PathBuf>,

        /// Override the configured DB path.
        #[arg(long)]
        db_path: Option<PathBuf>,
    },
}

#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab_case")]
enum ClientApiCodecArg {
    Json,
    Protobuf,
}

fn main() {
    redesmyn_logging::init();

    let span = redesmyn_logging::redesmyn_info_span!("startup", component = "redesmyn-server");
    redesmyn_logging::span::record_run_id(&span, std::process::id());

    let _guard = span.enter();
    redesmyn_logging::tracing::info!("starting");

    let cli = Cli::parse();
    match cli.command {
        Command::DemoError { task_id } => run_demo_error(task_id),
        #[cfg(unix)]
        Command::Serve {
            codec,
            socket_path,
            db_path,
        } => run_serve(codec, socket_path, db_path),
    };
}

fn run_demo_error(task_id: Option<String>) {
    let task_id = task_id.unwrap_or_default();
    match redesmyn_control_plane::demo::get_task(&task_id) {
        Ok(task) => {
            println!("ok: {}", task.task_id);
        }
        Err(err) => {
            let envelope: ErrorEnvelope = err.into();
            eprintln!("error[{}]: {}", envelope.category, envelope.message);
            std::process::exit(envelope.exit_code());
        }
    }
}

#[cfg(unix)]
fn run_serve(codec: ClientApiCodecArg, socket_path: Option<PathBuf>, db_path: Option<PathBuf>) {
    let config = match redesmyn_config::load_rust_config(Default::default()) {
        Ok(config) => config,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to load config");
            std::process::exit(2);
        }
    };

    let socket_path = socket_path.unwrap_or_else(|| config.control_plane.api.client_socket_path);
    let db_path = db_path.unwrap_or_else(|| config.control_plane.db.path);
    let codec = match codec {
        ClientApiCodecArg::Json => ClientApiCodec::Json,
        ClientApiCodecArg::Protobuf => ClientApiCodec::Protobuf,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let result: Result<(), redesmyn_control_plane::ControlPlaneStartError> = runtime.block_on(async move {
        let handle = ControlPlane::start(ControlPlaneStartOptions {
            db: ControlPlaneDb::Path(db_path),
            client_api_socket_path: Some(socket_path),
            client_api_codec: codec,
        })
        .await?;

        wait_for_shutdown_signal().await;
        handle.shutdown().await;
        Ok(())
    });

    if let Err(err) = result {
        redesmyn_logging::tracing::error!(error = %err, "control plane service exited");
        std::process::exit(1);
    }
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = sigterm.recv() => {}
    }
}
