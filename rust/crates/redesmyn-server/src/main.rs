use clap::{Parser, Subcommand};
use redesmyn_protocol::ErrorEnvelope;

#[cfg(unix)]
use std::path::PathBuf;

#[cfg(unix)]
use clap::ValueEnum;

#[cfg(unix)]
use redesmyn_control_plane::client_api::{ClientApiCodec, ClientApiServeError};

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

    /// Serves the client API over a Unix domain socket (T-12).
    #[cfg(unix)]
    ServeClientApi {
        /// Codec to use for the API (Protobuf by default; JSON for debug/diagnostics).
        #[arg(long, value_enum, default_value_t = ClientApiCodecArg::Protobuf)]
        codec: ClientApiCodecArg,

        /// Override the configured socket path.
        #[arg(long)]
        socket_path: Option<PathBuf>,
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
        Command::ServeClientApi { codec, socket_path } => run_serve_client_api(codec, socket_path),
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
fn run_serve_client_api(codec: ClientApiCodecArg, socket_path: Option<PathBuf>) {
    let config = match redesmyn_config::load_rust_config(Default::default()) {
        Ok(config) => config,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to load config");
            std::process::exit(2);
        }
    };

    let socket_path = socket_path.unwrap_or_else(|| config.control_plane.api.client_socket_path);
    let codec = match codec {
        ClientApiCodecArg::Json => ClientApiCodec::Json,
        ClientApiCodecArg::Protobuf => ClientApiCodec::Protobuf,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let result: Result<(), ClientApiServeError> = runtime.block_on(
        redesmyn_control_plane::client_api::serve_client_api_uds(socket_path, codec),
    );

    if let Err(err) = result {
        redesmyn_logging::tracing::error!(error = %err, "client API server exited");
        std::process::exit(1);
    }
}
