use clap::{Parser, Subcommand};
use redesmyn_protocol::ErrorEnvelope;

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
}

fn main() {
    redesmyn_logging::init();

    let cli = Cli::parse();
    match cli.command {
        Command::DemoError { task_id } => run_demo_error(task_id),
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
