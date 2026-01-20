use std::sync::{Arc, Mutex};

use clap::{Parser, Subcommand};
use redesmyn_logging::tracing;
use redesmyn_protocol::daemon::{ControlPlaneHelloAck, DaemonFrame, DaemonMessage};
use redesmyn_protocol::{ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::DaemonConnection;
use redesmyn_transport::TransportError;
use redesmyn_transport::in_proc::InProcEndpoint;

#[derive(Debug, Parser)]
#[command(name = "redesmyn-daemon")]
#[command(about = "Redesmyn daemon (Rust port).")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Runs the daemon against an in-proc stub control plane (dev/demo).
    DemoInProc {
        /// In-proc transport channel buffer size.
        #[arg(long, default_value_t = 8)]
        buffer: usize,
    },
}

fn main() {
    redesmyn_logging::init();

    let span = redesmyn_logging::redesmyn_info_span!("startup", component = "redesmyn-daemon");
    redesmyn_logging::span::record_run_id(&span, std::process::id());

    let _guard = span.enter();
    tracing::info!("starting");

    let cli = Cli::parse();
    match cli.command {
        Command::DemoInProc { buffer } => run_demo_in_proc(buffer),
    }
}

fn run_demo_in_proc(buffer: usize) {
    let config = match redesmyn_config::load_rust_config(Default::default()) {
        Ok(config) => config,
        Err(err) => {
            tracing::error!(error = %err, "failed to load config");
            std::process::exit(2);
        }
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async move {
        let (control_plane, daemon) = InProcEndpoint::pair(buffer);
        let control_plane = tokio::spawn(async move {
            if let Err(err) = run_stub_control_plane(control_plane).await {
                tracing::warn!(error = %err, "stub control plane terminated with error");
            }
        });

        let connector_endpoint: Arc<Mutex<Option<InProcEndpoint>>> = Arc::new(Mutex::new(Some(daemon)));
        let connector: Arc<dyn redesmyn_daemon::ControlPlaneConnector> = Arc::new(move || {
            let endpoint = Arc::clone(&connector_endpoint);
            async move {
                let mut guard = endpoint.lock().expect("lock poisoned");
                let endpoint = guard.take().ok_or(TransportError::ChannelClosed)?;
                Ok(Box::new(endpoint) as Box<dyn DaemonConnection>)
            }
        });

        let daemon_config = redesmyn_daemon::DaemonRuntimeConfig::new(config.daemon);
        let daemon = redesmyn_daemon::Daemon::start(daemon_config, connector);

        let state_task = tokio::spawn(log_connection_state(daemon.connection_state()));

        tracing::info!("demo running; press Ctrl-C to exit");
        let _ = tokio::signal::ctrl_c().await;

        daemon.shutdown().await;
        state_task.abort();
        control_plane.abort();
    });
}

async fn run_stub_control_plane(mut conn: InProcEndpoint) -> Result<(), TransportError> {
    let frame = conn.recv().await?;
    let peer_version = frame.envelope.protocol_version();

    let DaemonMessage::DaemonHello(_) = frame.message else {
        return Err(TransportError::ChannelClosed);
    };

    let accepted = ProtocolVersion::CURRENT
        .negotiate(peer_version)
        .map_err(|_| TransportError::ChannelClosed)?;

    let mut envelope = ProtocolEnvelope::new();
    envelope.protocol_major = accepted.major;
    envelope.protocol_minor = accepted.minor;
    envelope.correlation_id = Some(frame.envelope.msg_id);

    conn.send(DaemonFrame::new(
        envelope,
        DaemonMessage::ControlPlaneHelloAck(ControlPlaneHelloAck {
            accepted_protocol: accepted,
            capabilities: vec!["stub".to_string()],
        }),
    ))
    .await?;

    loop {
        let frame = conn.recv().await;
        match frame {
            Ok(_frame) => {}
            Err(TransportError::ChannelClosed) => return Ok(()),
            Err(err) => return Err(err),
        }
    }
}

async fn log_connection_state(mut rx: tokio::sync::watch::Receiver<redesmyn_daemon::ConnectionState>) {
    tracing::info!(state = ?*rx.borrow(), "connection state");
    while rx.changed().await.is_ok() {
        tracing::info!(state = ?*rx.borrow(), "connection state");
    }
}
