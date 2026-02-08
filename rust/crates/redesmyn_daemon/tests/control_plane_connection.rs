use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use redesmyn_config::{
    DaemonConfig, ExecutorConfig, SandboxConfig, SandboxNetworkMode, SandboxType,
};
use redesmyn_daemon::{ConnectionState, Daemon, DaemonRuntimeConfig, HostIdentity};
use redesmyn_protocol::ErrorEnvelope;
use redesmyn_protocol::daemon::{ControlPlaneHelloAck, DaemonFrame, DaemonMessage};
use redesmyn_protocol::{ErrorCategory, ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::in_proc::InProcEndpoint;
use redesmyn_transport::{DaemonConnection, TransportError};

#[derive(Debug, Clone)]
enum StubPlan {
    Ack {
        accepted_protocol: ProtocolVersion,
        close_after_handshake: bool,
    },
    Reject(ErrorEnvelope),
}

fn spawn_stub_control_plane(
    mut conn: InProcEndpoint,
    plan: StubPlan,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let frame = conn.recv().await.expect("recv hello");
        let DaemonMessage::DaemonHello(_hello) = frame.message else {
            panic!("expected DaemonHello");
        };

        match plan {
            StubPlan::Ack {
                accepted_protocol,
                close_after_handshake,
            } => {
                let mut envelope = ProtocolEnvelope::new();
                envelope.protocol_major = accepted_protocol.major;
                envelope.protocol_minor = accepted_protocol.minor;
                envelope.correlation_id = Some(frame.envelope.msg_id);

                conn.send(DaemonFrame::new(
                    envelope,
                    DaemonMessage::ControlPlaneHelloAck(ControlPlaneHelloAck {
                        accepted_protocol,
                        capabilities: Vec::new(),
                    }),
                ))
                .await
                .expect("send ack");

                if close_after_handshake {
                    return;
                }

                loop {
                    match conn.recv().await {
                        Ok(_frame) => {}
                        Err(TransportError::ChannelClosed) => return,
                        Err(err) => panic!("stub recv error: {err}"),
                    }
                }
            }
            StubPlan::Reject(err) => {
                conn.send(DaemonFrame::new(
                    ProtocolEnvelope::new(),
                    DaemonMessage::Error(err),
                ))
                .await
                .expect("send error");
            }
        }
    })
}

fn connector_from_plans(
    plans: Arc<Mutex<VecDeque<StubPlan>>>,
) -> Arc<dyn redesmyn_daemon::ControlPlaneConnector> {
    Arc::new(move || {
        let plans = Arc::clone(&plans);
        async move {
            let (control_plane, daemon) = InProcEndpoint::pair(8);
            let plan = plans
                .lock()
                .expect("lock poisoned")
                .pop_front()
                .expect("stub plan missing");
            let _task = spawn_stub_control_plane(control_plane, plan);
            Ok(Box::new(daemon) as Box<dyn DaemonConnection>)
        }
    })
}

fn test_daemon_config() -> (tempfile::TempDir, DaemonConfig) {
    let dir = tempfile::tempdir().expect("tempdir");
    let state_dir = dir.path().join("repos");

    let config = DaemonConfig {
        repo_registry_dir: state_dir,
        worktree_root: dir.path().to_path_buf(),
        executor: ExecutorConfig { max_concurrency: 1 },
        sandbox: SandboxConfig {
            kind: SandboxType::None,
            network: SandboxNetworkMode::Allow,
        },
    };

    (dir, config)
}

#[tokio::test]
async fn connection_manager_connects_and_handshakes() {
    let connector = connector_from_plans(Arc::new(Mutex::new(VecDeque::from([StubPlan::Ack {
        accepted_protocol: ProtocolVersion::CURRENT,
        close_after_handshake: false,
    }]))));

    let identity = HostIdentity::new(redesmyn_ids::HostId::new());
    let (_dir, daemon_config) = test_daemon_config();
    let mut config = DaemonRuntimeConfig::new(daemon_config).with_host_identity(identity);
    config.backoff.initial = Duration::from_millis(10);
    config.backoff.max = Duration::from_millis(10);

    let daemon = Daemon::start(config, connector);
    let mut state_rx = daemon.connection_state();

    while state_rx.changed().await.is_ok() {
        if let ConnectionState::Connected { accepted_protocol } = *state_rx.borrow() {
            assert_eq!(accepted_protocol, ProtocolVersion::CURRENT);
            break;
        }
    }

    daemon.shutdown().await;
}

#[tokio::test(start_paused = true)]
async fn reconnects_with_backoff_after_disconnect() {
    let plans = Arc::new(Mutex::new(VecDeque::from([
        StubPlan::Ack {
            accepted_protocol: ProtocolVersion::CURRENT,
            close_after_handshake: true,
        },
        StubPlan::Ack {
            accepted_protocol: ProtocolVersion::CURRENT,
            close_after_handshake: false,
        },
    ])));
    let connector = connector_from_plans(Arc::clone(&plans));

    let identity = HostIdentity::new(redesmyn_ids::HostId::new());
    let (_dir, daemon_config) = test_daemon_config();
    let mut config = DaemonRuntimeConfig::new(daemon_config).with_host_identity(identity);
    config.backoff.initial = Duration::from_secs(5);
    config.backoff.max = Duration::from_secs(5);

    let daemon = Daemon::start(config, connector);

    // Wait for the first connection attempt to consume the first plan. The first
    // session transitions quickly (connected → disconnected), so don't assert on
    // observing intermediate states via `watch`.
    loop {
        if plans.lock().expect("lock poisoned").len() == 1 {
            break;
        }
        tokio::task::yield_now().await;
    }

    tokio::time::advance(Duration::from_secs(4)).await;
    tokio::task::yield_now().await;
    assert_eq!(plans.lock().expect("lock poisoned").len(), 1);

    tokio::time::advance(Duration::from_secs(1)).await;
    tokio::task::yield_now().await;

    // Second connection attempt should only happen after 5s backoff.
    loop {
        if plans.lock().expect("lock poisoned").is_empty() {
            break;
        }
        tokio::task::yield_now().await;
    }

    let mut state_rx = daemon.connection_state();
    loop {
        if matches!(*state_rx.borrow(), ConnectionState::Connected { .. }) {
            break;
        }
        state_rx.changed().await.unwrap();
    }

    daemon.shutdown().await;
}

#[tokio::test]
async fn fatal_on_invalid_request_handshake_error() {
    let err = ErrorEnvelope::new(
        ErrorCategory::InvalidRequest,
        "Protocol major version mismatch.",
    );
    let connector = connector_from_plans(Arc::new(Mutex::new(VecDeque::from([StubPlan::Reject(
        err.clone(),
    )]))));

    let identity = HostIdentity::new(redesmyn_ids::HostId::new());
    let (_dir, daemon_config) = test_daemon_config();
    let mut config = DaemonRuntimeConfig::new(daemon_config).with_host_identity(identity);
    config.backoff.initial = Duration::from_millis(10);
    config.backoff.max = Duration::from_millis(10);

    let daemon = Daemon::start(config, connector);
    let mut state_rx = daemon.connection_state();

    loop {
        state_rx.changed().await.unwrap();
        if let ConnectionState::Fatal { error } = &*state_rx.borrow() {
            assert_eq!(error.category, ErrorCategory::InvalidRequest);
            break;
        }
    }

    daemon.shutdown().await;
}

#[tokio::test]
async fn minor_mismatch_is_accepted() {
    let connector = connector_from_plans(Arc::new(Mutex::new(VecDeque::from([StubPlan::Ack {
        accepted_protocol: ProtocolVersion::new(1, 0),
        close_after_handshake: false,
    }]))));

    let identity = HostIdentity::new(redesmyn_ids::HostId::new());
    let supported = ProtocolVersion::new(1, 7);
    let (_dir, daemon_config) = test_daemon_config();
    let mut config = DaemonRuntimeConfig::new(daemon_config).with_host_identity(identity);
    config.supported_protocol = supported;
    config.backoff.initial = Duration::from_millis(10);
    config.backoff.max = Duration::from_millis(10);

    let daemon = Daemon::start(config, connector);
    let mut state_rx = daemon.connection_state();

    loop {
        state_rx.changed().await.unwrap();
        if let ConnectionState::Connected { accepted_protocol } = *state_rx.borrow() {
            assert_eq!(accepted_protocol, ProtocolVersion::new(1, 0));
            break;
        }
    }

    daemon.shutdown().await;
}
