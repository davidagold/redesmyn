use std::collections::HashMap;
use std::path::{Path, PathBuf};

use redesmyn_logging::tracing;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, Event, EventLogEvent, GetEpicGraphResponse, HealthResponse,
    ListEpicsResponse, Response, ResponseResult, StatusResponse, Subscribed, SubscriptionEvent,
    SubscriptionTopic,
};
use redesmyn_protocol::{ErrorEnvelope, ProtocolEnvelope, ProtocolVersion, Timestamp};
use redesmyn_transport::client::codec::{Codec, JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::client::{ClientConnection, ClientTransportError};

use crate::error::ControlPlaneError;
use crate::event_log::EventLog as ServerEventLog;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientApiCodec {
    Json,
    Protobuf,
}

#[derive(Debug, Clone)]
pub struct ClientApiContext {
    pub event_log: ServerEventLog,
}

#[derive(Debug, thiserror::Error)]
pub enum ClientApiServeError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Transport(#[from] ClientTransportError),
}

#[cfg(not(unix))]
pub fn bind_client_socket(_path: &Path) -> Result<(), ClientApiServeError> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unix domain sockets are not supported on this platform",
    )
    .into())
}

#[cfg(unix)]
pub fn bind_client_socket(path: &Path) -> Result<tokio::net::UnixListener, ClientApiServeError> {
    use std::os::unix::fs::PermissionsExt as _;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if path.exists() {
        std::fs::remove_file(path)?;
    }

    let listener = tokio::net::UnixListener::bind(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}

#[cfg(unix)]
pub async fn serve_client_api_uds(
    socket_path: PathBuf,
    codec: ClientApiCodec,
    ctx: ClientApiContext,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError> {
    let listener = bind_client_socket(&socket_path)?;

    serve_client_api_listener(listener, socket_path, codec, ctx, shutdown).await
}

#[cfg(unix)]
pub async fn serve_client_api_listener(
    listener: tokio::net::UnixListener,
    socket_path: PathBuf,
    codec: ClientApiCodec,
    ctx: ClientApiContext,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError> {
    tracing::info!(
        socket_path = %socket_path.display(),
        codec = ?codec,
        "client API UDS server listening"
    );

    let mut connections = tokio::task::JoinSet::new();

    loop {
        tokio::select! {
            _ = shutdown.recv() => break,
            accept = listener.accept() => {
                let (stream, _addr) = accept?;
                let task_codec = codec;
                let task_ctx = ctx.clone();
                let mut task_shutdown = shutdown.resubscribe();

                connections.spawn(async move {
                    let result = match task_codec {
                        ClientApiCodec::Json => handle_connection(stream, JsonCodec::new(), task_ctx, &mut task_shutdown).await,
                        ClientApiCodec::Protobuf => handle_connection(stream, ProtobufCodec::new(), task_ctx, &mut task_shutdown).await,
                    };

                    if let Err(err) = result {
                        tracing::warn!(error = %err, "client API connection terminated with error");
                    }
                });
            }
            Some(join_result) = connections.join_next() => {
                if let Err(err) = join_result {
                    tracing::error!(error = %err, "client API connection task panicked");
                }
            }
        }
    }

    while let Some(join_result) = connections.join_next().await {
        if let Err(err) = join_result {
            tracing::error!(error = %err, "client API connection task panicked");
        }
    }

    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    tracing::info!("client API UDS server shut down");
    Ok(())
}

async fn handle_connection<C>(
    stream: tokio::net::UnixStream,
    codec: C,
    ctx: ClientApiContext,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError>
where
    C: Codec + 'static,
{
    let peer = stream.peer_addr().ok();
    let span = tracing::info_span!(
        "client.api.connection",
        peer = peer
            .as_ref()
            .map(|addr| addr.as_pathname().map(|p| p.display().to_string()))
            .flatten()
            .unwrap_or_else(|| "<unknown>".to_string())
    );
    let _enter = span.enter();

    let mut conn = FramedEndpoint::new(stream, codec);
    serve_connection(&mut conn, &ctx, shutdown).await?;
    Ok(())
}

/// Serve a single client ↔ control plane connection (T-12).
///
/// This is the shared per-connection implementation used by:
/// - the UDS server (`serve_client_api_*`), and
/// - embedded/in-proc callers (`ControlPlaneHandle::connect_in_proc_client`).
pub async fn serve_connection<C>(
    conn: &mut C,
    ctx: &ClientApiContext,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError>
where
    C: ClientConnection + Send + 'static,
{
    run_session(conn, ctx, shutdown).await
}

async fn run_session<C>(
    conn: &mut C,
    ctx: &ClientApiContext,
    shutdown: &mut tokio::sync::broadcast::Receiver<()>,
) -> Result<(), ClientApiServeError>
where
    C: ClientConnection + Send + 'static,
{
    let mut accepted_protocol: Option<ProtocolVersion> = None;
    let mut subscriptions: HashMap<redesmyn_ids::SubscriptionId, SubscriptionTopic> =
        HashMap::new();
    let mut event_log_rx = ctx.event_log.subscribe();

    loop {
        tokio::select! {
            _ = shutdown.recv() => return Ok(()),
            frame = conn.recv() => {
                let frame = match frame {
                    Ok(frame) => frame,
                    Err(ClientTransportError::ChannelClosed) => return Ok(()),
                    Err(err) => return Err(err.into()),
                };

                let peer_version = frame.envelope.protocol_version();
                let accepted = match accepted_protocol {
                    Some(version) => version,
                    None => match ProtocolVersion::CURRENT.negotiate(peer_version) {
                        Ok(version) => {
                            accepted_protocol = Some(version);
                            version
                        }
                        Err(err) => {
                            if let ClientMessage::Request(req) = frame.message {
                                let response = Response {
                                    request_id: req.request_id,
                                    result: ResponseResult::Error(err),
                                };

                                let response_frame = ClientFrame::new(
                                    error_response_envelope(&frame.envelope),
                                    ClientMessage::Response(response),
                                );

                                let _ = conn.send(response_frame).await;
                            }
                            return Ok(());
                        }
                    },
                };

                match frame.message {
                    ClientMessage::Request(req) => {
                        let response = handle_request(req, accepted).await;
                        let response_frame = ClientFrame::new(
                            response_envelope(&frame.envelope, accepted),
                            ClientMessage::Response(response),
                        );
                        conn.send(response_frame).await?;
                    }
                    ClientMessage::Subscribe(sub) => {
                        let subscription_id = sub.subscription_id;
                        subscriptions.insert(subscription_id, sub.topic());

                        let ack = ClientFrame::new(
                            response_envelope(&frame.envelope, accepted),
                            ClientMessage::Event(Event {
                                subscription_id,
                                event: SubscriptionEvent::Subscribed(Subscribed { topic: sub.topic() }),
                            }),
                        );
                        conn.send(ack).await?;

                        if sub.topic() == SubscriptionTopic::EventLog {
                            let event = ClientFrame::new(
                                response_envelope(&frame.envelope, accepted),
                                ClientMessage::Event(Event {
                                    subscription_id,
                                    event: SubscriptionEvent::EventLog(EventLogEvent {
                                        event_id: redesmyn_ids::EventId::new(),
                                        occurred_at: Timestamp::now_utc(),
                                        event_type: "event_log.appended".to_string(),
                                        json_payload: Vec::new(),
                                    }),
                                }),
                            );
                            conn.send(event).await?;
                        }
                    }
                    ClientMessage::Unsubscribe(unsub) => {
                        subscriptions.remove(&unsub.subscription_id);
                    }
                    ClientMessage::Response(_) | ClientMessage::Event(_) => {
                        let err: ErrorEnvelope = ControlPlaneError::InvalidTaskId {
                            task_id: "client.sent_server_message".to_string(),
                        }
                        .into();

                        tracing::debug!(error = %err.message, "ignoring unexpected client message");
                    }
                }
            }
            event = event_log_rx.recv(), if subscriptions.values().any(|topic| *topic == SubscriptionTopic::EventLog) => {
                let event = match event {
                    Ok(event) => event,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(()),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "event log subscriber lagged; dropping events");
                        continue;
                    }
                };

                let accepted = accepted_protocol.unwrap_or(ProtocolVersion::CURRENT);

                for (subscription_id, topic) in &subscriptions {
                    if *topic != SubscriptionTopic::EventLog {
                        continue;
                    }

                    let mut envelope = ProtocolEnvelope::new();
                    envelope.protocol_major = accepted.major;
                    envelope.protocol_minor = accepted.minor;

                    conn.send(ClientFrame::new(
                        envelope,
                        ClientMessage::Event(Event {
                            subscription_id: *subscription_id,
                            event: SubscriptionEvent::EventLog(event.clone()),
                        }),
                    ))
                    .await?;
                }
            }
        }
    }
}

async fn handle_request(
    req: redesmyn_protocol::client::Request,
    accepted: ProtocolVersion,
) -> Response {
    let request_id = req.request_id;
    let span = tracing::debug_span!(
        "client.api.request",
        request_id = %request_id,
        method = ?req.method(),
    );
    let _enter = span.enter();

    let result = match req.payload {
        redesmyn_protocol::client::RequestPayload::Health(_) => {
            ResponseResult::Health(HealthResponse { ok: true })
        }
        redesmyn_protocol::client::RequestPayload::Status(_) => {
            ResponseResult::Status(StatusResponse {
                accepted_protocol: accepted,
                server_name: "redesmyn-control-plane".to_string(),
                server_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            })
        }
        redesmyn_protocol::client::RequestPayload::ListEpics(_) => {
            ResponseResult::ListEpics(ListEpicsResponse { epics: Vec::new() })
        }
        redesmyn_protocol::client::RequestPayload::GetEpicGraph(get) => {
            ResponseResult::GetEpicGraph(GetEpicGraphResponse {
                graph: redesmyn_protocol::client::EpicGraph {
                    epic_slug: get.epic_slug,
                    nodes: Vec::new(),
                    edges: Vec::new(),
                },
            })
        }
    };

    Response { request_id, result }
}

fn response_envelope(
    request_envelope: &ProtocolEnvelope,
    accepted: ProtocolVersion,
) -> ProtocolEnvelope {
    let mut envelope = ProtocolEnvelope::new();
    envelope.protocol_major = accepted.major;
    envelope.protocol_minor = accepted.minor;
    envelope.scope = request_envelope.scope;
    envelope.trace_id = request_envelope.trace_id;
    envelope.correlation_id = Some(request_envelope.msg_id);
    envelope
}

fn error_response_envelope(request_envelope: &ProtocolEnvelope) -> ProtocolEnvelope {
    let mut envelope = ProtocolEnvelope::new();
    envelope.scope = request_envelope.scope;
    envelope.trace_id = request_envelope.trace_id;
    envelope.correlation_id = Some(request_envelope.msg_id);
    envelope
}
