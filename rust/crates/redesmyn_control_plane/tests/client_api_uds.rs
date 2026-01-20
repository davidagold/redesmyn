#![cfg(unix)]

use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::ControlPlaneDb;
use redesmyn_control_plane::ControlPlaneStartOptions;
use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_ids::{RepoId, RequestId, SubscriptionId, WorkspaceId};
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, EventLogFilter, HealthRequest, Request, RequestPayload,
    ResponseResult, StatusRequest, Subscribe, SubscriptionEvent, SubscriptionFilter,
};
use redesmyn_protocol::{ProtocolEnvelope, RepoScope, TraceId};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::ProtobufCodec;
use redesmyn_transport::client::framed::FramedEndpoint;

#[tokio::test]
async fn uds_server_binds_securely_and_supports_requests_and_subscriptions() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let options = ControlPlaneStartOptions {
        db: ControlPlaneDb::InMemory,
        client_api_socket_path: Some(socket_path.clone()),
        client_api_codec: ClientApiCodec::Protobuf,
    };
    let server = ControlPlane::start(options).await.expect("start");

    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(socket_path.exists(), "socket file was not created");

    let mode = std::fs::metadata(&socket_path)
        .expect("socket metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "unexpected socket mode: {mode:o}");

    let stream = tokio::net::UnixStream::connect(&socket_path)
        .await
        .expect("connect");
    let mut conn = FramedEndpoint::new(stream, ProtobufCodec::new());

    let health_request_id = RequestId::new();
    let status_request_id = RequestId::new();

    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id: health_request_id,
            payload: RequestPayload::Health(HealthRequest {}),
        }),
    ))
    .await
    .expect("send health request");

    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id: status_request_id,
            payload: RequestPayload::Status(StatusRequest {}),
        }),
    ))
    .await
    .expect("send status request");

    let mut saw_health = false;
    let mut saw_status = false;
    for _ in 0..2 {
        let frame = conn.recv().await.expect("recv response");
        let ClientMessage::Response(resp) = frame.message else {
            panic!("expected Response, got {:?}", frame.message);
        };

        match (resp.request_id, resp.result) {
            (request_id, ResponseResult::Health(payload)) => {
                assert_eq!(request_id, health_request_id);
                assert!(payload.ok);
                saw_health = true;
            }
            (request_id, ResponseResult::Status(payload)) => {
                assert_eq!(request_id, status_request_id);
                assert_eq!(payload.server_name, "redesmyn-control-plane");
                saw_status = true;
            }
            (request_id, other) => panic!("unexpected response {request_id}: {other:?}"),
        }
    }
    assert!(saw_health && saw_status);

    let subscription_id = SubscriptionId::new();
    let trace_id = TraceId::from_bytes([0x11; TraceId::BYTE_LEN]);
    let scope = RepoScope::new(WorkspaceId::new(), RepoId::new());

    let subscribe_envelope = ProtocolEnvelope::new()
        .with_scope(scope.into())
        .with_trace_id(trace_id);
    let subscribe_msg_id = subscribe_envelope.msg_id;

    conn.send(ClientFrame::new(
        subscribe_envelope,
        ClientMessage::Subscribe(Subscribe {
            subscription_id,
            filter: SubscriptionFilter::EventLog(EventLogFilter {
                after_event_id: None,
            }),
        }),
    ))
    .await
    .expect("send subscribe");

    let mut got_subscribed = false;
    let mut got_tick = false;
    let expected_scope = Some(scope.into());
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    while tokio::time::Instant::now() < deadline && !(got_subscribed && got_tick) {
        let frame = match tokio::time::timeout(Duration::from_millis(200), conn.recv()).await {
            Ok(Ok(frame)) => frame,
            Ok(Err(err)) => panic!("recv error: {err}"),
            Err(_) => continue,
        };

        let ClientMessage::Event(event) = frame.message else {
            continue;
        };
        if event.subscription_id != subscription_id {
            continue;
        }

        match event.event {
            SubscriptionEvent::Subscribed(_) => {
                assert_eq!(frame.envelope.scope, expected_scope);
                assert_eq!(frame.envelope.trace_id, Some(trace_id));
                assert_eq!(frame.envelope.correlation_id, Some(subscribe_msg_id));
                got_subscribed = true;
            }
            SubscriptionEvent::EventLog(event_log) => {
                if event_log.event_type != "event_log.tick" {
                    continue;
                }
                assert_eq!(frame.envelope.scope, expected_scope);
                assert_eq!(frame.envelope.trace_id, Some(trace_id));
                assert_eq!(frame.envelope.correlation_id, Some(subscribe_msg_id));
                got_tick = true;
            }
            SubscriptionEvent::Error(err) => panic!("unexpected subscription error: {err:?}"),
        }
    }

    assert!(got_subscribed, "did not receive Subscribed event");
    assert!(got_tick, "did not receive tick EventLog event");

    server.shutdown().await;
    assert!(
        !socket_path.exists(),
        "socket file should be removed on shutdown"
    );
}
