#![cfg(unix)]

use std::os::unix::fs::PermissionsExt as _;
use std::time::Duration;

use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{RequestId, SubscriptionId};
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, EventLogFilter, HealthRequest, Request, RequestPayload,
    ResponseResult, StatusRequest, Subscribe, SubscriptionEvent, SubscriptionFilter,
};
use redesmyn_storage::events::EventScope;
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::ProtobufCodec;
use redesmyn_transport::client::framed::FramedEndpoint;

async fn connect_with_retry(path: &std::path::Path) -> tokio::net::UnixStream {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match tokio::net::UnixStream::connect(path).await {
                Ok(stream) => return stream,
                Err(_) => tokio::task::yield_now().await,
            }
        }
    })
    .await
    .expect("connect timeout")
}

async fn recv_frame(
    conn: &mut FramedEndpoint<ProtobufCodec, tokio::net::UnixStream>,
) -> ClientFrame {
    tokio::time::timeout(Duration::from_secs(2), conn.recv())
        .await
        .expect("recv timeout")
        .expect("recv frame")
}

#[tokio::test]
async fn uds_server_binds_securely_and_streams_appended_events() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);
    let server_socket_path = socket_path.clone();
    let server_control_plane = control_plane.clone();
    let server_shutdown_tx = shutdown_tx.clone();
    let server = tokio::spawn(async move {
        let mut shutdown = server_shutdown_tx.subscribe();
        redesmyn_control_plane::client_api::serve_client_api_uds(
            server_control_plane,
            server_socket_path,
            ClientApiCodec::Protobuf,
            &mut shutdown,
        )
        .await
    });

    let stream = connect_with_retry(&socket_path).await;

    let mode = std::fs::metadata(&socket_path)
        .expect("socket metadata")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(mode, 0o600, "unexpected socket mode: {mode:o}");

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
        let frame = recv_frame(&mut conn).await;
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
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Subscribe(Subscribe {
            subscription_id,
            filter: SubscriptionFilter::EventLog(EventLogFilter {
                after_event_id: None,
            }),
        }),
    ))
    .await
    .expect("send subscribe");

    loop {
        let frame = recv_frame(&mut conn).await;
        let ClientMessage::Event(event) = frame.message else {
            continue;
        };
        if event.subscription_id != subscription_id {
            continue;
        }
        match event.event {
            SubscriptionEvent::Subscribed(_) => break,
            SubscriptionEvent::EventLog(_) => continue,
            SubscriptionEvent::Error(err) => panic!("unexpected subscription error: {err:?}"),
        }
    }

    let expected_payload = br#"{"hello":"world"}"#.to_vec();
    let expected_event_id = control_plane
        .event_log()
        .append_event(EventScope::None, "test.event", expected_payload.clone())
        .await
        .expect("append event");

    loop {
        let frame = recv_frame(&mut conn).await;
        let ClientMessage::Event(event) = frame.message else {
            continue;
        };
        if event.subscription_id != subscription_id {
            continue;
        }

        match event.event {
            SubscriptionEvent::EventLog(ev) => {
                assert_eq!(ev.event_id, expected_event_id);
                assert_eq!(ev.event_type, "test.event");
                assert_eq!(ev.json_payload, expected_payload);
                break;
            }
            SubscriptionEvent::Subscribed(_) => continue,
            SubscriptionEvent::Error(err) => panic!("unexpected subscription error: {err:?}"),
        }
    }

    let _ = shutdown_tx.send(());
    server.await.expect("server task").expect("server exit");
    assert!(
        !socket_path.exists(),
        "socket file should be removed on shutdown"
    );
}

#[tokio::test]
async fn event_log_subscription_supports_cursor_resume_over_uds() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (shutdown_tx, _shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    let after_event_id = control_plane
        .event_log()
        .append_event(EventScope::None, "test.first", br#"{"n":1}"#.to_vec())
        .await
        .expect("append first event");
    let expected_payload = br#"{"n":2}"#.to_vec();
    let expected_event_id = control_plane
        .event_log()
        .append_event(EventScope::None, "test.second", expected_payload.clone())
        .await
        .expect("append second event");

    let server_socket_path = socket_path.clone();
    let server_control_plane = control_plane.clone();
    let server_shutdown_tx = shutdown_tx.clone();
    let server = tokio::spawn(async move {
        let mut shutdown = server_shutdown_tx.subscribe();
        redesmyn_control_plane::client_api::serve_client_api_uds(
            server_control_plane,
            server_socket_path,
            ClientApiCodec::Protobuf,
            &mut shutdown,
        )
        .await
    });

    let stream = connect_with_retry(&socket_path).await;
    let mut conn = FramedEndpoint::new(stream, ProtobufCodec::new());

    let _handshake_request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id: _handshake_request_id,
            payload: RequestPayload::Health(HealthRequest {}),
        }),
    ))
    .await
    .expect("send health request");

    loop {
        let frame = recv_frame(&mut conn).await;
        if matches!(frame.message, ClientMessage::Response(_)) {
            break;
        }
    }

    let subscription_id = SubscriptionId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Subscribe(Subscribe {
            subscription_id,
            filter: SubscriptionFilter::EventLog(EventLogFilter {
                after_event_id: Some(after_event_id),
            }),
        }),
    ))
    .await
    .expect("send subscribe");

    loop {
        let frame = recv_frame(&mut conn).await;
        let ClientMessage::Event(event) = frame.message else {
            continue;
        };
        if event.subscription_id != subscription_id {
            continue;
        }

        match event.event {
            SubscriptionEvent::Subscribed(_) => continue,
            SubscriptionEvent::EventLog(ev) => {
                assert_eq!(ev.event_id, expected_event_id);
                assert_eq!(ev.event_type, "test.second");
                assert_eq!(ev.json_payload, expected_payload);
                break;
            }
            SubscriptionEvent::Error(err) => panic!("unexpected subscription error: {err:?}"),
        }
    }

    let _ = shutdown_tx.send(());
    server.await.expect("server task").expect("server exit");
}
