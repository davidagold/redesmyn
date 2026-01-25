use std::time::Duration;

use redesmyn_client_api::Client;
use redesmyn_ids::SessionId;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, Event, HealthRequest, HealthResponse, RequestPayload, Response,
    ResponseResult, StatusRequest, StatusResponse, Subscribe, Subscribed, SubscriptionEvent,
    SubscriptionTopic, Unsubscribe,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::in_proc::InProcEndpoint;

#[tokio::test]
async fn request_demux_handles_out_of_order_responses() {
    let (client_conn, mut server_conn) = InProcEndpoint::pair(16);
    let (client, client_task) = Client::connect(client_conn, 16);
    let client_runner = tokio::spawn(client_task.run());

    let server = tokio::spawn(async move {
        let mut pending = Vec::new();
        for _ in 0..2 {
            let frame = server_conn.recv().await.expect("recv request");
            let ClientMessage::Request(req) = frame.message else {
                panic!("expected Request, got {:?}", frame.message);
            };
            pending.push(req);
        }

        for req in pending.into_iter().rev() {
            let result = match req.payload {
                RequestPayload::Health(HealthRequest {}) => {
                    ResponseResult::Health(HealthResponse { ok: true })
                }
                RequestPayload::Status(StatusRequest {}) => {
                    ResponseResult::Status(StatusResponse {
                        accepted_protocol: ProtocolVersion::CURRENT,
                        server_name: "test".to_string(),
                        server_version: None,
                    })
                }
                other => panic!("unexpected request payload: {other:?}"),
            };

            server_conn
                .send(ClientFrame::new(
                    ProtocolEnvelope::new(),
                    ClientMessage::Response(Response {
                        request_id: req.request_id,
                        result,
                    }),
                ))
                .await
                .expect("send response");
        }
    });

    let status = client.request(RequestPayload::Status(StatusRequest {}));
    let health = client.request(RequestPayload::Health(HealthRequest {}));

    let (status, health) = tokio::join!(status, health);
    assert!(matches!(
        status.expect("status ok"),
        ResponseResult::Status(_)
    ));
    assert!(matches!(
        health.expect("health ok"),
        ResponseResult::Health(_)
    ));

    drop(client);
    server.await.expect("server task");
    client_runner.await.expect("client runner");
}

#[tokio::test]
async fn subscriptions_deliver_events_and_unsubscribe_stops_delivery() {
    let (client_conn, mut server_conn) = InProcEndpoint::pair(16);
    let (client, client_task) = Client::connect(client_conn, 16);
    let client_runner = tokio::spawn(client_task.run());

    let session_id = SessionId::new();

    let server = tokio::spawn(async move {
        let frame = server_conn.recv().await.expect("recv subscribe");
        let ClientMessage::Subscribe(Subscribe {
            subscription_id, ..
        }) = frame.message
        else {
            panic!("expected Subscribe, got {:?}", frame.message);
        };

        server_conn
            .send(ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Event(Event {
                    subscription_id,
                    event: SubscriptionEvent::Subscribed(Subscribed {
                        topic: SubscriptionTopic::SessionEvents,
                    }),
                }),
            ))
            .await
            .expect("send subscribed");

        server_conn
            .send(ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Event(Event {
                    subscription_id,
                    event: SubscriptionEvent::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "boom",
                    )),
                }),
            ))
            .await
            .expect("send event");

        let frame = server_conn.recv().await.expect("recv unsubscribe");
        let ClientMessage::Unsubscribe(Unsubscribe {
            subscription_id: unsub_id,
        }) = frame.message
        else {
            panic!("expected Unsubscribe, got {:?}", frame.message);
        };
        assert_eq!(unsub_id, subscription_id);

        server_conn
            .send(ClientFrame::new(
                ProtocolEnvelope::new(),
                ClientMessage::Event(Event {
                    subscription_id,
                    event: SubscriptionEvent::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "should-not-deliver",
                    )),
                }),
            ))
            .await
            .expect("send trailing event");
    });

    let (subscription_id, mut events_rx) = client
        .subscribe_session_events(session_id, None)
        .await
        .expect("subscribe");

    let first = tokio::time::timeout(Duration::from_secs(1), events_rx.recv())
        .await
        .expect("timeout")
        .expect("event");
    assert!(matches!(first, SubscriptionEvent::Subscribed(_)));

    let second = tokio::time::timeout(Duration::from_secs(1), events_rx.recv())
        .await
        .expect("timeout")
        .expect("event");
    let SubscriptionEvent::Error(err) = second else {
        panic!("expected error, got {second:?}");
    };
    assert_eq!(err.message, "boom");

    client
        .unsubscribe(subscription_id)
        .await
        .expect("unsubscribe");

    let terminal = tokio::time::timeout(Duration::from_millis(200), events_rx.recv())
        .await
        .expect("timeout");
    assert_eq!(terminal, None);

    drop(client);
    server.await.expect("server task");
    client_runner.await.expect("client runner");
}
