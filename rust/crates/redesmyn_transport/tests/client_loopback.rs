use redesmyn_ids::{RequestId, SubscriptionId};
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, Event, EventLogFilter, HealthRequest, HealthResponse, Request,
    RequestPayload, Response, ResponseResult, Subscribe, Subscribed, SubscriptionEvent,
    SubscriptionFilter, SubscriptionTopic,
};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::codec::{JsonCodec, ProtobufCodec};
use redesmyn_transport::client::framed::FramedEndpoint;
use redesmyn_transport::client::in_proc::InProcEndpoint;

async fn health_roundtrip<C, S>(mut client: C, mut server: S)
where
    C: ClientConnection + Send + 'static,
    S: ClientConnection + Send + 'static,
{
    let request_id = RequestId::new();
    let request = ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id,
            payload: RequestPayload::Health(HealthRequest {}),
        }),
    );

    let server_task = tokio::spawn(async move {
        let received = server.recv().await.unwrap();
        let ClientMessage::Request(req) = received.message else {
            panic!("expected Request, got {:?}", received.message);
        };
        assert_eq!(req.request_id, request_id);

        let response = ClientFrame::new(
            ProtocolEnvelope::new(),
            ClientMessage::Response(Response {
                request_id,
                result: ResponseResult::Health(HealthResponse { ok: true }),
            }),
        );
        server.send(response).await.unwrap();
    });

    client.send(request).await.unwrap();
    let response = client.recv().await.unwrap();
    server_task.await.unwrap();

    let ClientMessage::Response(response) = response.message else {
        panic!("expected Response, got {:?}", response.message);
    };
    assert_eq!(response.request_id, request_id);
    let ResponseResult::Health(payload) = response.result else {
        panic!("expected Health response, got {:?}", response.result);
    };
    assert!(payload.ok);
}

#[tokio::test]
async fn loopback_client_in_proc_health_roundtrip() {
    let (client, server) = InProcEndpoint::pair(8);
    health_roundtrip(client, server).await;
}

#[tokio::test]
async fn loopback_client_framed_json_health_roundtrip() {
    let (client_stream, server_stream) = tokio::io::duplex(8 * 1024);
    let client = FramedEndpoint::new(client_stream, JsonCodec::new());
    let server = FramedEndpoint::new(server_stream, JsonCodec::new());
    health_roundtrip(client, server).await;
}

#[tokio::test]
async fn loopback_client_framed_protobuf_health_roundtrip() {
    let (client_stream, server_stream) = tokio::io::duplex(8 * 1024);
    let client = FramedEndpoint::new(client_stream, ProtobufCodec::new());
    let server = FramedEndpoint::new(server_stream, ProtobufCodec::new());
    health_roundtrip(client, server).await;
}

#[tokio::test]
async fn loopback_client_subscription_event_stream() {
    let (mut client, mut server) = InProcEndpoint::pair(8);

    let subscription_id = SubscriptionId::new();
    let subscribe = ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Subscribe(Subscribe {
            subscription_id,
            filter: SubscriptionFilter::EventLog(EventLogFilter {
                after_event_id: None,
            }),
        }),
    );

    let server_task = tokio::spawn(async move {
        let received = server.recv().await.unwrap();
        let ClientMessage::Subscribe(subscribe) = received.message else {
            panic!("expected Subscribe, got {:?}", received.message);
        };
        assert_eq!(subscribe.subscription_id, subscription_id);

        let ack = ClientFrame::new(
            ProtocolEnvelope::new(),
            ClientMessage::Event(Event {
                subscription_id,
                event: SubscriptionEvent::Subscribed(Subscribed {
                    topic: SubscriptionTopic::EventLog,
                }),
            }),
        );
        server.send(ack).await.unwrap();
    });

    client.send(subscribe).await.unwrap();
    let received = client.recv().await.unwrap();
    server_task.await.unwrap();

    let ClientMessage::Event(event) = received.message else {
        panic!("expected Event, got {:?}", received.message);
    };
    assert_eq!(event.subscription_id, subscription_id);
    let SubscriptionEvent::Subscribed(payload) = event.event else {
        panic!("expected Subscribed, got {:?}", event.event);
    };
    assert_eq!(payload.topic, SubscriptionTopic::EventLog);
}
