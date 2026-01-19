use redesmyn_ids::MsgId;
use redesmyn_protocol::daemon::{
    DaemonFrame, DaemonMessage, HelloRequest, HelloResponse, MessageEnvelope,
};
use redesmyn_transport::DaemonConnection;
use redesmyn_transport::codec::{JsonCodec, ProtobufCodec};
use redesmyn_transport::framed::FramedEndpoint;
use redesmyn_transport::in_proc::InProcEndpoint;

async fn hello_roundtrip<C, D>(mut control_plane: C, mut daemon: D)
where
    C: DaemonConnection + Send + 'static,
    D: DaemonConnection + Send + 'static,
{
    let request_id = MsgId::new();
    let request = DaemonFrame::new(
        MessageEnvelope::new(request_id),
        DaemonMessage::HelloRequest(HelloRequest {
            client_name: "control-plane".to_string(),
        }),
    );

    let daemon_task = tokio::spawn(async move {
        let received = daemon.recv().await.unwrap();
        let request_envelope = received.envelope;

        let DaemonMessage::HelloRequest(request) = received.message else {
            panic!("expected HelloRequest, got {:?}", received.message);
        };
        assert_eq!(request.client_name, "control-plane");

        let response = DaemonFrame::new(
            MessageEnvelope::reply(MsgId::new(), &request_envelope),
            DaemonMessage::HelloResponse(HelloResponse {
                daemon_name: "daemon".to_string(),
            }),
        );

        daemon.send(response).await.unwrap();
    });

    control_plane.send(request).await.unwrap();
    let response_frame = control_plane.recv().await.unwrap();
    daemon_task.await.unwrap();

    let response_envelope = response_frame.envelope;
    let DaemonMessage::HelloResponse(response) = response_frame.message else {
        panic!("expected HelloResponse, got {:?}", response_frame.message);
    };
    assert_eq!(response.daemon_name, "daemon");
    assert_eq!(response_envelope.in_reply_to, Some(request_id));
}

#[tokio::test]
async fn loopback_in_proc_hello_roundtrip() {
    let (control_plane, daemon) = InProcEndpoint::pair(8);
    hello_roundtrip(control_plane, daemon).await;
}

#[tokio::test]
async fn loopback_framed_json_hello_roundtrip() {
    let (cp_stream, daemon_stream) = tokio::io::duplex(8 * 1024);
    let control_plane = FramedEndpoint::new(cp_stream, JsonCodec::new());
    let daemon = FramedEndpoint::new(daemon_stream, JsonCodec::new());
    hello_roundtrip(control_plane, daemon).await;
}

#[tokio::test]
async fn loopback_framed_protobuf_hello_roundtrip() {
    let (cp_stream, daemon_stream) = tokio::io::duplex(8 * 1024);
    let control_plane = FramedEndpoint::new(cp_stream, ProtobufCodec::new());
    let daemon = FramedEndpoint::new(daemon_stream, ProtobufCodec::new());
    hello_roundtrip(control_plane, daemon).await;
}
