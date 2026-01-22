use redesmyn_ids::{HostId, HostInstanceId};
use redesmyn_protocol::daemon::{ControlPlaneHelloAck, DaemonFrame, DaemonHello, DaemonMessage};
use redesmyn_protocol::{ProtocolEnvelope, ProtocolVersion};
use redesmyn_transport::DaemonConnection;
use redesmyn_transport::codec::{JsonCodec, ProtobufCodec};
use redesmyn_transport::framed::FramedEndpoint;
use redesmyn_transport::in_proc::InProcEndpoint;

async fn handshake_roundtrip<C, D>(mut control_plane: C, mut daemon: D)
where
    C: DaemonConnection + Send + 'static,
    D: DaemonConnection + Send + 'static,
{
    let hello = DaemonHello {
        host_id: HostId::new(),
        host_instance_id: HostInstanceId::new(),
        capabilities: vec!["git".to_string()],
        supported_protocol: ProtocolVersion::CURRENT,
    };

    let request_envelope = ProtocolEnvelope::new();
    let request_id = request_envelope.msg_id;

    let request = DaemonFrame::new(request_envelope, DaemonMessage::DaemonHello(hello.clone()));

    let control_plane_task = tokio::spawn(async move {
        let received = control_plane.recv().await.unwrap();

        let DaemonMessage::DaemonHello(received_hello) = received.message else {
            panic!("expected DaemonHello, got {:?}", received.message);
        };
        assert_eq!(received_hello, hello);

        let ack = ControlPlaneHelloAck {
            accepted_protocol: ProtocolVersion::CURRENT,
            capabilities: vec!["server".to_string()],
        };

        let response = DaemonFrame::new(
            ProtocolEnvelope::new().with_correlation_id(received.envelope.msg_id),
            DaemonMessage::ControlPlaneHelloAck(ack),
        );

        control_plane.send(response).await.unwrap();
    });

    daemon.send(request).await.unwrap();
    let response_frame = daemon.recv().await.unwrap();
    control_plane_task.await.unwrap();

    let response_envelope = response_frame.envelope;
    let DaemonMessage::ControlPlaneHelloAck(_response) = response_frame.message else {
        panic!(
            "expected ControlPlaneHelloAck, got {:?}",
            response_frame.message
        );
    };
    assert_eq!(response_envelope.correlation_id, Some(request_id));
}

#[tokio::test]
async fn loopback_in_proc_hello_roundtrip() {
    let (control_plane, daemon) = InProcEndpoint::pair(8);
    handshake_roundtrip(control_plane, daemon).await;
}

#[tokio::test]
async fn loopback_framed_json_hello_roundtrip() {
    let (cp_stream, daemon_stream) = tokio::io::duplex(8 * 1024);
    let control_plane = FramedEndpoint::new(cp_stream, JsonCodec::new());
    let daemon = FramedEndpoint::new(daemon_stream, JsonCodec::new());
    handshake_roundtrip(control_plane, daemon).await;
}

#[tokio::test]
async fn loopback_framed_protobuf_hello_roundtrip() {
    let (cp_stream, daemon_stream) = tokio::io::duplex(8 * 1024);
    let control_plane = FramedEndpoint::new(cp_stream, ProtobufCodec::new());
    let daemon = FramedEndpoint::new(daemon_stream, ProtobufCodec::new());
    handshake_roundtrip(control_plane, daemon).await;
}
