use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_control_plane::ControlPlaneDb;
use redesmyn_control_plane::ControlPlaneStartOptions;
use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_ids::RequestId;
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, HealthRequest, Request, RequestPayload, ResponseResult,
    StatusRequest,
};
use redesmyn_transport::client::ClientConnection;

#[tokio::test]
async fn in_proc_client_supports_health_and_status_requests() {
    let options = ControlPlaneStartOptions {
        db: ControlPlaneDb::InMemory,
        client_api_socket_path: None,
        client_api_codec: ClientApiCodec::Protobuf,
    };
    let mut control_plane = ControlPlane::start(options).await.expect("start");

    let mut conn = control_plane.connect_in_proc_client(8);

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
        let frame = tokio::time::timeout(Duration::from_secs(1), conn.recv())
            .await
            .expect("timeout waiting for response")
            .expect("recv response");

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

    control_plane.shutdown().await;
}
