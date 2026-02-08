use std::time::Duration;

use redesmyn_control_plane::client_api::ClientApiCodec;
use redesmyn_control_plane::{ControlPlane, ControlPlaneDb, ControlPlaneStartOptions};
use redesmyn_ids::RequestId;
use redesmyn_protocol::ProtocolEnvelope;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, Request, RequestPayload, ResponseResult, StatusRequest,
    WaitForIdleRequest,
};
use redesmyn_transport::client::ClientConnection;

#[tokio::test]
async fn in_proc_client_multiplexing_allows_status_while_wait_in_flight() {
    let options = ControlPlaneStartOptions {
        db: ControlPlaneDb::InMemory,
        client_api_socket_path: None,
        client_api_codec: ClientApiCodec::Protobuf,
    };
    let mut control_plane = ControlPlane::start(options).await.expect("start");

    let mut conn = control_plane.connect_in_proc_client(8);

    let wait_request_id = RequestId::new();
    let status_request_id = RequestId::new();

    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id: wait_request_id,
            payload: RequestPayload::WaitForIdle(WaitForIdleRequest {
                scope: None,
                timeout_ms: 5_000,
                quiescence_ms: 1_500,
            }),
        }),
    ))
    .await
    .expect("send wait_for_idle request");

    conn.send(ClientFrame::new(
        ProtocolEnvelope::new(),
        ClientMessage::Request(Request {
            request_id: status_request_id,
            payload: RequestPayload::Status(StatusRequest {}),
        }),
    ))
    .await
    .expect("send status request");

    let status_result = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let frame = conn.recv().await.expect("recv response");
            let ClientMessage::Response(resp) = frame.message else {
                continue;
            };

            if resp.request_id == status_request_id {
                return resp.result;
            }
        }
    })
    .await
    .expect("status response should not be blocked by wait");

    match status_result {
        ResponseResult::Status(payload) => {
            assert_eq!(payload.server_name, "redesmyn-control-plane");
        }
        ResponseResult::Error(err) => panic!("unexpected error response: {err:?}"),
        other => panic!("unexpected status response: {other:?}"),
    }

    control_plane.shutdown().await;
}
