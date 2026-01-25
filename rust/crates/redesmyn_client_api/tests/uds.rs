#![cfg(all(unix, feature = "uds"))]

use redesmyn_client_api::uds::{UdsConnectOptions, connect_uds};
use redesmyn_control_plane::client_api::ClientApiCodec as ServerClientApiCodec;
use redesmyn_control_plane::{ControlPlane, ControlPlaneDb, ControlPlaneStartOptions};
use redesmyn_protocol::client::{HealthRequest, RequestPayload, ResponseResult};

#[tokio::test]
async fn uds_dialer_connects_and_speaks_client_api() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let socket_path = tmp.path().join("control_plane.sock");

    let options = ControlPlaneStartOptions {
        db: ControlPlaneDb::InMemory,
        client_api_socket_path: Some(socket_path.clone()),
        client_api_codec: ServerClientApiCodec::Protobuf,
    };
    let control_plane = ControlPlane::start(options).await.expect("start");

    let (client, client_task) = connect_uds(UdsConnectOptions::new(socket_path.clone()), 64)
        .await
        .expect("connect");
    let client_runner = tokio::spawn(client_task.run());

    let health = client
        .request(RequestPayload::Health(HealthRequest {}))
        .await
        .expect("health");
    match health {
        ResponseResult::Health(resp) => assert!(resp.ok),
        other => panic!("unexpected Health response: {other:?}"),
    }

    control_plane.shutdown().await;
    client_runner.await.expect("client runner");

    assert!(
        !socket_path.exists(),
        "socket file should be removed on shutdown"
    );
}
