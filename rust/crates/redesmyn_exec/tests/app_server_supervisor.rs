use std::time::Duration;

use redesmyn_domain::agent::AppServerTurnIntent;
use redesmyn_exec::app_server::{
    AppServerClient, AppServerConnection, AppServerEvent, AppServerProcess, AppServerProcessError,
    AppServerRequest, AppServerRequestError, AppServerResponse, AppServerSessionSpec,
    AppServerSupervisor, AppServerSupervisorConfig, BoxFuture,
};
use redesmyn_exec::artifact_store::LocalArtifactStore;
use redesmyn_ids::TaskId;
use redesmyn_protocol::daemon::DaemonMessage;
use redesmyn_protocol::session::{SessionEvent, SessionEventKind, SessionScope};
use tokio::sync::mpsc;

#[derive(Clone)]
struct FakeAppServerProcess;

impl AppServerProcess for FakeAppServerProcess {
    fn start(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }

    fn connect(&self) -> BoxFuture<'_, Result<AppServerConnection, AppServerProcessError>> {
        Box::pin(async {
            let (events_tx, events_rx) = mpsc::channel::<AppServerEvent>(16);
            Ok(AppServerConnection {
                client: std::sync::Arc::new(FakeAppServerClient { events_tx }),
                events: events_rx,
            })
        })
    }

    fn shutdown(&self) -> BoxFuture<'_, Result<(), AppServerProcessError>> {
        Box::pin(async { Ok(()) })
    }
}

struct FakeAppServerClient {
    events_tx: mpsc::Sender<AppServerEvent>,
}

impl AppServerClient for FakeAppServerClient {
    fn request(
        &self,
        request: AppServerRequest,
    ) -> BoxFuture<'_, Result<AppServerResponse, AppServerRequestError>> {
        let events_tx = self.events_tx.clone();
        Box::pin(async move {
            match request {
                AppServerRequest::SendMessage { intent } => {
                    let prompt = match intent {
                        AppServerTurnIntent::StartNew { prompt } => prompt,
                        AppServerTurnIntent::Resume { prompt, .. } => prompt,
                    };

                    let _ = events_tx
                        .send(AppServerEvent::AssistantMessage {
                            text: format!("echo: {prompt}"),
                        })
                        .await;
                    Ok(AppServerResponse::MessageAccepted)
                }
                AppServerRequest::Interrupt => Ok(AppServerResponse::Interrupted),
            }
        })
    }
}

async fn collect_session_records(
    rx: &mut mpsc::Receiver<redesmyn_protocol::daemon::DaemonFrame>,
    session_id: redesmyn_ids::SessionId,
    timeout: Duration,
) -> Vec<SessionEvent> {
    let frame = tokio::time::timeout(timeout, rx.recv())
        .await
        .expect("timeout waiting for daemon frame")
        .expect("frame channel closed");

    match frame.message {
        DaemonMessage::SessionEventBatch(batch) => batch
            .events
            .into_iter()
            .filter(|event| event.session_id == session_id)
            .collect(),
        _ => Vec::new(),
    }
}

#[tokio::test]
async fn send_message_round_trips_and_emits_structured_event() {
    let (frames_tx, mut frames_rx) = mpsc::channel(256);
    let tmp = tempfile::tempdir().expect("tempdir");
    let artifact_store = LocalArtifactStore::new(tmp.path().to_path_buf());

    let supervisor = AppServerSupervisor::new(
        AppServerSupervisorConfig::default(),
        artifact_store,
        frames_tx,
    )
    .await
    .expect("supervisor");

    let task_id = TaskId::new();
    let scope = SessionScope::Task { task_id };

    let session_id = supervisor
        .start_session(
            task_id,
            AppServerSessionSpec {
                scope,
                allow_concurrent_for_task: false,
                process: std::sync::Arc::new(FakeAppServerProcess),
            },
        )
        .await
        .expect("start_session");

    let response = supervisor
        .send_message(
            session_id,
            AppServerTurnIntent::StartNew {
                prompt: "hi".to_owned(),
            },
        )
        .await
        .expect("send_message");
    assert_eq!(response, AppServerResponse::MessageAccepted);

    let mut saw_assistant_message = false;
    for _ in 0..10 {
        let records =
            collect_session_records(&mut frames_rx, session_id, Duration::from_secs(2)).await;
        for event in records {
            if matches!(event.kind, SessionEventKind::AssistantMessage(_)) {
                saw_assistant_message = true;
                break;
            }
        }
        if saw_assistant_message {
            break;
        }
    }
    assert!(saw_assistant_message, "expected an assistant message event");

    supervisor
        .stop_session(session_id)
        .await
        .expect("stop_session");
    supervisor.shutdown().await;
}
