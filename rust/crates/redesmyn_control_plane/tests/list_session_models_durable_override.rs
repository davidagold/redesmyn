use std::time::Duration;

use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{
    EpicId, HostId, HostInstanceId, RepoId, RequestId, SessionEventId, SessionId, TaskId,
    WorkspaceId,
};
use redesmyn_protocol::agent_commands::SESSION_AGENT_LIST_MODELS;
use redesmyn_protocol::client::{
    ClientFrame, ClientMessage, ListSessionModelsRequest, ModelReasoningEffort, Request,
    RequestPayload, ResponseResult, SessionModelSelection,
};
use redesmyn_protocol::daemon::{
    CommandState as DaemonCommandState, CommandUpdate as DaemonCommandUpdate, DaemonMessage,
};
use redesmyn_protocol::session::{
    SessionEventKind, SessionModelChanged, SessionModelReasoningEffort, SessionScope,
};
use redesmyn_protocol::{ErrorDetail, ProtocolEnvelope, RepoScope, Scope, Timestamp};
use redesmyn_storage::schema::{
    AgentSessionScopeKind as StorageAgentSessionScopeKind,
    AgentSessionStatus as StorageAgentSessionStatus,
};
use redesmyn_storage::sessions::{AgentSessionRecord, insert_agent_session};
use redesmyn_transport::client::ClientConnection;
use redesmyn_transport::client::in_proc::InProcEndpoint;
use tokio::sync::{broadcast, mpsc};

async fn request_with_scope(
    conn: &mut InProcEndpoint,
    scope: Scope,
    payload: RequestPayload,
) -> ResponseResult {
    let request_id = RequestId::new();
    conn.send(ClientFrame::new(
        ProtocolEnvelope::new().with_scope(scope),
        ClientMessage::Request(Request {
            request_id,
            payload,
        }),
    ))
    .await
    .expect("send request");

    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let frame = conn.recv().await.expect("recv frame");
            match frame.message {
                ClientMessage::Response(resp) if resp.request_id == request_id => {
                    return resp.result;
                }
                _ => continue,
            }
        }
    })
    .await
    .expect("response timeout")
}

async fn seed_repo_task(
    pool: &redesmyn_storage::SqlitePool,
) -> (WorkspaceId, RepoId, TaskId, RepoScope) {
    let now_ms = 1_i64;
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();
    let task_id = TaskId::new();
    let branch_name = format!("rn/task/{}", task_id);

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test")
    .execute(pool)
    .await
    .expect("insert workspace");

    sqlx::query(
        r#"
        INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(repo_id)
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("repo")
    .bind("Repo")
    .execute(pool)
    .await
    .expect("insert repo");

    sqlx::query(
        r#"
        INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
    )
    .bind(epic_id)
    .bind(repo_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("epic")
    .bind("Epic")
    .execute(pool)
    .await
    .expect("insert epic");

    sqlx::query(
        r#"
        INSERT INTO tasks (id, epic_id, created_at_ms, updated_at_ms, title, branch_name, merge_readiness)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'unknown')
        "#,
    )
    .bind(task_id)
    .bind(epic_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("Task")
    .bind(branch_name)
    .execute(pool)
    .await
    .expect("insert task");

    (
        workspace_id,
        repo_id,
        task_id,
        RepoScope::new(workspace_id, repo_id),
    )
}

#[tokio::test]
async fn list_session_models_prefers_durable_selection_for_stale_running_sessions() {
    redesmyn_logging::init();

    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let pool = control_plane.pool().clone();

    let (workspace_id, repo_id, task_id, repo_scope) = seed_repo_task(&pool).await;
    let session_id = SessionId::new();

    insert_agent_session(
        &pool,
        &AgentSessionRecord {
            session_id,
            created_at_ms: 2,
            updated_at_ms: 2,
            scope_workspace_id: workspace_id,
            scope_repo_id: repo_id,
            scope_kind: StorageAgentSessionScopeKind::Task,
            task_id: Some(task_id),
            epic_id: None,
            agent_kind: redesmyn_storage::schema::AgentKind::Codex,
            status: StorageAgentSessionStatus::Running,
            external_session_ref: r#"{"type":"none"}"#.to_owned(),
            title: None,
            started_at_ms: Some(2),
            ended_at_ms: None,
            archived_at_ms: None,
            repo_name: None,
        },
    )
    .await
    .expect("insert session");

    control_plane
        .session_events()
        .append_session_event(&redesmyn_protocol::SessionEvent {
            session_event_id: SessionEventId::new(),
            created_at: Timestamp::from_unix_millis(3).expect("timestamp"),
            scope: SessionScope::Task { task_id },
            session_id,
            turn_id: None,
            kind: SessionEventKind::SessionModelChanged(SessionModelChanged {
                model_id: Some("gpt-5.3-codex".to_string()),
                reasoning_effort: Some(SessionModelReasoningEffort::High),
            }),
        })
        .await
        .expect("append session model changed event");

    let host_id = HostId::new();
    let host_instance_id = HostInstanceId::new();
    control_plane
        .register_daemon_presence(host_id, host_instance_id)
        .await
        .expect("register daemon presence");

    let (outbound_tx, mut outbound_rx) = mpsc::channel(8);
    control_plane
        .daemons()
        .register_connection(
            host_id,
            host_instance_id,
            redesmyn_protocol::ProtocolVersion::CURRENT,
            outbound_tx,
        )
        .await;
    control_plane
        .daemons()
        .attach_repo(host_instance_id, repo_scope)
        .await;

    let responder_control_plane = control_plane.clone();
    let responder = tokio::spawn(async move {
        let frame = outbound_rx.recv().await.expect("dispatch frame");
        let DaemonMessage::CommandDispatch(dispatch) = frame.message else {
            panic!("unexpected daemon frame");
        };
        assert_eq!(dispatch.command_kind, SESSION_AGENT_LIST_MODELS);

        responder_control_plane
            .apply_daemon_command_update(
                host_instance_id,
                DaemonCommandUpdate {
                    command_id: dispatch.command_id,
                    state: DaemonCommandState::Accepted,
                    message: Some("accepted".to_string()),
                    progress: None,
                    detail: None,
                    error: None,
                },
            )
            .await
            .expect("append accepted update");

        let backend_response = redesmyn_protocol::client::ListSessionModelsResponse {
            options: Vec::new(),
            selection: SessionModelSelection {
                model_id: Some("backend-default".to_string()),
                reasoning_effort: Some(ModelReasoningEffort::Medium),
            },
        };
        let models_json = serde_json::to_string(&backend_response).expect("encode models json");
        let detail = ErrorDetail::from([("models_json".to_string(), models_json)]);
        responder_control_plane
            .apply_daemon_command_update(
                host_instance_id,
                DaemonCommandUpdate {
                    command_id: dispatch.command_id,
                    state: DaemonCommandState::Succeeded,
                    message: Some("listed".to_string()),
                    progress: None,
                    detail: Some(detail),
                    error: None,
                },
            )
            .await
            .expect("append succeeded update");
    });

    let (mut client, mut server) = InProcEndpoint::pair(8);
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let mut shutdown = shutdown_tx.subscribe();
    let server_control_plane = control_plane.clone();
    let server_task = tokio::spawn(async move {
        let _ = redesmyn_control_plane::client_api::serve_connection(
            &mut server,
            server_control_plane,
            &mut shutdown,
        )
        .await;
    });

    let result = request_with_scope(
        &mut client,
        Scope::from(repo_scope),
        RequestPayload::ListSessionModels(ListSessionModelsRequest { session_id }),
    )
    .await;

    let resp = match result {
        ResponseResult::ListSessionModels(resp) => resp,
        other => panic!("unexpected response: {other:?}"),
    };

    assert_eq!(
        resp.selection.model_id.as_deref(),
        Some("gpt-5.3-codex"),
        "durable model selection should override backend default"
    );
    assert_eq!(
        resp.selection.reasoning_effort,
        Some(ModelReasoningEffort::High),
        "durable reasoning effort should override backend default"
    );

    responder.await.expect("join responder");
    let _ = shutdown_tx.send(());
    let _ = server_task.await;
}
