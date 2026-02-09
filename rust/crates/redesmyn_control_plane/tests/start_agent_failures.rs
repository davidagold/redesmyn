use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{EpicId, RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_protocol::ErrorCategory;
use redesmyn_protocol::Timestamp;
use redesmyn_protocol::client::{
    AgentKind, AgentMessageConflictAction, ModelReasoningEffort, RestartAgentRequest,
    SessionEventKindFilter, SessionModelSelection, StartAgentRequest,
};
use redesmyn_protocol::session::{
    CodexApprovalPolicy, CodexApprovalPolicyChanged, CodexSandboxPolicy, CodexSandboxPolicyChanged,
    SessionEvent, SessionEventKind, SessionModelChanged, SessionModelReasoningEffort, SessionScope,
};

async fn seed_repo_and_task(control_plane: &ControlPlane) -> (WorkspaceId, RepoId, TaskId) {
    let pool = control_plane.pool();

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

    (workspace_id, repo_id, task_id)
}

async fn insert_stopped_task_session(
    control_plane: &ControlPlane,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    task_id: TaskId,
    now_ms: i64,
) -> SessionId {
    let session_id = SessionId::new();
    sqlx::query(
        r#"
        INSERT INTO agent_sessions (
            session_id,
            created_at_ms,
            updated_at_ms,
            scope_workspace_id,
            scope_repo_id,
            scope_kind,
            task_id,
            agent_kind,
            status,
            external_session_ref,
            title,
            started_at_ms,
            ended_at_ms,
            archived_at_ms
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, 'task', ?6, 'codex', 'stopped', ?7, NULL, ?8, ?9, NULL
        )
        "#,
    )
    .bind(session_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(workspace_id)
    .bind(repo_id)
    .bind(task_id)
    .bind(r#"{"type":"none"}"#)
    .bind(now_ms)
    .bind(now_ms)
    .execute(control_plane.pool())
    .await
    .expect("insert stopped task session");
    session_id
}

async fn append_restart_sticky_events(
    control_plane: &ControlPlane,
    session_id: SessionId,
    task_id: TaskId,
) {
    let created_at = Timestamp::now_utc();
    let events = [
        SessionEventKind::SessionModelChanged(SessionModelChanged {
            model_id: Some("gpt-5.3-codex".to_string()),
            reasoning_effort: Some(SessionModelReasoningEffort::Xhigh),
        }),
        SessionEventKind::CodexApprovalPolicyChanged(CodexApprovalPolicyChanged {
            approval_policy: Some(CodexApprovalPolicy::OnFailure),
        }),
        SessionEventKind::CodexSandboxPolicyChanged(CodexSandboxPolicyChanged {
            sandbox_policy: Some(CodexSandboxPolicy::DangerFullAccess),
        }),
    ];

    for kind in events {
        control_plane
            .session_events()
            .append_session_event(&SessionEvent {
                session_event_id: SessionEventId::new(),
                created_at,
                scope: SessionScope::Task { task_id },
                session_id,
                turn_id: None,
                kind,
            })
            .await
            .expect("append sticky session event");
    }
}

async fn assert_session_has_sticky_events(control_plane: &ControlPlane, session_id: SessionId) {
    assert_session_has_expected_sticky_events(
        control_plane,
        session_id,
        "gpt-5.3-codex",
        SessionModelReasoningEffort::Xhigh,
        CodexApprovalPolicy::OnFailure,
        CodexSandboxPolicy::DangerFullAccess,
    )
    .await;
}

async fn assert_session_has_expected_sticky_events(
    control_plane: &ControlPlane,
    session_id: SessionId,
    expected_model_id: &str,
    expected_reasoning_effort: SessionModelReasoningEffort,
    expected_approval_policy: CodexApprovalPolicy,
    expected_sandbox_policy: CodexSandboxPolicy,
) {
    let (events, _) = control_plane
        .session_events()
        .get_session_events(
            session_id,
            None,
            20,
            &[
                SessionEventKindFilter::SessionModelChanged,
                SessionEventKindFilter::CodexApprovalPolicyChanged,
                SessionEventKindFilter::CodexSandboxPolicyChanged,
            ],
        )
        .await
        .expect("load session sticky events");

    let mut saw_model = false;
    let mut saw_approval = false;
    let mut saw_sandbox = false;
    for event in events {
        match event.kind {
            SessionEventKind::SessionModelChanged(changed) => {
                saw_model = changed.model_id.as_deref() == Some(expected_model_id)
                    && changed.reasoning_effort == Some(expected_reasoning_effort);
            }
            SessionEventKind::CodexApprovalPolicyChanged(changed) => {
                saw_approval = changed.approval_policy == Some(expected_approval_policy);
            }
            SessionEventKind::CodexSandboxPolicyChanged(changed) => {
                saw_sandbox = changed.sandbox_policy == Some(expected_sandbox_policy.clone());
            }
            _ => {}
        }
    }

    assert!(saw_model, "expected model selection carry-forward event");
    assert!(saw_approval, "expected approval policy carry-forward event");
    assert!(saw_sandbox, "expected sandbox policy carry-forward event");
}

#[tokio::test]
async fn start_agent_returns_unavailable_and_rolls_back_session_when_no_daemon_is_connected() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;

    let result = control_plane
        .start_agent(
            workspace_id,
            repo_id,
            StartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: None,
                on_conflict: AgentMessageConflictAction::Fail,
                session_model_selection: None,
                codex_approval_policy: None,
                codex_sandbox_policy: None,
            },
        )
        .await;

    let err = result.expect_err("expected start-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);
    assert!(
        err.message.contains("No daemon connection is available")
            || err.message.contains("Failed to start agent."),
        "unexpected start-agent error message: {}",
        err.message
    );

    let active_sessions: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM agent_sessions
        WHERE task_id = ?1 AND ended_at_ms IS NULL
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("count active sessions");
    assert_eq!(active_sessions, 0, "expected no active task sessions");

    let stopped_sessions: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM agent_sessions
        WHERE task_id = ?1 AND status = 'stopped' AND ended_at_ms IS NOT NULL
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("count stopped sessions");
    assert_eq!(
        stopped_sessions, 1,
        "expected a rolled-back stopped session"
    );
}

#[tokio::test]
async fn start_agent_failure_does_not_append_initial_prompt_message() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;

    let result = control_plane
        .start_agent(
            workspace_id,
            repo_id,
            StartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: Some("prelude message".to_string()),
                on_conflict: AgentMessageConflictAction::Fail,
                session_model_selection: None,
                codex_approval_policy: None,
                codex_sandbox_policy: None,
            },
        )
        .await;

    let err = result.expect_err("expected start-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);

    let persisted_user_messages: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM session_events
        WHERE task_id = ?1 AND kind = 'user_message'
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("count persisted user messages");

    assert_eq!(
        persisted_user_messages, 0,
        "initial prompt should not be persisted when start fails"
    );
}

#[tokio::test]
async fn restart_agent_carries_forward_latest_session_settings() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;
    let old_session_id =
        insert_stopped_task_session(&control_plane, workspace_id, repo_id, task_id, 11).await;
    append_restart_sticky_events(&control_plane, old_session_id, task_id).await;

    let result = control_plane
        .restart_agent(
            workspace_id,
            repo_id,
            RestartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: Some("prelude template".to_string()),
                session_model_selection: None,
                codex_approval_policy: None,
                codex_sandbox_policy: None,
            },
        )
        .await;

    let err = result.expect_err("expected restart-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);

    let latest_session_id: SessionId = sqlx::query_scalar(
        r#"
        SELECT session_id
        FROM agent_sessions
        WHERE task_id = ?1
        ORDER BY created_at_ms DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("load latest session id");
    assert_ne!(latest_session_id, old_session_id);

    assert_session_has_sticky_events(&control_plane, latest_session_id).await;
}

#[tokio::test]
async fn restart_agent_uses_latest_non_empty_sticky_settings() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;

    let sticky_session_id =
        insert_stopped_task_session(&control_plane, workspace_id, repo_id, task_id, 11).await;
    append_restart_sticky_events(&control_plane, sticky_session_id, task_id).await;

    let empty_latest_session_id =
        insert_stopped_task_session(&control_plane, workspace_id, repo_id, task_id, 12).await;

    let result = control_plane
        .restart_agent(
            workspace_id,
            repo_id,
            RestartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: Some("prelude template".to_string()),
                session_model_selection: None,
                codex_approval_policy: None,
                codex_sandbox_policy: None,
            },
        )
        .await;

    let err = result.expect_err("expected restart-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);

    let latest_session_id: SessionId = sqlx::query_scalar(
        r#"
        SELECT session_id
        FROM agent_sessions
        WHERE task_id = ?1
        ORDER BY created_at_ms DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("load latest session id");
    assert_ne!(latest_session_id, empty_latest_session_id);
    assert_ne!(latest_session_id, sticky_session_id);

    assert_session_has_sticky_events(&control_plane, latest_session_id).await;
}

#[tokio::test]
async fn restart_agent_falls_back_to_request_settings_when_no_last_seen_projection() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;

    let result = control_plane
        .restart_agent(
            workspace_id,
            repo_id,
            RestartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: Some("prelude template".to_string()),
                session_model_selection: Some(SessionModelSelection {
                    model_id: Some("gpt-5.3-codex".to_string()),
                    reasoning_effort: Some(ModelReasoningEffort::Xhigh),
                }),
                codex_approval_policy: Some(CodexApprovalPolicy::OnFailure),
                codex_sandbox_policy: Some(CodexSandboxPolicy::DangerFullAccess),
            },
        )
        .await;

    let err = result.expect_err("expected restart-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);

    let latest_session_id: SessionId = sqlx::query_scalar(
        r#"
        SELECT session_id
        FROM agent_sessions
        WHERE task_id = ?1
        ORDER BY created_at_ms DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("load latest session id");

    assert_session_has_sticky_events(&control_plane, latest_session_id).await;
}

#[tokio::test]
async fn restart_agent_prefers_request_settings_over_last_seen_projection() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");
    let (workspace_id, repo_id, task_id) = seed_repo_and_task(&control_plane).await;
    let old_session_id =
        insert_stopped_task_session(&control_plane, workspace_id, repo_id, task_id, 11).await;
    append_restart_sticky_events(&control_plane, old_session_id, task_id).await;

    let result = control_plane
        .restart_agent(
            workspace_id,
            repo_id,
            RestartAgentRequest {
                task_id,
                agent_kind: AgentKind::Codex,
                initial_prompt: Some("prelude template".to_string()),
                session_model_selection: Some(SessionModelSelection {
                    model_id: Some("gpt-5.2-codex".to_string()),
                    reasoning_effort: Some(ModelReasoningEffort::Low),
                }),
                codex_approval_policy: Some(CodexApprovalPolicy::OnRequest),
                codex_sandbox_policy: Some(CodexSandboxPolicy::ReadOnly),
            },
        )
        .await;

    let err = result.expect_err("expected restart-agent failure without daemon");
    assert_eq!(err.category, ErrorCategory::Unavailable);

    let latest_session_id: SessionId = sqlx::query_scalar(
        r#"
        SELECT session_id
        FROM agent_sessions
        WHERE task_id = ?1
        ORDER BY created_at_ms DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_one(control_plane.pool())
    .await
    .expect("load latest session id");
    assert_ne!(latest_session_id, old_session_id);

    assert_session_has_expected_sticky_events(
        &control_plane,
        latest_session_id,
        "gpt-5.2-codex",
        SessionModelReasoningEffort::Low,
        CodexApprovalPolicy::OnRequest,
        CodexSandboxPolicy::ReadOnly,
    )
    .await;
}
