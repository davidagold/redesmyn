use prost::Message;
use redesmyn_ids::{EpicId, RepoId, WorkspaceId};
use redesmyn_protocol::pb::redesmyn::protocol::v1 as pbv1;
use redesmyn_storage::{
    open_test_sqlite_pool,
    schema::AgentKind,
    sessions::{
        NewSessionEvent, SessionEventCursor, SessionEventsQuery, append_session_event,
        create_chat_session, get_session_events, pin_chat_session_to_epic,
        unpin_chat_session_from_epic,
    },
};

async fn insert_workspace(pool: &sqlx::SqlitePool, workspace_id: WorkspaceId) {
    let now_ms = 1_i64;
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
    .unwrap();
}

async fn insert_repo(pool: &sqlx::SqlitePool, repo_id: RepoId, workspace_id: WorkspaceId) {
    let now_ms = 2_i64;
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
    .unwrap();
}

async fn insert_epic(pool: &sqlx::SqlitePool, epic_id: EpicId, repo_id: RepoId) {
    let now_ms = 3_i64;
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
    .unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn persists_session_events_with_pagination_and_kind_filter() {
    redesmyn_logging::init();

    let pool = open_test_sqlite_pool().await.unwrap();

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();

    insert_workspace(&pool, workspace_id).await;
    insert_repo(&pool, repo_id, workspace_id).await;
    insert_epic(&pool, epic_id, repo_id).await;

    let session_id = create_chat_session(
        &pool,
        workspace_id,
        repo_id,
        None,
        AgentKind::Codex,
        Some("Chat"),
    )
    .await
    .unwrap();

    pin_chat_session_to_epic(&pool, epic_id, session_id)
        .await
        .unwrap();
    unpin_chat_session_from_epic(&pool, epic_id).await.unwrap();

    let payload_a = pbv1::UnknownSessionEvent {
        event_type: "assistant_message".to_owned(),
        json_payload: br#"{"text":"hi"}"#.to_vec(),
    }
    .encode_to_vec();
    let payload_b = pbv1::UnknownSessionEvent {
        event_type: "tool_invocation".to_owned(),
        json_payload: br#"{"tool":"x"}"#.to_vec(),
    }
    .encode_to_vec();
    let payload_c = pbv1::UnknownSessionEvent {
        event_type: "assistant_message".to_owned(),
        json_payload: br#"{"text":"bye"}"#.to_vec(),
    }
    .encode_to_vec();

    let event_a = append_session_event(
        &pool,
        session_id,
        &NewSessionEvent {
            kind: "assistant_message".to_owned(),
            turn_id: Some("turn_1".to_owned()),
            message_preview: Some("hi".to_owned()),
            artifact_id: None,
            payload: payload_a.clone(),
            created_at_ms: Some(10),
        },
    )
    .await
    .unwrap();

    let event_b = append_session_event(
        &pool,
        session_id,
        &NewSessionEvent {
            kind: "tool_invocation".to_owned(),
            turn_id: Some("turn_1".to_owned()),
            message_preview: None,
            artifact_id: None,
            payload: payload_b.clone(),
            created_at_ms: Some(20),
        },
    )
    .await
    .unwrap();

    let event_c = append_session_event(
        &pool,
        session_id,
        &NewSessionEvent {
            kind: "assistant_message".to_owned(),
            turn_id: Some("turn_1".to_owned()),
            message_preview: Some("bye".to_owned()),
            artifact_id: None,
            payload: payload_c.clone(),
            created_at_ms: Some(30),
        },
    )
    .await
    .unwrap();

    let page_1 = get_session_events(&pool, session_id, &SessionEventsQuery::with_limit(2))
        .await
        .unwrap();
    assert_eq!(page_1, vec![event_a.clone(), event_b.clone()]);

    let cursor = SessionEventCursor {
        created_at_ms: page_1[1].created_at_ms,
        id: page_1[1].id,
    };

    let page_2 = get_session_events(
        &pool,
        session_id,
        &SessionEventsQuery {
            limit: 10,
            after: Some(cursor),
            kinds: Vec::new(),
        },
    )
    .await
    .unwrap();
    assert_eq!(page_2, vec![event_c.clone()]);

    let filtered = get_session_events(
        &pool,
        session_id,
        &SessionEventsQuery {
            limit: 10,
            after: None,
            kinds: vec!["assistant_message".to_owned()],
        },
    )
    .await
    .unwrap();

    assert_eq!(filtered, vec![event_a, event_c]);
}
