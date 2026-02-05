use redesmyn_control_plane::ControlPlane;
use redesmyn_ids::{ArtifactId, RepoId, SessionEventId, WorkspaceId};
use redesmyn_protocol::artifacts::{ArtifactKind, ArtifactRef, StorageHint};
use redesmyn_protocol::session::{AssistantMessage, SessionEventKind, SessionScope};
use redesmyn_protocol::{SessionEvent, Timestamp};
use redesmyn_storage::schema::AgentKind as StorageAgentKind;

#[tokio::test]
async fn session_event_append_inserts_artifacts() {
    let control_plane = ControlPlane::open_test().await.expect("control plane");

    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let now_ms = 0_i64;

    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(workspace_id)
    .bind(now_ms)
    .bind(now_ms)
    .bind("test-workspace")
    .execute(control_plane.pool())
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
    .bind("test-repo")
    .bind("Test Repo")
    .execute(control_plane.pool())
    .await
    .expect("insert repository");

    let session_id = redesmyn_storage::sessions::create_chat_session(
        control_plane.pool(),
        workspace_id,
        repo_id,
        StorageAgentKind::Codex,
        None,
    )
    .await
    .expect("create chat session");

    let artifact_id = ArtifactId::new();
    let artifact = ArtifactRef {
        artifact_id,
        kind: ArtifactKind::Log,
        content_hash: None,
        byte_len: Some(5),
        mime: Some("text/plain".to_owned()),
        storage_hint: Some(StorageHint::BlobKey {
            blob_key: format!("artifact/{artifact_id}"),
        }),
    };

    let event = SessionEvent {
        session_event_id: SessionEventId::new(),
        created_at: Timestamp::now_utc(),
        scope: SessionScope::Chat,
        session_id,
        turn_id: None,
        kind: SessionEventKind::AssistantMessage(AssistantMessage {
            text: "hello".to_owned(),
            preview: "hello".to_owned(),
            full_text_artifact: Some(artifact.clone()),
        }),
    };

    control_plane
        .session_events()
        .append_session_event(&event)
        .await
        .expect("append session event");

    #[derive(Debug, sqlx::FromRow)]
    struct ArtifactRow {
        id: ArtifactId,
        scope_kind: String,
        scope_workspace_id: Option<WorkspaceId>,
        scope_repo_id: Option<RepoId>,
        kind: String,
        storage_hint: Option<String>,
    }

    let row: Option<ArtifactRow> = sqlx::query_as(
        r#"
        SELECT
            id,
            scope_kind,
            scope_workspace_id,
            scope_repo_id,
            kind,
            storage_hint
        FROM artifacts
        WHERE id = ?1
        "#,
    )
    .bind(artifact_id)
    .fetch_optional(control_plane.pool())
    .await
    .expect("select artifact row");

    let Some(row) = row else {
        panic!("expected artifact row to be inserted");
    };

    assert_eq!(row.id, artifact_id);
    assert_eq!(row.scope_kind, "repo");
    assert_eq!(row.scope_workspace_id, Some(workspace_id));
    assert_eq!(row.scope_repo_id, Some(repo_id));
    assert_eq!(row.kind, "log");
    assert!(
        row.storage_hint
            .as_deref()
            .unwrap_or_default()
            .contains("artifact/"),
        "expected storage hint to reference blob key"
    );
}
