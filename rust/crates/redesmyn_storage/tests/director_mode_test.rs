use redesmyn_ids::{EpicId, RepoId, SessionId, WorkspaceId};
use redesmyn_storage::director_mode::{
    DirectorActivationIntent, DirectorModeLifecycle, DirectorModeTransition,
    MergeAuthorityPolicySource, clear_epic_merge_authority_override,
    load_or_default_director_mode_state, resolve_merge_authority_policy,
    set_epic_merge_authority_override, set_global_merge_authority_default,
    transition_director_mode_state,
};
use redesmyn_storage::open_test_sqlite_pool;
use redesmyn_storage::schema::AgentKind;
use redesmyn_storage::sessions::create_chat_session;

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
    .expect("insert workspace");
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
    .expect("insert repo");
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
    .expect("insert epic");
}

async fn seed_scope(pool: &sqlx::SqlitePool) -> (WorkspaceId, RepoId, EpicId) {
    let workspace_id = WorkspaceId::new();
    let repo_id = RepoId::new();
    let epic_id = EpicId::new();

    insert_workspace(pool, workspace_id).await;
    insert_repo(pool, repo_id, workspace_id).await;
    insert_epic(pool, epic_id, repo_id).await;

    (workspace_id, repo_id, epic_id)
}

async fn create_director_chat_session(
    pool: &sqlx::SqlitePool,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    epic_id: EpicId,
) -> SessionId {
    create_chat_session(
        pool,
        workspace_id,
        repo_id,
        Some(epic_id),
        AgentKind::Codex,
        Some("Director"),
    )
    .await
    .expect("create chat session")
}

#[tokio::test]
async fn director_mode_lifecycle_transitions_are_explicit() {
    let pool = open_test_sqlite_pool().await.expect("db");
    let (workspace_id, repo_id, epic_id) = seed_scope(&pool).await;
    let director_session_id =
        create_director_chat_session(&pool, workspace_id, repo_id, epic_id).await;

    let default_state = load_or_default_director_mode_state(&pool, epic_id)
        .await
        .expect("load default director mode");
    assert_eq!(default_state.lifecycle, DirectorModeLifecycle::Inactive);
    assert_eq!(default_state.director_session_id, None);

    let invalid = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::Activate {
            director_session_id,
        },
    )
    .await;
    assert!(invalid.is_err(), "activation must require explicit intent");

    let selected = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::SelectActivationIntent(
            DirectorActivationIntent::RunInCurrentSession,
        ),
    )
    .await
    .expect("select activation intent");
    assert_eq!(
        selected.activation_intent,
        Some(DirectorActivationIntent::RunInCurrentSession)
    );
    assert_eq!(selected.lifecycle, DirectorModeLifecycle::Inactive);

    let active = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::Activate {
            director_session_id,
        },
    )
    .await
    .expect("activate director mode");
    assert_eq!(active.lifecycle, DirectorModeLifecycle::Active);
    assert_eq!(active.director_session_id, Some(director_session_id));

    let paused = transition_director_mode_state(&pool, epic_id, &DirectorModeTransition::Pause)
        .await
        .expect("pause");
    assert_eq!(paused.lifecycle, DirectorModeLifecycle::Paused);

    let activate_while_paused = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::Activate {
            director_session_id,
        },
    )
    .await;
    assert!(
        activate_while_paused.is_err(),
        "activate must be rejected while paused"
    );

    let resume_required = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::RequireResume {
            reason: Some("session interrupted".to_string()),
        },
    )
    .await
    .expect("mark resume required");
    assert_eq!(
        resume_required.lifecycle,
        DirectorModeLifecycle::ResumeRequired
    );
    assert_eq!(
        resume_required.resume_required_reason.as_deref(),
        Some("session interrupted")
    );
    assert!(resume_required.resume_required_at_ms.is_some());

    let activate_while_resume_required = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::Activate {
            director_session_id,
        },
    )
    .await;
    assert!(
        activate_while_resume_required.is_err(),
        "activate must be rejected while resume_required"
    );

    let resumed = transition_director_mode_state(&pool, epic_id, &DirectorModeTransition::Resume)
        .await
        .expect("resume");
    assert_eq!(resumed.lifecycle, DirectorModeLifecycle::Active);
    assert_eq!(resumed.resume_required_reason, None);
    assert_eq!(resumed.resume_required_at_ms, None);

    let errored = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::MarkError {
            reason: Some("runtime failed".to_string()),
        },
    )
    .await
    .expect("mark error");
    assert_eq!(errored.lifecycle, DirectorModeLifecycle::Error);
    assert_eq!(errored.director_session_id, None);
    assert_eq!(errored.error_reason.as_deref(), Some("runtime failed"));
    assert!(errored.error_at_ms.is_some());
    assert_eq!(errored.resume_required_reason, None);
    assert_eq!(errored.resume_required_at_ms, None);

    let activate_while_error = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::Activate {
            director_session_id,
        },
    )
    .await;
    assert!(
        activate_while_error.is_err(),
        "activate must be rejected while error"
    );

    let deactivated =
        transition_director_mode_state(&pool, epic_id, &DirectorModeTransition::Deactivate)
            .await
            .expect("deactivate");
    assert_eq!(deactivated.lifecycle, DirectorModeLifecycle::Inactive);
    assert_eq!(deactivated.director_session_id, None);
    assert_eq!(deactivated.activation_intent, None);
    assert_eq!(deactivated.error_reason, None);
    assert_eq!(deactivated.error_at_ms, None);

    let reactivate_without_intent = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::Activate {
            director_session_id,
        },
    )
    .await;
    assert!(
        reactivate_without_intent.is_err(),
        "deactivate should clear activation intent"
    );
}

#[tokio::test]
async fn director_mode_rejects_invalid_pause_resume_and_error_transitions() {
    let pool = open_test_sqlite_pool().await.expect("db");
    let (_, _, epic_id) = seed_scope(&pool).await;

    let pause_inactive =
        transition_director_mode_state(&pool, epic_id, &DirectorModeTransition::Pause).await;
    assert!(pause_inactive.is_err(), "pause from inactive must fail");

    let resume_inactive =
        transition_director_mode_state(&pool, epic_id, &DirectorModeTransition::Resume).await;
    assert!(resume_inactive.is_err(), "resume from inactive must fail");

    let error_inactive = transition_director_mode_state(
        &pool,
        epic_id,
        &DirectorModeTransition::MarkError {
            reason: Some("unexpected".to_string()),
        },
    )
    .await;
    assert!(
        error_inactive.is_err(),
        "mark error from inactive must fail"
    );
}

#[tokio::test]
async fn merge_authority_policy_resolves_global_default_and_epic_override() {
    let pool = open_test_sqlite_pool().await.expect("db");
    let (_, _, epic_id) = seed_scope(&pool).await;

    let default_policy = resolve_merge_authority_policy(&pool, epic_id)
        .await
        .expect("resolve default policy");
    assert!(!default_policy.yolo_merge);
    assert_eq!(
        default_policy.source,
        MergeAuthorityPolicySource::GlobalDefault
    );

    set_global_merge_authority_default(&pool, true)
        .await
        .expect("set global default");
    let global_true = resolve_merge_authority_policy(&pool, epic_id)
        .await
        .expect("resolve global true");
    assert!(global_true.yolo_merge);
    assert_eq!(
        global_true.source,
        MergeAuthorityPolicySource::GlobalDefault
    );

    set_epic_merge_authority_override(&pool, epic_id, false)
        .await
        .expect("set epic override");
    let overridden = resolve_merge_authority_policy(&pool, epic_id)
        .await
        .expect("resolve override");
    assert!(!overridden.yolo_merge);
    assert_eq!(overridden.source, MergeAuthorityPolicySource::EpicOverride);

    clear_epic_merge_authority_override(&pool, epic_id)
        .await
        .expect("clear epic override");
    let fallback = resolve_merge_authority_policy(&pool, epic_id)
        .await
        .expect("resolve fallback");
    assert!(fallback.yolo_merge);
    assert_eq!(fallback.source, MergeAuthorityPolicySource::GlobalDefault);
}
