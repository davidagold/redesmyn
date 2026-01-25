use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use redesmyn_ids::{EpicId, RepoId, SessionEventId, SessionId, TaskId, WorkspaceId};
use redesmyn_logging::tracing::{Instrument, info, warn};
use sqlx::{
    SqliteConnection, SqlitePool,
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
};
use ulid::Ulid;

use crate::{
    StorageError,
    schema::{MergeReadiness, SessionScopeKind},
    sqlite::open_sqlite_pool,
};

#[derive(Debug, Clone)]
struct RustSchemaFeatures {
    has_task_state: bool,
    agent_sessions: Option<AgentSessionsSchema>,
}

#[derive(Debug, Clone)]
struct AgentSessionsSchema {
    pk_column: String,
    required_columns: HashSet<String>,
    columns: HashSet<String>,
}

#[derive(Debug, Clone)]
pub struct ImportLegacyOptions {
    pub repo_root: PathBuf,
    pub legacy_db_path: PathBuf,
    pub rust_db_path: PathBuf,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct ImportLegacyCounts {
    pub epics: usize,
    pub tasks: usize,
    pub session_previews: usize,
}

#[derive(Debug, Clone)]
pub struct ImportLegacyApplied {
    pub workspace_id: WorkspaceId,
    pub repo_id: RepoId,
    pub upserted_workspaces: u64,
    pub upserted_repositories: u64,
    pub upserted_epics: u64,
    pub upserted_tasks: u64,
    pub upserted_session_events: u64,
}

#[derive(Debug, Clone)]
pub struct ImportLegacyOutcome {
    pub legacy_alembic_version: Option<String>,
    pub rust_sqlx_version_before: Option<i64>,
    pub rust_sqlx_version_after: Option<i64>,
    pub planned: ImportLegacyCounts,
    pub applied: Option<ImportLegacyApplied>,
}

#[derive(Debug, Clone)]
pub struct LegacyRepoKey {
    pub legacy_repository_id: i64,
    pub workspace_key: String,
    pub repo_key: String,
    pub repo_root: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct LegacyRepositoryRow {
    id: i64,
    workspace_id: String,
    repo_id: String,
    repo_root: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct LegacyEpicRow {
    id: i64,
    slug: String,
    name: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct LegacyTaskRow {
    id: i64,
    epic_id: i64,
    parent_task_id: Option<i64>,
    branch_name: Option<String>,
    title: String,
    local_path: Option<String>,
    merge_ready_at: Option<String>,
    state: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct LegacyAgentSessionRow {
    id: i64,
    task_id: i64,
    agent_preview: String,
}

#[derive(Debug, serde::Deserialize)]
struct LegacyAgentPreview {
    #[serde(default)]
    last_assistant_message_preview: Option<String>,
    #[serde(default)]
    last_message_turn_id: Option<String>,
    #[serde(default)]
    last_assistant_message_at: Option<String>,
}

#[derive(Debug, Clone)]
struct LegacySessionPreviewPlan {
    legacy_agent_session_id: i64,
    task_id: i64,
    message_preview: String,
    turn_id: Option<String>,
    created_at_ms: i64,
}

pub async fn import_legacy(options: ImportLegacyOptions) -> Result<ImportLegacyOutcome, StorageError> {
    let repo_root_display = options.repo_root.display().to_string();
    let legacy_db_display = options.legacy_db_path.display().to_string();
    let rust_db_display = options.rust_db_path.display().to_string();
    let dry_run = options.dry_run;

    async move {
        if !options.legacy_db_path.exists() {
            return Err(StorageError::LegacyDbNotFound {
                path: options.legacy_db_path.clone(),
            });
        }

        let rust_sqlx_version_before =
            read_sqlx_schema_version_if_present(&options.rust_db_path).await?;

        let now_ms = unix_epoch_ms_now();

        let legacy_snapshot = LegacyDbSnapshot::create(&options.legacy_db_path)?;
        let legacy_pool = open_legacy_pool_readonly(&legacy_snapshot.db_path).await?;

        let legacy_alembic_version = read_legacy_alembic_version(&legacy_pool).await?;

        let legacy_repo =
            read_legacy_repository_for_repo_root(&legacy_pool, &options.repo_root).await?;
        let legacy_repo_key = LegacyRepoKey {
            legacy_repository_id: legacy_repo.id,
            workspace_key: legacy_repo.workspace_id.clone(),
            repo_key: legacy_repo.repo_id.clone(),
            repo_root: legacy_repo.repo_root.clone(),
        };

        let legacy_epics = read_legacy_epics_for_repository(&legacy_pool, legacy_repo.id).await?;
        let legacy_tasks = read_legacy_tasks_for_epics(&legacy_pool, &legacy_epics).await?;
        let legacy_session_previews =
            plan_legacy_session_previews(&legacy_pool, &legacy_tasks, now_ms).await?;

        let planned = ImportLegacyCounts {
            epics: legacy_epics.len(),
            tasks: legacy_tasks.len(),
            session_previews: legacy_session_previews.len(),
        };

        info!(
            repo_root = %options.repo_root.display(),
            legacy_repo_id = %legacy_repo_key.repo_key,
            legacy_workspace_id = %legacy_repo_key.workspace_key,
            planned_epics = planned.epics,
            planned_tasks = planned.tasks,
            planned_session_previews = planned.session_previews,
            "legacy import plan built"
        );

        if options.dry_run {
            return Ok(ImportLegacyOutcome {
                legacy_alembic_version,
                rust_sqlx_version_before,
                rust_sqlx_version_after: rust_sqlx_version_before,
                planned,
                applied: None,
            });
        }

        let rust_pool = open_sqlite_pool(&options.rust_db_path).await?;
        let rust_sqlx_version_after = read_sqlx_schema_version(&rust_pool).await?;

        let applied = crate::in_transaction(&rust_pool, move |conn| {
            let legacy_repo_key = legacy_repo_key.clone();
            let legacy_repo = legacy_repo.clone();
            let legacy_epics = legacy_epics.clone();
            let legacy_tasks = legacy_tasks.clone();
            let legacy_session_previews = legacy_session_previews.clone();

            Box::pin(async move {
                import_into_rust_db(
                    conn,
                    now_ms,
                    legacy_repo_key,
                    legacy_repo,
                    legacy_epics,
                    legacy_tasks,
                    legacy_session_previews,
                )
                .await
            })
        })
        .await?;

        Ok(ImportLegacyOutcome {
            legacy_alembic_version,
            rust_sqlx_version_before,
            rust_sqlx_version_after,
            planned,
            applied: Some(applied),
        })
    }
    .instrument(redesmyn_logging::tracing::info_span!(
        "storage.import_legacy",
        repo_root = %repo_root_display,
        legacy_db = %legacy_db_display,
        rust_db = %rust_db_display,
        dry_run = dry_run,
    ))
    .await
}

async fn import_into_rust_db(
    conn: &mut SqliteConnection,
    now_ms: i64,
    legacy_repo_key: LegacyRepoKey,
    legacy_repo: LegacyRepositoryRow,
    legacy_epics: Vec<LegacyEpicRow>,
    legacy_tasks: Vec<LegacyTaskRow>,
    legacy_session_previews: Vec<LegacySessionPreviewPlan>,
) -> Result<ImportLegacyApplied, StorageError> {
    let mut upserted_workspaces = 0_u64;
    let mut upserted_repositories = 0_u64;
    let mut upserted_epics = 0_u64;
    let mut upserted_tasks = 0_u64;
    let mut upserted_session_events = 0_u64;

    let (workspace_id, workspace_inserted) =
        get_or_create_workspace(conn, now_ms, &legacy_repo.workspace_id).await?;
    if workspace_inserted {
        upserted_workspaces += 1;
    }

    let repo_title = repo_title_from_repo_root(&legacy_repo.repo_root, &legacy_repo.repo_id);

    let repo_ulid = get_or_create_mapped_ulid(conn, now_ms, "repositories", legacy_repo.id).await?;
    let mapped_repo_id = RepoId::from_ulid(repo_ulid);

    let repo_id = match find_repo_id(conn, workspace_id, &legacy_repo.repo_id).await? {
        Some(existing) if existing == mapped_repo_id => existing,
        Some(existing) => {
            return Err(StorageError::Conflict {
                message: format!(
                    "repository already exists in Rust DB with slug={slug} but legacy id map points to a different id (existing={existing}, mapped={mapped})",
                    slug = legacy_repo.repo_id,
                    existing = existing,
                    mapped = mapped_repo_id,
                ),
            });
        }
        None => {
            let rows = sqlx::query(
                r#"
                INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                "#,
            )
            .bind(mapped_repo_id)
            .bind(workspace_id)
            .bind(now_ms)
            .bind(now_ms)
            .bind(&legacy_repo.repo_id)
            .bind(&repo_title)
            .execute(&mut *conn)
            .await?
            .rows_affected();
            upserted_repositories += rows;
            mapped_repo_id
        }
    };

    // Keep repo metadata fresh on re-import.
    sqlx::query(
        r#"
        UPDATE repositories
        SET updated_at_ms = ?1, title = ?2
        WHERE id = ?3
        "#,
    )
    .bind(now_ms)
    .bind(&repo_title)
    .bind(repo_id)
    .execute(&mut *conn)
    .await?;

    let mut epic_id_by_legacy_id: HashMap<i64, EpicId> = HashMap::new();
    for epic in &legacy_epics {
        let epic_ulid = get_or_create_mapped_ulid(conn, now_ms, "epics", epic.id).await?;
        let mapped_epic_id = EpicId::from_ulid(epic_ulid);

        let epic_id = match find_epic_id(conn, repo_id, &epic.slug).await? {
            Some(existing) if existing == mapped_epic_id => existing,
            Some(existing) => {
                return Err(StorageError::Conflict {
                    message: format!(
                        "epic already exists in Rust DB with slug={slug} but legacy id map points to a different id (existing={existing}, mapped={mapped})",
                        slug = epic.slug,
                        existing = existing,
                        mapped = mapped_epic_id,
                    ),
                });
            }
            None => {
                let rows = sqlx::query(
                    r#"
                    INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                    "#,
                )
                .bind(mapped_epic_id)
                .bind(repo_id)
                .bind(now_ms)
                .bind(now_ms)
                .bind(&epic.slug)
                .bind(&epic.name)
                .execute(&mut *conn)
                .await?
                .rows_affected();
                upserted_epics += rows;
                mapped_epic_id
            }
        };

        // Keep epic metadata fresh on re-import.
        sqlx::query(
            r#"
            UPDATE epics
            SET updated_at_ms = ?1, title = ?2
            WHERE id = ?3
            "#,
        )
        .bind(now_ms)
        .bind(&epic.name)
        .bind(epic_id)
        .execute(&mut *conn)
        .await?;

        epic_id_by_legacy_id.insert(epic.id, epic_id);
    }

    let schema_features = detect_rust_schema_features(conn).await?;

    for task in &legacy_tasks {
        let epic_id = epic_id_by_legacy_id.get(&task.epic_id).copied().ok_or_else(|| {
            StorageError::InvalidData {
                message: format!(
                    "legacy task references missing epic_id: task_id={task_id} epic_id={epic_id}",
                    task_id = task.id,
                    epic_id = task.epic_id
                ),
            }
        })?;

        let task_ulid = get_or_create_mapped_ulid(conn, now_ms, "tasks", task.id).await?;
        let task_id = TaskId::from_ulid(task_ulid);

        let parent_task_id = match task.parent_task_id {
            Some(parent_legacy_id) => {
                let parent_ulid =
                    get_or_create_mapped_ulid(conn, now_ms, "tasks", parent_legacy_id).await?;
                Some(TaskId::from_ulid(parent_ulid))
            }
            None => None,
        };

        let local_ref = task
            .local_path
            .as_deref()
            .and_then(local_ref_from_legacy_local_path);

        let merge_readiness = merge_readiness_from_legacy(task);

        let rows = if schema_features.has_task_state {
            let state = task_state_from_legacy(task)?;
            sqlx::query(
                r#"
                INSERT INTO tasks (
                    id,
                    epic_id,
                    parent_task_id,
                    created_at_ms,
                    updated_at_ms,
                    local_ref,
                    title,
                    branch_name,
                    merge_readiness,
                    state
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT (id) DO UPDATE SET
                    epic_id = excluded.epic_id,
                    parent_task_id = excluded.parent_task_id,
                    updated_at_ms = excluded.updated_at_ms,
                    local_ref = excluded.local_ref,
                    title = excluded.title,
                    branch_name = excluded.branch_name,
                    merge_readiness = excluded.merge_readiness,
                    state = excluded.state
                "#,
            )
            .bind(task_id)
            .bind(epic_id)
            .bind(parent_task_id)
            .bind(now_ms)
            .bind(now_ms)
            .bind(local_ref.as_deref())
            .bind(&task.title)
            .bind(task.branch_name.as_deref())
            .bind(merge_readiness.as_str())
            .bind(state)
            .execute(&mut *conn)
            .await?
            .rows_affected()
        } else {
            sqlx::query(
                r#"
                INSERT INTO tasks (
                    id,
                    epic_id,
                    parent_task_id,
                    created_at_ms,
                    updated_at_ms,
                    local_ref,
                    title,
                    branch_name,
                    merge_readiness
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                ON CONFLICT (id) DO UPDATE SET
                    epic_id = excluded.epic_id,
                    parent_task_id = excluded.parent_task_id,
                    updated_at_ms = excluded.updated_at_ms,
                    local_ref = excluded.local_ref,
                    title = excluded.title,
                    branch_name = excluded.branch_name,
                    merge_readiness = excluded.merge_readiness
                "#,
            )
            .bind(task_id)
            .bind(epic_id)
            .bind(parent_task_id)
            .bind(now_ms)
            .bind(now_ms)
            .bind(local_ref.as_deref())
            .bind(&task.title)
            .bind(task.branch_name.as_deref())
            .bind(merge_readiness.as_str())
            .execute(&mut *conn)
            .await?
            .rows_affected()
        };
        upserted_tasks += rows;
    }

    // Import one best-effort "preview" session event per task (latest session in legacy).
    for preview in &legacy_session_previews {
        let task_ulid = get_or_create_mapped_ulid(conn, now_ms, "tasks", preview.task_id).await?;
        let task_id = TaskId::from_ulid(task_ulid);

        let task_epic_id: EpicId = sqlx::query_scalar(
            r#"
            SELECT epic_id FROM tasks WHERE id = ?1
            "#,
        )
        .bind(task_id)
        .fetch_one(&mut *conn)
        .await?;

        let session_ulid = get_or_create_mapped_ulid(
            conn,
            now_ms,
            "agent_sessions",
            preview.legacy_agent_session_id,
        )
        .await?;
        let session_id = SessionId::from_ulid(session_ulid);

        let session_event_ulid = get_or_create_mapped_ulid(
            conn,
            now_ms,
            "agent_session_preview_events",
            preview.legacy_agent_session_id,
        )
        .await?;
        let session_event_id = SessionEventId::from_ulid(session_event_ulid);

        if let Some(agent_sessions_schema) = &schema_features.agent_sessions {
            ensure_agent_session(
                conn,
                agent_sessions_schema,
                session_id,
                task_id,
                workspace_id,
                repo_id,
                now_ms,
            )
            .await?;
        }

        let rows = sqlx::query(
            r#"
            INSERT INTO session_events (
                id,
                session_id,
                created_at_ms,
                scope_kind,
                scope_workspace_id,
                scope_repo_id,
                epic_id,
                task_id,
                kind,
                turn_id,
                message_preview,
                artifact_id,
                payload
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, NULL, X'')
            ON CONFLICT (id) DO UPDATE SET
                created_at_ms = excluded.created_at_ms,
                scope_kind = excluded.scope_kind,
                scope_workspace_id = excluded.scope_workspace_id,
                scope_repo_id = excluded.scope_repo_id,
                epic_id = excluded.epic_id,
                task_id = excluded.task_id,
                kind = excluded.kind,
                turn_id = excluded.turn_id,
                message_preview = excluded.message_preview
            "#,
        )
        .bind(session_event_id)
        .bind(session_id)
        .bind(preview.created_at_ms)
        .bind(SessionScopeKind::Task.as_str())
        .bind(workspace_id)
        .bind(repo_id)
        .bind(task_epic_id)
        .bind(task_id)
        .bind("legacy.agent_preview")
        .bind(preview.turn_id.as_deref())
        .bind(&preview.message_preview)
        .execute(&mut *conn)
        .await?
        .rows_affected();
        upserted_session_events += rows;
    }

    info!(
        legacy_repo_id = %legacy_repo_key.repo_key,
        legacy_workspace_id = %legacy_repo_key.workspace_key,
        upserted_workspaces,
        upserted_repositories,
        upserted_epics,
        upserted_tasks,
        upserted_session_events,
        "legacy import applied"
    );

    Ok(ImportLegacyApplied {
        workspace_id,
        repo_id,
        upserted_workspaces,
        upserted_repositories,
        upserted_epics,
        upserted_tasks,
        upserted_session_events,
    })
}

async fn get_or_create_workspace(
    conn: &mut SqliteConnection,
    now_ms: i64,
    name: &str,
) -> Result<(WorkspaceId, bool), StorageError> {
    let existing: Option<WorkspaceId> = sqlx::query_scalar(
        r#"
        SELECT id FROM workspaces WHERE name = ?1 LIMIT 1
        "#,
    )
    .bind(name)
    .fetch_optional(&mut *conn)
    .await?;

    if let Some(id) = existing {
        return Ok((id, false));
    }

    let id = WorkspaceId::new();
    sqlx::query(
        r#"
        INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
        VALUES (?1, ?2, ?3, ?4)
        "#,
    )
    .bind(id)
    .bind(now_ms)
    .bind(now_ms)
    .bind(name)
    .execute(&mut *conn)
    .await?;

    Ok((id, true))
}

async fn find_repo_id(
    conn: &mut SqliteConnection,
    workspace_id: WorkspaceId,
    slug: &str,
) -> Result<Option<RepoId>, StorageError> {
    let existing: Option<RepoId> = sqlx::query_scalar(
        r#"
        SELECT id FROM repositories WHERE workspace_id = ?1 AND slug = ?2 LIMIT 1
        "#,
    )
    .bind(workspace_id)
    .bind(slug)
    .fetch_optional(&mut *conn)
    .await?;
    Ok(existing)
}

async fn find_epic_id(
    conn: &mut SqliteConnection,
    repo_id: RepoId,
    slug: &str,
) -> Result<Option<EpicId>, StorageError> {
    let existing: Option<EpicId> = sqlx::query_scalar(
        r#"
        SELECT id FROM epics WHERE repo_id = ?1 AND slug = ?2 LIMIT 1
        "#,
    )
    .bind(repo_id)
    .bind(slug)
    .fetch_optional(&mut *conn)
    .await?;
    Ok(existing)
}

fn merge_readiness_from_legacy(task: &LegacyTaskRow) -> MergeReadiness {
    if task.merge_ready_at.is_some() {
        return MergeReadiness::Ready;
    }
    if task.state == "blocked" {
        return MergeReadiness::Blocked;
    }
    MergeReadiness::Unknown
}

fn task_state_from_legacy(task: &LegacyTaskRow) -> Result<&'static str, StorageError> {
    match task.state.as_str() {
        "todo" => Ok("todo"),
        "in_progress" => Ok("in_progress"),
        "blocked" => Ok("blocked"),
        "done" => Ok("done"),
        other => Err(StorageError::InvalidData {
            message: format!("unsupported legacy task state: {other}"),
        }),
    }
}

fn local_ref_from_legacy_local_path(local_path: &str) -> Option<String> {
    let parts: Vec<&str> = local_path.split('/').collect();
    for (idx, part) in parts.iter().enumerate() {
        if *part == "tasks" {
            let candidate = parts.get(idx + 1)?;
            if candidate.starts_with("T-") {
                return Some(candidate.to_string());
            }
            return None;
        }
    }
    None
}

fn repo_title_from_repo_root(repo_root: &str, fallback: &str) -> String {
    Path::new(repo_root)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback.to_owned())
}

async fn read_legacy_repository_for_repo_root(
    pool: &SqlitePool,
    repo_root: &Path,
) -> Result<LegacyRepositoryRow, StorageError> {
    let repo_root_string = repo_root.display().to_string();

    let direct: Option<LegacyRepositoryRow> = sqlx::query_as(
        r#"
        SELECT id, workspace_id, repo_id, repo_root
        FROM repositories
        WHERE repo_root = ?1
        LIMIT 1
        "#,
    )
    .bind(&repo_root_string)
    .fetch_optional(pool)
    .await?;

    if let Some(row) = direct {
        return Ok(row);
    }

    // Fallback: most legacy DBs are repo-scoped and contain a single repo row.
    let repos: Vec<LegacyRepositoryRow> = sqlx::query_as(
        r#"
        SELECT id, workspace_id, repo_id, repo_root
        FROM repositories
        "#,
    )
    .fetch_all(pool)
    .await?;

    if repos.len() == 1 {
        warn!(
            repo_root = %repo_root_string,
            legacy_repo_root = %repos[0].repo_root,
            "legacy DB repository row did not match repo_root; falling back to the only repository row present"
        );
        return Ok(repos[0].clone());
    }

    Err(StorageError::LegacyRepoNotFound {
        repo_root: repo_root.to_path_buf(),
    })
}

async fn read_legacy_epics_for_repository(
    pool: &SqlitePool,
    legacy_repository_id: i64,
) -> Result<Vec<LegacyEpicRow>, StorageError> {
    let epics: Vec<LegacyEpicRow> = sqlx::query_as(
        r#"
        SELECT id, slug, name
        FROM epics
        WHERE repository_id = ?1
        ORDER BY id
        "#,
    )
    .bind(legacy_repository_id)
    .fetch_all(pool)
    .await?;
    Ok(epics)
}

async fn read_legacy_tasks_for_epics(
    pool: &SqlitePool,
    legacy_epics: &[LegacyEpicRow],
) -> Result<Vec<LegacyTaskRow>, StorageError> {
    let mut tasks = Vec::new();
    for epic in legacy_epics {
        let mut epic_tasks: Vec<LegacyTaskRow> = sqlx::query_as(
            r#"
            SELECT
                id,
                epic_id,
                parent_task_id,
                branch_name,
                title,
                local_path,
                merge_ready_at,
                state
            FROM tasks
            WHERE epic_id = ?1
            ORDER BY id
            "#,
        )
        .bind(epic.id)
        .fetch_all(pool)
        .await?;
        tasks.append(&mut epic_tasks);
    }
    Ok(tasks)
}

async fn plan_legacy_session_previews(
    pool: &SqlitePool,
    legacy_tasks: &[LegacyTaskRow],
    now_ms: i64,
) -> Result<Vec<LegacySessionPreviewPlan>, StorageError> {
    let task_ids: HashSet<i64> = legacy_tasks.iter().map(|t| t.id).collect();
    if task_ids.is_empty() {
        return Ok(Vec::new());
    }

    let sessions: Vec<LegacyAgentSessionRow> =
        sqlx::query_as(r#"SELECT id, task_id, agent_preview FROM agent_sessions ORDER BY id"#)
            .fetch_all(pool)
            .await?;

    let mut best_by_task_id: HashMap<i64, LegacySessionPreviewPlan> = HashMap::new();
    for row in sessions {
        if !task_ids.contains(&row.task_id) {
            continue;
        }

        let preview: LegacyAgentPreview = match serde_json::from_str(&row.agent_preview) {
            Ok(preview) => preview,
            Err(_) => continue,
        };

        let message_preview = match preview.last_assistant_message_preview {
            Some(text) if !text.trim().is_empty() => text,
            _ => continue,
        };

        let created_at_ms = preview
            .last_assistant_message_at
            .as_deref()
            .and_then(parse_rfc3339_ms)
            .unwrap_or(now_ms);

        let plan = LegacySessionPreviewPlan {
            legacy_agent_session_id: row.id,
            task_id: row.task_id,
            message_preview,
            turn_id: preview.last_message_turn_id,
            created_at_ms,
        };

        match best_by_task_id.get(&row.task_id) {
            Some(existing) if existing.legacy_agent_session_id >= row.id => {}
            _ => {
                best_by_task_id.insert(row.task_id, plan);
            }
        }
    }

    let mut planned: Vec<LegacySessionPreviewPlan> = best_by_task_id.into_values().collect();
    planned.sort_by_key(|p| p.task_id);
    Ok(planned)
}

fn parse_rfc3339_ms(value: &str) -> Option<i64> {
    // Keep parsing lightweight: accept the common `"2026-01-20T12:34:56.789Z"` shape.
    // If it fails, fall back to "now" without failing the import.
    let parsed = time::OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339)
        .ok()?;
    let millis = parsed.unix_timestamp_nanos() / 1_000_000;
    i64::try_from(millis).ok()
}

fn unix_epoch_ms_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

async fn read_legacy_alembic_version(pool: &SqlitePool) -> Result<Option<String>, StorageError> {
    if !sqlite_table_exists(pool, "alembic_version").await? {
        return Ok(None);
    }
    let version: Option<String> =
        sqlx::query_scalar(r#"SELECT version_num FROM alembic_version LIMIT 1"#)
            .fetch_optional(pool)
            .await?;
    Ok(version)
}

async fn read_sqlx_schema_version_if_present(
    db_path: &Path,
) -> Result<Option<i64>, StorageError> {
    if !db_path.exists() {
        return Ok(None);
    }

    let pool = open_sqlite_pool_readonly(db_path).await?;
    read_sqlx_schema_version(&pool).await
}

async fn read_sqlx_schema_version(pool: &SqlitePool) -> Result<Option<i64>, StorageError> {
    if !sqlite_table_exists(pool, "_sqlx_migrations").await? {
        return Ok(None);
    }

    let version: Option<i64> = sqlx::query_scalar(r#"SELECT MAX(version) FROM _sqlx_migrations"#)
        .fetch_one(pool)
        .await?;
    Ok(version)
}

async fn sqlite_table_exists(pool: &SqlitePool, table_name: &str) -> Result<bool, StorageError> {
    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?1
        LIMIT 1
        "#,
    )
    .bind(table_name)
    .fetch_optional(pool)
    .await?;
    Ok(found.is_some())
}

async fn open_legacy_pool_readonly(db_path: &Path) -> Result<SqlitePool, StorageError> {
    let connect_options = SqliteConnectOptions::new()
        .filename(db_path)
        .read_only(true)
        .create_if_missing(false)
        .foreign_keys(false);

    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(connect_options)
        .await?;

    Ok(pool)
}

async fn open_sqlite_pool_readonly(db_path: &Path) -> Result<SqlitePool, StorageError> {
    let connect_options = SqliteConnectOptions::new()
        .filename(db_path)
        .read_only(true)
        .create_if_missing(false)
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(connect_options)
        .await?;

    Ok(pool)
}

async fn get_or_create_mapped_ulid(
    conn: &mut SqliteConnection,
    now_ms: i64,
    table_name: &str,
    legacy_id: i64,
) -> Result<Ulid, StorageError> {
    if let Some(existing_bytes) = sqlx::query_scalar::<_, Vec<u8>>(
        r#"
        SELECT new_id FROM legacy_id_map WHERE table_name = ?1 AND legacy_id = ?2
        "#,
    )
    .bind(table_name)
    .bind(legacy_id)
    .fetch_optional(&mut *conn)
    .await?
    {
        return ulid_from_bytes(&existing_bytes);
    }

    const MAX_ATTEMPTS: usize = 8;
    for _ in 0..MAX_ATTEMPTS {
        let candidate = Ulid::new();
        let candidate_bytes = candidate.to_bytes().to_vec();

        let inserted = sqlx::query(
            r#"
            INSERT OR IGNORE INTO legacy_id_map (table_name, legacy_id, new_id, created_at_ms)
            VALUES (?1, ?2, ?3, ?4)
            "#,
        )
        .bind(table_name)
        .bind(legacy_id)
        .bind(candidate_bytes)
        .bind(now_ms)
        .execute(&mut *conn)
        .await?
        .rows_affected()
            > 0;

        if inserted {
            return Ok(candidate);
        }

        if let Some(stored_bytes) = sqlx::query_scalar::<_, Vec<u8>>(
            r#"
            SELECT new_id FROM legacy_id_map WHERE table_name = ?1 AND legacy_id = ?2
            "#,
        )
        .bind(table_name)
        .bind(legacy_id)
        .fetch_optional(&mut *conn)
        .await?
        {
            return ulid_from_bytes(&stored_bytes);
        }
    }

    Err(StorageError::Conflict {
        message: format!(
            "failed to allocate unique legacy id mapping for {table_name}:{legacy_id}"
        ),
    })
}

fn ulid_from_bytes(bytes: &[u8]) -> Result<Ulid, StorageError> {
    let array: [u8; 16] = bytes.try_into().map_err(|_| StorageError::InvalidData {
        message: format!("expected ULID bytes length 16, got {}", bytes.len()),
    })?;
    Ok(Ulid::from_bytes(array))
}

struct LegacyDbSnapshot {
    _dir: tempfile::TempDir,
    db_path: PathBuf,
}

impl LegacyDbSnapshot {
    fn create(legacy_db_path: &Path) -> Result<Self, StorageError> {
        let dir = tempfile::tempdir().map_err(|err| StorageError::LegacyDbSnapshot {
            path: legacy_db_path.to_path_buf(),
            source: err,
        })?;

        let file_name = legacy_db_path.file_name().ok_or_else(|| StorageError::InvalidData {
            message: format!(
                "legacy DB path does not have a file name: {}",
                legacy_db_path.display()
            ),
        })?;

        let snapshot_db_path = dir.path().join(file_name);

        std::fs::copy(legacy_db_path, &snapshot_db_path).map_err(|err| StorageError::LegacyDbSnapshot {
            path: legacy_db_path.to_path_buf(),
            source: err,
        })?;

        for suffix in ["-wal", "-shm"] {
            let original = sqlite_sidecar_path(legacy_db_path, suffix);
            if !original.exists() {
                continue;
            }
            let snapshot = sqlite_sidecar_path(&snapshot_db_path, suffix);
            std::fs::copy(&original, &snapshot).map_err(|err| StorageError::LegacyDbSnapshot {
                path: original,
                source: err,
            })?;
        }

        Ok(Self {
            _dir: dir,
            db_path: snapshot_db_path,
        })
    }
}

fn sqlite_sidecar_path(db_path: &Path, suffix: &str) -> PathBuf {
    let mut buf = OsString::from(db_path.as_os_str());
    buf.push(suffix);
    PathBuf::from(buf)
}

#[derive(Debug, sqlx::FromRow)]
struct SqliteTableInfoRow {
    name: String,
    #[sqlx(rename = "notnull")]
    not_null: i64,
    dflt_value: Option<String>,
    pk: i64,
}

async fn detect_rust_schema_features(
    conn: &mut SqliteConnection,
) -> Result<RustSchemaFeatures, StorageError> {
    let has_task_state = sqlite_column_exists(conn, "tasks", "state").await?;

    let agent_sessions = if sqlite_table_exists_conn(conn, "agent_sessions").await? {
        Some(load_agent_sessions_schema(conn).await?)
    } else {
        None
    };

    Ok(RustSchemaFeatures {
        has_task_state,
        agent_sessions,
    })
}

async fn sqlite_table_exists_conn(
    conn: &mut SqliteConnection,
    table_name: &str,
) -> Result<bool, StorageError> {
    let found: Option<i64> = sqlx::query_scalar(
        r#"
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?1
        LIMIT 1
        "#,
    )
    .bind(table_name)
    .fetch_optional(&mut *conn)
    .await?;
    Ok(found.is_some())
}

async fn sqlite_column_exists(
    conn: &mut SqliteConnection,
    table_name: &str,
    column_name: &str,
) -> Result<bool, StorageError> {
    if !table_name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(StorageError::InvalidData {
            message: format!("invalid sqlite table name: {table_name}"),
        });
    }

    let query = format!(
        "SELECT 1 FROM pragma_table_info('{table_name}') WHERE name = ?1 LIMIT 1"
    );

    let found: Option<i64> = sqlx::query_scalar(&query)
        .bind(column_name)
        .fetch_optional(&mut *conn)
        .await?;
    Ok(found.is_some())
}

async fn load_agent_sessions_schema(
    conn: &mut SqliteConnection,
) -> Result<AgentSessionsSchema, StorageError> {
    let rows: Vec<SqliteTableInfoRow> = sqlx::query_as(
        r#"
        SELECT name, "notnull" AS "notnull", dflt_value, pk
        FROM pragma_table_info('agent_sessions')
        "#,
    )
    .fetch_all(&mut *conn)
    .await?;

    let mut columns = HashSet::new();
    let mut required_columns = HashSet::new();
    let mut pk_columns: Vec<(i64, String)> = Vec::new();

    for row in rows {
        columns.insert(row.name.clone());
        if row.not_null != 0 && row.dflt_value.is_none() {
            required_columns.insert(row.name.clone());
        }
        if row.pk != 0 {
            pk_columns.push((row.pk, row.name));
        }
    }

    pk_columns.sort_by_key(|(idx, _)| *idx);
    let pk_column = match pk_columns.as_slice() {
        [(_, name)] => name.clone(),
        [] => {
            return Err(StorageError::InvalidData {
                message: "agent_sessions table has no primary key".to_owned(),
            });
        }
        _ => {
            return Err(StorageError::InvalidData {
                message: "agent_sessions table has a composite primary key; legacy importer expects a single-column primary key".to_owned(),
            });
        }
    };

    Ok(AgentSessionsSchema {
        pk_column,
        required_columns,
        columns,
    })
}

async fn ensure_agent_session(
    conn: &mut SqliteConnection,
    schema: &AgentSessionsSchema,
    session_id: SessionId,
    task_id: TaskId,
    workspace_id: WorkspaceId,
    repo_id: RepoId,
    now_ms: i64,
) -> Result<(), StorageError> {
    let mut missing = Vec::new();
    for column in &schema.required_columns {
        if agent_session_value_kind(schema, column).is_none() {
            missing.push(column.clone());
        }
    }
    if !missing.is_empty() {
        missing.sort();
        return Err(StorageError::InvalidData {
            message: format!(
                "agent_sessions schema has required columns the importer does not know how to populate: {missing:?}"
            ),
        });
    }

    let mut insert_columns = Vec::new();
    insert_columns.push(schema.pk_column.clone());

    for column in [
        "task_id",
        "scope_kind",
        "scope_workspace_id",
        "scope_repo_id",
        "workspace_id",
        "repo_id",
        "agent_kind",
        "interface_mode",
        "status",
        "created_at_ms",
        "updated_at_ms",
        "ended_at_ms",
    ] {
        if schema.columns.contains(column) && column != schema.pk_column {
            insert_columns.push(column.to_owned());
        }
    }

    let placeholders: Vec<String> = (1..=insert_columns.len())
        .map(|idx| format!("?{idx}"))
        .collect();

    let sql = format!(
        "INSERT OR IGNORE INTO agent_sessions ({cols}) VALUES ({values})",
        cols = insert_columns.join(", "),
        values = placeholders.join(", "),
    );

    let mut query = sqlx::query(&sql);
    for column in &insert_columns {
        match agent_session_value_kind(schema, column) {
            Some(AgentSessionValueKind::SessionId) => {
                query = query.bind(session_id);
            }
            Some(AgentSessionValueKind::TaskId) => {
                query = query.bind(task_id);
            }
            Some(AgentSessionValueKind::ScopeKindTask) => {
                query = query.bind("task");
            }
            Some(AgentSessionValueKind::WorkspaceId) => {
                query = query.bind(workspace_id);
            }
            Some(AgentSessionValueKind::RepoId) => {
                query = query.bind(repo_id);
            }
            Some(AgentSessionValueKind::AgentKindShell) => {
                query = query.bind("shell");
            }
            Some(AgentSessionValueKind::InterfaceModeShellTmux) => {
                query = query.bind("shell_tmux");
            }
            Some(AgentSessionValueKind::StatusStopped) => {
                query = query.bind("stopped");
            }
            Some(AgentSessionValueKind::TimestampNowMs) => {
                query = query.bind(now_ms);
            }
            None => {
                return Err(StorageError::InvalidData {
                    message: format!(
                        "agent_sessions column unexpectedly selected for insert but has no value mapping: {column}"
                    ),
                });
            }
        }
    }

    query.execute(&mut *conn).await?;

    let pk_column = &schema.pk_column;
    if !pk_column
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(StorageError::InvalidData {
            message: format!("invalid agent_sessions primary key column: {pk_column}"),
        });
    }

    let exists_sql = format!(
        "SELECT 1 FROM agent_sessions WHERE {pk_column} = ?1 LIMIT 1"
    );
    let exists: Option<i64> = sqlx::query_scalar(&exists_sql)
        .bind(session_id)
        .fetch_optional(&mut *conn)
        .await?;

    if exists.is_none() {
        return Err(StorageError::InvalidData {
            message:
                "agent_sessions insert was ignored and the row does not exist; importer could not satisfy constraints for this schema".to_owned(),
        });
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum AgentSessionValueKind {
    SessionId,
    TaskId,
    ScopeKindTask,
    WorkspaceId,
    RepoId,
    AgentKindShell,
    InterfaceModeShellTmux,
    StatusStopped,
    TimestampNowMs,
}

fn agent_session_value_kind(schema: &AgentSessionsSchema, column: &str) -> Option<AgentSessionValueKind> {
    if column == schema.pk_column {
        return Some(AgentSessionValueKind::SessionId);
    }

    match column {
        "task_id" => Some(AgentSessionValueKind::TaskId),
        "scope_kind" => Some(AgentSessionValueKind::ScopeKindTask),
        "scope_workspace_id" => Some(AgentSessionValueKind::WorkspaceId),
        "scope_repo_id" => Some(AgentSessionValueKind::RepoId),
        "workspace_id" if !schema.columns.contains("scope_workspace_id") => {
            Some(AgentSessionValueKind::WorkspaceId)
        }
        "repo_id" if !schema.columns.contains("scope_repo_id") => Some(AgentSessionValueKind::RepoId),
        "agent_kind" => Some(AgentSessionValueKind::AgentKindShell),
        "interface_mode" => Some(AgentSessionValueKind::InterfaceModeShellTmux),
        "status" => Some(AgentSessionValueKind::StatusStopped),
        "created_at_ms" | "updated_at_ms" | "ended_at_ms" => Some(AgentSessionValueKind::TimestampNowMs),
        _ => None,
    }
}
