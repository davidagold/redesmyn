use redesmyn_ids::TaskId;
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope};
use sqlx::SqlitePool;

#[derive(Debug, Clone)]
pub(crate) struct TaskStartBranchContext {
    pub task_branch_name: String,
    pub task_base_branch_name: Option<String>,
}

fn normalize_branch_name(value: Option<String>) -> Option<String> {
    value
        .map(|raw| raw.trim().to_string())
        .filter(|trimmed| !trimmed.is_empty())
}

pub(crate) async fn load_task_start_branch_context(
    pool: &SqlitePool,
    task_id: TaskId,
) -> Result<TaskStartBranchContext, ErrorEnvelope> {
    let row: Option<(
        Option<String>,
        Option<TaskId>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    )> = sqlx::query_as(
        r#"
        SELECT
            t.branch_name,
            p.id AS parent_task_id,
            p.local_ref AS parent_task_ref,
            p.state AS parent_task_state,
            p.merge_readiness AS parent_task_merge_readiness,
            p.branch_name AS parent_task_branch_name
        FROM tasks t
        LEFT JOIN tasks p ON p.id = t.parent_task_id
        WHERE t.id = ?1
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await
    .map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to load task branch startup context.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })?;

    let (
        task_branch_name,
        parent_task_id,
        parent_task_ref,
        parent_task_state,
        parent_task_merge_readiness,
        parent_task_branch_name,
    ) = row.ok_or_else(|| {
        ErrorEnvelope::new(
            ErrorCategory::NotFound,
            "Task not found while loading branch startup context.",
        )
        .with_detail(ErrorDetail::from([(
            "task_id".to_string(),
            task_id.to_string(),
        )]))
    })?;

    let task_branch_name = normalize_branch_name(task_branch_name).ok_or_else(|| {
        ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            "Task branch is required to start an agent session.",
        )
        .with_detail(ErrorDetail::from([(
            "task_id".to_string(),
            task_id.to_string(),
        )]))
    })?;

    let task_base_branch_name = if let Some(parent_task_id) = parent_task_id {
        let parent_state = parent_task_state.unwrap_or_else(|| "unknown".to_string());
        let parent_merge_readiness =
            parent_task_merge_readiness.unwrap_or_else(|| "unknown".to_string());
        let parent_is_ready = parent_state == "done" || parent_merge_readiness == "ready";
        let parent_task_ref = parent_task_ref.unwrap_or_else(|| parent_task_id.to_string());

        if !parent_is_ready {
            return Err(ErrorEnvelope::new(
                ErrorCategory::Conflict,
                "Parent task is not ready; wait before starting a stacked child task session.",
            )
            .with_detail(ErrorDetail::from([
                ("task_id".to_string(), task_id.to_string()),
                ("parent_task_id".to_string(), parent_task_id.to_string()),
                ("parent_task_ref".to_string(), parent_task_ref),
                ("parent_state".to_string(), parent_state),
                ("parent_merge_readiness".to_string(), parent_merge_readiness),
            ])));
        }

        Some(
            normalize_branch_name(parent_task_branch_name).ok_or_else(|| {
                ErrorEnvelope::new(
                    ErrorCategory::InvalidRequest,
                    "Parent task branch is required to start a stacked child task session.",
                )
                .with_detail(ErrorDetail::from([
                    ("task_id".to_string(), task_id.to_string()),
                    ("parent_task_id".to_string(), parent_task_id.to_string()),
                    ("parent_task_ref".to_string(), parent_task_ref),
                ]))
            })?,
        )
    } else {
        None
    };

    Ok(TaskStartBranchContext {
        task_branch_name,
        task_base_branch_name,
    })
}

#[cfg(test)]
mod tests {
    use super::load_task_start_branch_context;
    use crate::ControlPlane;
    use redesmyn_ids::{EpicId, RepoId, TaskId, WorkspaceId};
    use redesmyn_protocol::ErrorCategory;
    use sqlx::SqlitePool;

    async fn insert_workspace(pool: &SqlitePool, workspace_id: WorkspaceId) {
        sqlx::query(
            r#"
            INSERT INTO workspaces (id, created_at_ms, updated_at_ms, name)
            VALUES (?1, 1, 1, 'workspace')
            "#,
        )
        .bind(workspace_id)
        .execute(pool)
        .await
        .expect("insert workspace");
    }

    async fn insert_repo(pool: &SqlitePool, workspace_id: WorkspaceId, repo_id: RepoId) {
        sqlx::query(
            r#"
            INSERT INTO repositories (id, workspace_id, created_at_ms, updated_at_ms, slug, title)
            VALUES (?1, ?2, 2, 2, 'repo', 'Repo')
            "#,
        )
        .bind(repo_id)
        .bind(workspace_id)
        .execute(pool)
        .await
        .expect("insert repo");
    }

    async fn insert_epic(pool: &SqlitePool, repo_id: RepoId, epic_id: EpicId) {
        sqlx::query(
            r#"
            INSERT INTO epics (id, repo_id, created_at_ms, updated_at_ms, slug, title)
            VALUES (?1, ?2, 3, 3, 'gpui', 'GPUI')
            "#,
        )
        .bind(epic_id)
        .bind(repo_id)
        .execute(pool)
        .await
        .expect("insert epic");
    }

    async fn insert_task(
        pool: &SqlitePool,
        task_id: TaskId,
        epic_id: EpicId,
        local_ref: &str,
        parent_task_id: Option<TaskId>,
        branch_name: &str,
        state: &str,
        merge_readiness: &str,
    ) {
        sqlx::query(
            r#"
            INSERT INTO tasks (
                id, epic_id, parent_task_id, created_at_ms, updated_at_ms,
                local_ref, title, branch_name, merge_readiness, state
            )
            VALUES (?1, ?2, ?3, 4, 4, ?4, ?5, ?6, ?7, ?8)
            "#,
        )
        .bind(task_id)
        .bind(epic_id)
        .bind(parent_task_id)
        .bind(local_ref)
        .bind(format!("Task {local_ref}"))
        .bind(branch_name)
        .bind(merge_readiness)
        .bind(state)
        .execute(pool)
        .await
        .expect("insert task");
    }

    #[tokio::test]
    async fn stacked_child_requires_ready_parent() {
        let control_plane = ControlPlane::open_test().await.expect("control plane");
        let pool = control_plane.pool();
        let workspace_id = WorkspaceId::new();
        let repo_id = RepoId::new();
        let epic_id = EpicId::new();
        insert_workspace(pool, workspace_id).await;
        insert_repo(pool, workspace_id, repo_id).await;
        insert_epic(pool, repo_id, epic_id).await;

        let parent_task_id = TaskId::new();
        let child_task_id = TaskId::new();
        insert_task(
            pool,
            parent_task_id,
            epic_id,
            "T-29",
            None,
            "rn/gpui/T-29-merge-restack-planner",
            "in_progress",
            "unknown",
        )
        .await;
        insert_task(
            pool,
            child_task_id,
            epic_id,
            "T-30",
            Some(parent_task_id),
            "rn/gpui/T-30-merge-restack-executor",
            "todo",
            "unknown",
        )
        .await;

        let err = load_task_start_branch_context(pool, child_task_id)
            .await
            .expect_err("parent should block startup");
        assert_eq!(err.category, ErrorCategory::Conflict);
        assert!(err.message.contains("Parent task is not ready"));
    }

    #[tokio::test]
    async fn stacked_child_uses_parent_branch_as_base() {
        let control_plane = ControlPlane::open_test().await.expect("control plane");
        let pool = control_plane.pool();
        let workspace_id = WorkspaceId::new();
        let repo_id = RepoId::new();
        let epic_id = EpicId::new();
        insert_workspace(pool, workspace_id).await;
        insert_repo(pool, workspace_id, repo_id).await;
        insert_epic(pool, repo_id, epic_id).await;

        let parent_task_id = TaskId::new();
        let child_task_id = TaskId::new();
        insert_task(
            pool,
            parent_task_id,
            epic_id,
            "T-29",
            None,
            "rn/gpui/T-29-merge-restack-planner",
            "in_progress",
            "ready",
        )
        .await;
        insert_task(
            pool,
            child_task_id,
            epic_id,
            "T-30",
            Some(parent_task_id),
            "rn/gpui/T-30-merge-restack-executor",
            "todo",
            "unknown",
        )
        .await;

        let context = load_task_start_branch_context(pool, child_task_id)
            .await
            .expect("task start context");
        assert_eq!(
            context.task_branch_name,
            "rn/gpui/T-30-merge-restack-executor"
        );
        assert_eq!(
            context.task_base_branch_name.as_deref(),
            Some("rn/gpui/T-29-merge-restack-planner")
        );
    }
}
