use redesmyn_ids::{SessionId, TaskId};
use sqlx::SqlitePool;

use redesmyn_protocol::client::ModelReasoningEffort;
use redesmyn_protocol::session::{
    CodexApprovalPolicy, CodexSandboxPolicy, PermissionsMode, SessionEventKind,
};
use redesmyn_protocol::{SessionEvent, Timestamp};
use redesmyn_storage::StorageError;

#[derive(Debug, Clone, Default)]
pub(crate) struct PolicySnapshotProjection {
    pub permissions_mode: Option<PermissionsMode>,
    pub codex_approval_policy: Option<CodexApprovalPolicy>,
    pub codex_sandbox_policy: Option<CodexSandboxPolicy>,
    pub model_id: Option<String>,
    pub model_reasoning_effort: Option<ModelReasoningEffort>,
}

pub(crate) async fn apply_session_event_to_agent_session_row(
    pool: &SqlitePool,
    event: &SessionEvent,
) -> Result<(), StorageError> {
    apply_runtime_session_projection(pool, event).await?;
    apply_policy_snapshot_projection(pool, event).await?;
    Ok(())
}

async fn apply_runtime_session_projection(
    pool: &SqlitePool,
    event: &SessionEvent,
) -> Result<(), StorageError> {
    let now_ms = timestamp_ms(event.created_at);

    let (status, ended_at_ms, status_is_runtime) = match &event.kind {
        SessionEventKind::TurnStarted(_) => (Some("running"), None, true),
        SessionEventKind::TurnCompleted(completed) => {
            if completed.error.is_some() {
                (Some("error"), None, true)
            } else {
                (Some("blocked"), None, true)
            }
        }
        SessionEventKind::SessionEnded(_) => (Some("stopped"), Some(now_ms), false),
        _ => (None, None, false),
    };

    let external_session_ref_json = match &event.kind {
        SessionEventKind::TurnStarted(ev) => ev
            .external_session_ref
            .as_ref()
            .and_then(|r| serde_json::to_string(r).ok()),
        SessionEventKind::TurnCompleted(ev) => ev
            .external_session_ref
            .as_ref()
            .and_then(|r| serde_json::to_string(r).ok()),
        _ => None,
    };

    if status.is_none() && ended_at_ms.is_none() && external_session_ref_json.is_none() {
        return Ok(());
    }

    let mut query = sqlx::QueryBuilder::new("UPDATE agent_sessions SET updated_at_ms = ");
    query.push_bind(now_ms);

    if let Some(status) = status {
        query.push(", status = ");
        if status_is_runtime {
            query.push("CASE WHEN ended_at_ms IS NULL THEN ");
            query.push_bind(status);
            query.push(" ELSE status END");
        } else {
            query.push_bind(status);
        }
    }

    if let Some(ended_at_ms) = ended_at_ms {
        query.push(", ended_at_ms = COALESCE(ended_at_ms, ");
        query.push_bind(ended_at_ms);
        query.push(")");
    }

    if let Some(external) = external_session_ref_json {
        query.push(", external_session_ref = ");
        query.push_bind(external);
    }

    query.push(" WHERE session_id = ");
    query.push_bind(event.session_id);

    query.build().execute(pool).await?;
    Ok(())
}

async fn apply_policy_snapshot_projection(
    pool: &SqlitePool,
    event: &SessionEvent,
) -> Result<(), StorageError> {
    let now_ms = timestamp_ms(event.created_at);

    let mut permissions_mode = None::<Option<String>>;
    let mut permissions_mode_observed_at_ms = None::<i64>;
    let mut codex_approval_policy = None::<Option<String>>;
    let mut codex_approval_policy_observed_at_ms = None::<i64>;
    let mut codex_sandbox_policy_json = None::<Option<String>>;
    let mut codex_sandbox_policy_observed_at_ms = None::<i64>;
    let mut model_id = None::<Option<String>>;
    let mut model_reasoning_effort = None::<Option<String>>;
    let mut model_observed_at_ms = None::<i64>;

    match &event.kind {
        SessionEventKind::PermissionsModeChanged(changed) => {
            if let Some(mode) = permissions_mode_to_db_value(changed.mode) {
                permissions_mode = Some(Some(mode.to_string()));
                permissions_mode_observed_at_ms = Some(now_ms);
            }
        }
        SessionEventKind::CodexApprovalPolicyChanged(changed) => match changed.approval_policy {
            Some(policy) => {
                if let Some(value) = codex_approval_policy_to_db_value(policy) {
                    codex_approval_policy = Some(Some(value.to_string()));
                    codex_approval_policy_observed_at_ms = Some(now_ms);
                }
            }
            None => {
                codex_approval_policy = Some(None);
                codex_approval_policy_observed_at_ms = Some(now_ms);
            }
        },
        SessionEventKind::CodexSandboxPolicyChanged(changed) => {
            match changed.sandbox_policy.as_ref() {
                Some(CodexSandboxPolicy::Unknown) => {}
                Some(policy) => {
                    codex_sandbox_policy_json =
                        Some(Some(serde_json::to_string(policy).map_err(|err| {
                            StorageError::InvalidData {
                                message: format!("failed to encode codex sandbox policy: {err}"),
                            }
                        })?));
                    codex_sandbox_policy_observed_at_ms = Some(now_ms);
                }
                None => {
                    codex_sandbox_policy_json = Some(None);
                    codex_sandbox_policy_observed_at_ms = Some(now_ms);
                }
            }
        }
        SessionEventKind::SessionModelChanged(changed) => {
            let normalized_model_id = changed.model_id.as_ref().and_then(|value| {
                let trimmed = value.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            });
            let normalized_reasoning_effort = changed
                .reasoning_effort
                .and_then(session_model_reasoning_effort_to_db_value)
                .map(str::to_string);
            model_id = Some(normalized_model_id);
            model_reasoning_effort = Some(normalized_reasoning_effort);
            model_observed_at_ms = Some(now_ms);
        }
        _ => {}
    }

    if permissions_mode_observed_at_ms.is_none()
        && codex_approval_policy_observed_at_ms.is_none()
        && codex_sandbox_policy_observed_at_ms.is_none()
        && model_observed_at_ms.is_none()
    {
        return Ok(());
    }

    sqlx::query(
        r#"
        INSERT INTO session_policy_projection (
            session_id,
            updated_at_ms,
            permissions_mode,
            permissions_mode_observed_at_ms,
            codex_approval_policy,
            codex_approval_policy_observed_at_ms,
            codex_sandbox_policy_json,
            codex_sandbox_policy_observed_at_ms,
            model_id,
            model_reasoning_effort,
            model_observed_at_ms
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
        )
        ON CONFLICT(session_id) DO UPDATE SET
            updated_at_ms = CASE
                WHEN excluded.updated_at_ms >= session_policy_projection.updated_at_ms
                    THEN excluded.updated_at_ms
                ELSE session_policy_projection.updated_at_ms
            END,
            permissions_mode = CASE
                WHEN excluded.permissions_mode_observed_at_ms IS NULL
                    THEN session_policy_projection.permissions_mode
                WHEN session_policy_projection.permissions_mode_observed_at_ms IS NULL
                    OR excluded.permissions_mode_observed_at_ms >= session_policy_projection.permissions_mode_observed_at_ms
                    THEN excluded.permissions_mode
                ELSE session_policy_projection.permissions_mode
            END,
            permissions_mode_observed_at_ms = CASE
                WHEN excluded.permissions_mode_observed_at_ms IS NULL
                    THEN session_policy_projection.permissions_mode_observed_at_ms
                WHEN session_policy_projection.permissions_mode_observed_at_ms IS NULL
                    OR excluded.permissions_mode_observed_at_ms >= session_policy_projection.permissions_mode_observed_at_ms
                    THEN excluded.permissions_mode_observed_at_ms
                ELSE session_policy_projection.permissions_mode_observed_at_ms
            END,
            codex_approval_policy = CASE
                WHEN excluded.codex_approval_policy_observed_at_ms IS NULL
                    THEN session_policy_projection.codex_approval_policy
                WHEN session_policy_projection.codex_approval_policy_observed_at_ms IS NULL
                    OR excluded.codex_approval_policy_observed_at_ms >= session_policy_projection.codex_approval_policy_observed_at_ms
                    THEN excluded.codex_approval_policy
                ELSE session_policy_projection.codex_approval_policy
            END,
            codex_approval_policy_observed_at_ms = CASE
                WHEN excluded.codex_approval_policy_observed_at_ms IS NULL
                    THEN session_policy_projection.codex_approval_policy_observed_at_ms
                WHEN session_policy_projection.codex_approval_policy_observed_at_ms IS NULL
                    OR excluded.codex_approval_policy_observed_at_ms >= session_policy_projection.codex_approval_policy_observed_at_ms
                    THEN excluded.codex_approval_policy_observed_at_ms
                ELSE session_policy_projection.codex_approval_policy_observed_at_ms
            END,
            codex_sandbox_policy_json = CASE
                WHEN excluded.codex_sandbox_policy_observed_at_ms IS NULL
                    THEN session_policy_projection.codex_sandbox_policy_json
                WHEN session_policy_projection.codex_sandbox_policy_observed_at_ms IS NULL
                    OR excluded.codex_sandbox_policy_observed_at_ms >= session_policy_projection.codex_sandbox_policy_observed_at_ms
                    THEN excluded.codex_sandbox_policy_json
                ELSE session_policy_projection.codex_sandbox_policy_json
            END,
            codex_sandbox_policy_observed_at_ms = CASE
                WHEN excluded.codex_sandbox_policy_observed_at_ms IS NULL
                    THEN session_policy_projection.codex_sandbox_policy_observed_at_ms
                WHEN session_policy_projection.codex_sandbox_policy_observed_at_ms IS NULL
                    OR excluded.codex_sandbox_policy_observed_at_ms >= session_policy_projection.codex_sandbox_policy_observed_at_ms
                    THEN excluded.codex_sandbox_policy_observed_at_ms
                ELSE session_policy_projection.codex_sandbox_policy_observed_at_ms
            END,
            model_id = CASE
                WHEN excluded.model_observed_at_ms IS NULL
                    THEN session_policy_projection.model_id
                WHEN session_policy_projection.model_observed_at_ms IS NULL
                    OR excluded.model_observed_at_ms >= session_policy_projection.model_observed_at_ms
                    THEN excluded.model_id
                ELSE session_policy_projection.model_id
            END,
            model_reasoning_effort = CASE
                WHEN excluded.model_observed_at_ms IS NULL
                    THEN session_policy_projection.model_reasoning_effort
                WHEN session_policy_projection.model_observed_at_ms IS NULL
                    OR excluded.model_observed_at_ms >= session_policy_projection.model_observed_at_ms
                    THEN excluded.model_reasoning_effort
                ELSE session_policy_projection.model_reasoning_effort
            END,
            model_observed_at_ms = CASE
                WHEN excluded.model_observed_at_ms IS NULL
                    THEN session_policy_projection.model_observed_at_ms
                WHEN session_policy_projection.model_observed_at_ms IS NULL
                    OR excluded.model_observed_at_ms >= session_policy_projection.model_observed_at_ms
                    THEN excluded.model_observed_at_ms
                ELSE session_policy_projection.model_observed_at_ms
            END
        "#,
    )
    .bind(event.session_id)
    .bind(now_ms)
    .bind(permissions_mode.flatten())
    .bind(permissions_mode_observed_at_ms)
    .bind(codex_approval_policy.flatten())
    .bind(codex_approval_policy_observed_at_ms)
    .bind(codex_sandbox_policy_json.flatten())
    .bind(codex_sandbox_policy_observed_at_ms)
    .bind(model_id.flatten())
    .bind(model_reasoning_effort.flatten())
    .bind(model_observed_at_ms)
    .execute(pool)
    .await?;
    Ok(())
}

pub(crate) async fn load_session_policy_projection(
    pool: &SqlitePool,
    session_id: SessionId,
) -> Result<Option<PolicySnapshotProjection>, StorageError> {
    let row: Option<(
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    )> = sqlx::query_as(
        r#"
        SELECT
            permissions_mode,
            codex_approval_policy,
            codex_sandbox_policy_json,
            model_id,
            model_reasoning_effort
        FROM session_policy_projection
        WHERE session_id = ?1
        LIMIT 1
        "#,
    )
    .bind(session_id)
    .fetch_optional(pool)
    .await?;

    let Some((
        permissions_mode_raw,
        codex_approval_policy_raw,
        codex_sandbox_policy_json,
        model_id,
        model_reasoning_effort_raw,
    )) = row
    else {
        return Ok(None);
    };

    let codex_sandbox_policy = codex_sandbox_policy_json
        .as_deref()
        .and_then(|value| serde_json::from_str::<CodexSandboxPolicy>(value).ok())
        .and_then(|value| (!matches!(value, CodexSandboxPolicy::Unknown)).then_some(value));

    Ok(Some(PolicySnapshotProjection {
        permissions_mode: permissions_mode_raw
            .as_deref()
            .and_then(permissions_mode_from_db_value),
        codex_approval_policy: codex_approval_policy_raw
            .as_deref()
            .and_then(codex_approval_policy_from_db_value),
        codex_sandbox_policy,
        model_id: model_id.and_then(|value| {
            let trimmed = value.trim().to_string();
            (!trimmed.is_empty()).then_some(trimmed)
        }),
        model_reasoning_effort: model_reasoning_effort_raw
            .as_deref()
            .and_then(model_reasoning_effort_from_db_value),
    }))
}

pub(crate) async fn load_task_last_seen_policy_projection(
    pool: &SqlitePool,
    task_id: TaskId,
) -> Result<PolicySnapshotProjection, StorageError> {
    let model_row: Option<(Option<String>, Option<String>)> = sqlx::query_as(
        r#"
        SELECT
            p.model_id,
            p.model_reasoning_effort
        FROM session_policy_projection p
        JOIN agent_sessions s ON s.session_id = p.session_id
        WHERE s.scope_kind = 'task'
          AND s.task_id = ?1
          AND p.model_observed_at_ms IS NOT NULL
        ORDER BY p.model_observed_at_ms DESC, s.created_at_ms DESC, s.session_id DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await?;

    let permissions_mode_raw: Option<String> = sqlx::query_scalar(
        r#"
        SELECT p.permissions_mode
        FROM session_policy_projection p
        JOIN agent_sessions s ON s.session_id = p.session_id
        WHERE s.scope_kind = 'task'
          AND s.task_id = ?1
          AND p.permissions_mode_observed_at_ms IS NOT NULL
        ORDER BY p.permissions_mode_observed_at_ms DESC, s.created_at_ms DESC, s.session_id DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await?;

    let codex_approval_policy_raw: Option<String> = sqlx::query_scalar(
        r#"
        SELECT p.codex_approval_policy
        FROM session_policy_projection p
        JOIN agent_sessions s ON s.session_id = p.session_id
        WHERE s.scope_kind = 'task'
          AND s.task_id = ?1
          AND p.codex_approval_policy_observed_at_ms IS NOT NULL
        ORDER BY p.codex_approval_policy_observed_at_ms DESC, s.created_at_ms DESC, s.session_id DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await?;

    let codex_sandbox_policy_json: Option<String> = sqlx::query_scalar(
        r#"
        SELECT p.codex_sandbox_policy_json
        FROM session_policy_projection p
        JOIN agent_sessions s ON s.session_id = p.session_id
        WHERE s.scope_kind = 'task'
          AND s.task_id = ?1
          AND p.codex_sandbox_policy_observed_at_ms IS NOT NULL
        ORDER BY p.codex_sandbox_policy_observed_at_ms DESC, s.created_at_ms DESC, s.session_id DESC
        LIMIT 1
        "#,
    )
    .bind(task_id)
    .fetch_optional(pool)
    .await?;

    let (model_id, model_reasoning_effort_raw) = model_row.unwrap_or((None, None));
    let codex_sandbox_policy = codex_sandbox_policy_json
        .as_deref()
        .and_then(|value| serde_json::from_str::<CodexSandboxPolicy>(value).ok())
        .and_then(|value| (!matches!(value, CodexSandboxPolicy::Unknown)).then_some(value));

    Ok(PolicySnapshotProjection {
        permissions_mode: permissions_mode_raw
            .as_deref()
            .and_then(permissions_mode_from_db_value),
        codex_approval_policy: codex_approval_policy_raw
            .as_deref()
            .and_then(codex_approval_policy_from_db_value),
        codex_sandbox_policy,
        model_id: model_id.and_then(|value| {
            let trimmed = value.trim().to_string();
            (!trimmed.is_empty()).then_some(trimmed)
        }),
        model_reasoning_effort: model_reasoning_effort_raw
            .as_deref()
            .and_then(model_reasoning_effort_from_db_value),
    })
}

fn permissions_mode_from_db_value(value: &str) -> Option<PermissionsMode> {
    match value {
        "ask" => Some(PermissionsMode::Ask),
        "auto_approve" => Some(PermissionsMode::AutoApprove),
        "deny" => Some(PermissionsMode::Deny),
        _ => None,
    }
}

fn codex_approval_policy_from_db_value(value: &str) -> Option<CodexApprovalPolicy> {
    match value {
        "untrusted" => Some(CodexApprovalPolicy::UnlessTrusted),
        "on_failure" => Some(CodexApprovalPolicy::OnFailure),
        "on_request" => Some(CodexApprovalPolicy::OnRequest),
        "never" => Some(CodexApprovalPolicy::Never),
        _ => None,
    }
}

fn model_reasoning_effort_from_db_value(value: &str) -> Option<ModelReasoningEffort> {
    match value {
        "minimal" => Some(ModelReasoningEffort::Minimal),
        "low" => Some(ModelReasoningEffort::Low),
        "medium" => Some(ModelReasoningEffort::Medium),
        "high" => Some(ModelReasoningEffort::High),
        "xhigh" => Some(ModelReasoningEffort::Xhigh),
        _ => None,
    }
}

fn permissions_mode_to_db_value(value: PermissionsMode) -> Option<&'static str> {
    match value {
        PermissionsMode::Ask => Some("ask"),
        PermissionsMode::AutoApprove => Some("auto_approve"),
        PermissionsMode::Deny => Some("deny"),
        PermissionsMode::Unknown => None,
    }
}

fn codex_approval_policy_to_db_value(value: CodexApprovalPolicy) -> Option<&'static str> {
    match value {
        CodexApprovalPolicy::UnlessTrusted => Some("untrusted"),
        CodexApprovalPolicy::OnFailure => Some("on_failure"),
        CodexApprovalPolicy::OnRequest => Some("on_request"),
        CodexApprovalPolicy::Never => Some("never"),
        CodexApprovalPolicy::Unknown => None,
    }
}

fn session_model_reasoning_effort_to_db_value(
    value: redesmyn_protocol::session::SessionModelReasoningEffort,
) -> Option<&'static str> {
    match value {
        redesmyn_protocol::session::SessionModelReasoningEffort::Minimal => Some("minimal"),
        redesmyn_protocol::session::SessionModelReasoningEffort::Low => Some("low"),
        redesmyn_protocol::session::SessionModelReasoningEffort::Medium => Some("medium"),
        redesmyn_protocol::session::SessionModelReasoningEffort::High => Some("high"),
        redesmyn_protocol::session::SessionModelReasoningEffort::Xhigh => Some("xhigh"),
        redesmyn_protocol::session::SessionModelReasoningEffort::Unknown => None,
    }
}

fn timestamp_ms(value: Timestamp) -> i64 {
    let nanos = value.into_offset_date_time().unix_timestamp_nanos();
    let ms = nanos / 1_000_000;
    i64::try_from(ms).unwrap_or_else(|_| if ms.is_negative() { i64::MIN } else { i64::MAX })
}
