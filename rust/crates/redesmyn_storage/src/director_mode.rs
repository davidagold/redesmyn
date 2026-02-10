use std::time::{SystemTime, UNIX_EPOCH};

use redesmyn_ids::{EpicId, SessionId};
use sqlx::{Executor, Sqlite};

use crate::StorageError;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectorModeLifecycle {
    Inactive,
    Active,
    Paused,
    ResumeRequired,
    Error,
}

impl DirectorModeLifecycle {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inactive => "inactive",
            Self::Active => "active",
            Self::Paused => "paused",
            Self::ResumeRequired => "resume_required",
            Self::Error => "error",
        }
    }

    fn from_db_value(value: &str) -> Result<Self, StorageError> {
        match value {
            "inactive" => Ok(Self::Inactive),
            "active" => Ok(Self::Active),
            "paused" => Ok(Self::Paused),
            "resume_required" => Ok(Self::ResumeRequired),
            "error" => Ok(Self::Error),
            _ => Err(StorageError::InvalidData {
                message: format!("unknown director_mode_state.lifecycle={value}"),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectorActivationIntent {
    RunInCurrentSession,
    RunInNewSession,
}

impl DirectorActivationIntent {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RunInCurrentSession => "run_in_current_session",
            Self::RunInNewSession => "run_in_new_session",
        }
    }

    fn from_db_value(value: &str) -> Result<Self, StorageError> {
        match value {
            "run_in_current_session" => Ok(Self::RunInCurrentSession),
            "run_in_new_session" => Ok(Self::RunInNewSession),
            _ => Err(StorageError::InvalidData {
                message: format!("unknown director_mode_state.activation_intent={value}"),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectorModeStateRecord {
    pub epic_id: EpicId,
    pub updated_at_ms: i64,
    pub lifecycle: DirectorModeLifecycle,
    pub director_session_id: Option<SessionId>,
    pub activation_intent: Option<DirectorActivationIntent>,
    pub resume_required_reason: Option<String>,
    pub resume_required_at_ms: Option<i64>,
}

impl DirectorModeStateRecord {
    #[must_use]
    pub fn inactive(epic_id: EpicId) -> Self {
        Self {
            epic_id,
            updated_at_ms: 0,
            lifecycle: DirectorModeLifecycle::Inactive,
            director_session_id: None,
            activation_intent: None,
            resume_required_reason: None,
            resume_required_at_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectorModeTransition {
    SelectActivationIntent(DirectorActivationIntent),
    Activate { director_session_id: SessionId },
    Pause,
    Resume,
    RequireResume { reason: Option<String> },
    MarkError { reason: Option<String> },
    Deactivate,
}

fn invalid_transition(
    lifecycle: DirectorModeLifecycle,
    transition: &DirectorModeTransition,
) -> StorageError {
    StorageError::InvalidData {
        message: format!(
            "invalid director mode transition from {} via {:?}",
            lifecycle.as_str(),
            transition
        ),
    }
}

pub fn apply_director_mode_transition(
    current: DirectorModeStateRecord,
    transition: &DirectorModeTransition,
    now_ms: i64,
) -> Result<DirectorModeStateRecord, StorageError> {
    let mut next = current;
    next.updated_at_ms = now_ms;

    match transition {
        DirectorModeTransition::SelectActivationIntent(intent) => {
            next.activation_intent = Some(*intent);
        }
        DirectorModeTransition::Activate {
            director_session_id,
        } => {
            if next.activation_intent.is_none() {
                return Err(StorageError::InvalidData {
                    message: "director activation requires an explicit activation intent selection"
                        .to_string(),
                });
            }
            if matches!(next.lifecycle, DirectorModeLifecycle::Active) {
                return Err(invalid_transition(next.lifecycle, transition));
            }

            next.lifecycle = DirectorModeLifecycle::Active;
            next.director_session_id = Some(*director_session_id);
            next.resume_required_reason = None;
            next.resume_required_at_ms = None;
        }
        DirectorModeTransition::Pause => {
            if !matches!(next.lifecycle, DirectorModeLifecycle::Active) {
                return Err(invalid_transition(next.lifecycle, transition));
            }

            next.lifecycle = DirectorModeLifecycle::Paused;
        }
        DirectorModeTransition::Resume => {
            if !matches!(
                next.lifecycle,
                DirectorModeLifecycle::Paused | DirectorModeLifecycle::ResumeRequired
            ) {
                return Err(invalid_transition(next.lifecycle, transition));
            }
            if next.director_session_id.is_none() {
                return Err(StorageError::InvalidData {
                    message: "director resume requires a director session binding".to_string(),
                });
            }

            next.lifecycle = DirectorModeLifecycle::Active;
            next.resume_required_reason = None;
            next.resume_required_at_ms = None;
        }
        DirectorModeTransition::RequireResume { reason } => {
            if !matches!(
                next.lifecycle,
                DirectorModeLifecycle::Active | DirectorModeLifecycle::Paused
            ) {
                return Err(invalid_transition(next.lifecycle, transition));
            }

            next.lifecycle = DirectorModeLifecycle::ResumeRequired;
            next.resume_required_reason = reason
                .as_ref()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            next.resume_required_at_ms = Some(now_ms);
        }
        DirectorModeTransition::MarkError { reason: _ } => {
            next.lifecycle = DirectorModeLifecycle::Error;
            next.resume_required_reason = None;
            next.resume_required_at_ms = None;
        }
        DirectorModeTransition::Deactivate => {
            next.lifecycle = DirectorModeLifecycle::Inactive;
            next.director_session_id = None;
            next.resume_required_reason = None;
            next.resume_required_at_ms = None;
        }
    }

    Ok(next)
}

type DirectorModeStateRow = (
    i64,
    String,
    Option<SessionId>,
    Option<String>,
    Option<String>,
    Option<i64>,
);

fn decode_director_mode_state_row(
    epic_id: EpicId,
    (
        updated_at_ms,
        lifecycle_raw,
        director_session_id,
        activation_intent_raw,
        resume_required_reason,
        resume_required_at_ms,
    ): DirectorModeStateRow,
) -> Result<DirectorModeStateRecord, StorageError> {
    Ok(DirectorModeStateRecord {
        epic_id,
        updated_at_ms,
        lifecycle: DirectorModeLifecycle::from_db_value(&lifecycle_raw)?,
        director_session_id,
        activation_intent: activation_intent_raw
            .as_deref()
            .map(DirectorActivationIntent::from_db_value)
            .transpose()?,
        resume_required_reason,
        resume_required_at_ms,
    })
}

pub async fn load_or_default_director_mode_state<'e, E>(
    executor: E,
    epic_id: EpicId,
) -> Result<DirectorModeStateRecord, StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row: Option<DirectorModeStateRow> = sqlx::query_as(
        r#"
        SELECT
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms
        FROM director_mode_state
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_optional(executor)
    .await?;

    row.map(|value| decode_director_mode_state_row(epic_id, value))
        .transpose()
        .map(|value| value.unwrap_or_else(|| DirectorModeStateRecord::inactive(epic_id)))
}

pub async fn set_director_mode_state<'e, E>(
    executor: E,
    state: &DirectorModeStateRecord,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO director_mode_state (
            epic_id,
            updated_at_ms,
            lifecycle,
            director_session_id,
            activation_intent,
            resume_required_reason,
            resume_required_at_ms
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(epic_id) DO UPDATE SET
            updated_at_ms = excluded.updated_at_ms,
            lifecycle = excluded.lifecycle,
            director_session_id = excluded.director_session_id,
            activation_intent = excluded.activation_intent,
            resume_required_reason = excluded.resume_required_reason,
            resume_required_at_ms = excluded.resume_required_at_ms
        "#,
    )
    .bind(state.epic_id)
    .bind(state.updated_at_ms)
    .bind(state.lifecycle.as_str())
    .bind(state.director_session_id)
    .bind(
        state
            .activation_intent
            .map(DirectorActivationIntent::as_str),
    )
    .bind(&state.resume_required_reason)
    .bind(state.resume_required_at_ms)
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn transition_director_mode_state<'e, E>(
    executor: E,
    epic_id: EpicId,
    transition: &DirectorModeTransition,
) -> Result<DirectorModeStateRecord, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let current = load_or_default_director_mode_state(executor, epic_id).await?;
    let next = apply_director_mode_transition(current, transition, now_ms())?;
    set_director_mode_state(executor, &next).await?;
    Ok(next)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeAuthorityPolicySource {
    GlobalDefault,
    EpicOverride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeAuthorityPolicyRecord {
    pub yolo_merge: bool,
    pub source: MergeAuthorityPolicySource,
}

fn decode_sqlite_bool(value: i64, field: &str) -> Result<bool, StorageError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(StorageError::InvalidData {
            message: format!("invalid sqlite bool for {field}: {value}"),
        }),
    }
}

pub async fn resolve_merge_authority_policy<'e, E>(
    executor: E,
    epic_id: EpicId,
) -> Result<MergeAuthorityPolicyRecord, StorageError>
where
    E: Executor<'e, Database = Sqlite> + Copy,
{
    let override_row: Option<(i64,)> = sqlx::query_as(
        r#"
        SELECT yolo_merge
        FROM epic_merge_authority_policy_override
        WHERE epic_id = ?1
        "#,
    )
    .bind(epic_id)
    .fetch_optional(executor)
    .await?;

    if let Some((yolo_merge_raw,)) = override_row {
        return Ok(MergeAuthorityPolicyRecord {
            yolo_merge: decode_sqlite_bool(yolo_merge_raw, "epic override yolo_merge")?,
            source: MergeAuthorityPolicySource::EpicOverride,
        });
    }

    let default_row: Option<(i64,)> = sqlx::query_as(
        r#"
        SELECT yolo_merge
        FROM director_merge_authority_policy_default
        WHERE singleton = 1
        "#,
    )
    .fetch_optional(executor)
    .await?;

    Ok(MergeAuthorityPolicyRecord {
        yolo_merge: decode_sqlite_bool(
            default_row.map_or(0, |(value,)| value),
            "global yolo_merge",
        )?,
        source: MergeAuthorityPolicySource::GlobalDefault,
    })
}

pub async fn set_global_merge_authority_default<'e, E>(
    executor: E,
    yolo_merge: bool,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO director_merge_authority_policy_default (
            singleton,
            updated_at_ms,
            yolo_merge
        )
        VALUES (1, ?1, ?2)
        ON CONFLICT(singleton) DO UPDATE SET
            updated_at_ms = excluded.updated_at_ms,
            yolo_merge = excluded.yolo_merge
        "#,
    )
    .bind(now_ms())
    .bind(if yolo_merge { 1_i64 } else { 0_i64 })
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn set_epic_merge_authority_override<'e, E>(
    executor: E,
    epic_id: EpicId,
    yolo_merge: bool,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query(
        r#"
        INSERT INTO epic_merge_authority_policy_override (
            epic_id,
            updated_at_ms,
            yolo_merge
        )
        VALUES (?1, ?2, ?3)
        ON CONFLICT(epic_id) DO UPDATE SET
            updated_at_ms = excluded.updated_at_ms,
            yolo_merge = excluded.yolo_merge
        "#,
    )
    .bind(epic_id)
    .bind(now_ms())
    .bind(if yolo_merge { 1_i64 } else { 0_i64 })
    .execute(executor)
    .await?;

    Ok(())
}

pub async fn clear_epic_merge_authority_override<'e, E>(
    executor: E,
    epic_id: EpicId,
) -> Result<(), StorageError>
where
    E: Executor<'e, Database = Sqlite>,
{
    sqlx::query("DELETE FROM epic_merge_authority_policy_override WHERE epic_id = ?1")
        .bind(epic_id)
        .execute(executor)
        .await?;
    Ok(())
}
