use redesmyn_ids::SessionId;
use redesmyn_protocol::agent_commands::SessionPolicySnapshot;
use redesmyn_protocol::client::SessionEventKindFilter;
use redesmyn_protocol::session::{PermissionsMode, SessionEventKind};
use redesmyn_storage::StorageError;

use crate::session_events::SessionEvents;
use crate::session_events_projection::load_session_policy_projection;

pub(crate) async fn load_session_policy_snapshot(
    session_events: &SessionEvents,
    session_id: SessionId,
) -> Result<SessionPolicySnapshot, StorageError> {
    let projected = load_session_policy_projection(session_events.pool(), session_id).await?;

    let mut permissions_mode = projected
        .as_ref()
        .and_then(|value| value.permissions_mode)
        .filter(|value| !matches!(value, PermissionsMode::Unknown));
    let mut codex_approval_policy = projected
        .as_ref()
        .and_then(|value| value.codex_approval_policy)
        .filter(|value| {
            !matches!(
                value,
                redesmyn_protocol::session::CodexApprovalPolicy::Unknown
            )
        });
    let mut codex_sandbox_policy = projected
        .as_ref()
        .and_then(|value| value.codex_sandbox_policy.clone())
        .filter(|value| {
            !matches!(
                value,
                redesmyn_protocol::session::CodexSandboxPolicy::Unknown
            )
        });
    let mut model_id = projected.as_ref().and_then(|value| value.model_id.clone());
    let mut model_reasoning_effort = projected
        .as_ref()
        .and_then(|value| value.model_reasoning_effort)
        .filter(|value| {
            !matches!(
                value,
                redesmyn_protocol::client::ModelReasoningEffort::Unknown
            )
        });

    if permissions_mode.is_none() {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::PermissionsModeChanged],
            )
            .await?;
        permissions_mode = events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::PermissionsModeChanged(changed) => Some(changed.mode),
                _ => None,
            });
    }

    if codex_approval_policy.is_none() {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::CodexApprovalPolicyChanged],
            )
            .await?;
        codex_approval_policy = events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::CodexApprovalPolicyChanged(changed) => changed.approval_policy,
                _ => None,
            });
    }

    if codex_sandbox_policy.is_none() {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::CodexSandboxPolicyChanged],
            )
            .await?;
        codex_sandbox_policy = events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::CodexSandboxPolicyChanged(changed) => changed.sandbox_policy,
                _ => None,
            });
    }

    if model_id.is_none() && model_reasoning_effort.is_none() {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::SessionModelChanged],
            )
            .await?;

        let (resolved_model_id, resolved_model_reasoning_effort) = events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::SessionModelChanged(changed) => Some((
                    changed.model_id,
                    changed.reasoning_effort.and_then(|effort| match effort {
                        redesmyn_protocol::session::SessionModelReasoningEffort::Minimal => {
                            Some(redesmyn_protocol::client::ModelReasoningEffort::Minimal)
                        }
                        redesmyn_protocol::session::SessionModelReasoningEffort::Low => {
                            Some(redesmyn_protocol::client::ModelReasoningEffort::Low)
                        }
                        redesmyn_protocol::session::SessionModelReasoningEffort::Medium => {
                            Some(redesmyn_protocol::client::ModelReasoningEffort::Medium)
                        }
                        redesmyn_protocol::session::SessionModelReasoningEffort::High => {
                            Some(redesmyn_protocol::client::ModelReasoningEffort::High)
                        }
                        redesmyn_protocol::session::SessionModelReasoningEffort::Xhigh => {
                            Some(redesmyn_protocol::client::ModelReasoningEffort::Xhigh)
                        }
                        redesmyn_protocol::session::SessionModelReasoningEffort::Unknown => None,
                    }),
                )),
                _ => None,
            })
            .unwrap_or((None, None));
        model_id = resolved_model_id;
        model_reasoning_effort = resolved_model_reasoning_effort;
    }

    Ok(SessionPolicySnapshot {
        permissions_mode: permissions_mode.unwrap_or(PermissionsMode::Ask),
        codex_approval_policy,
        codex_sandbox_policy,
        model_id,
        model_reasoning_effort,
    })
}
