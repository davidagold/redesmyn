use redesmyn_ids::SessionId;
use redesmyn_protocol::agent_commands::SessionPolicySnapshot;
use redesmyn_protocol::client::SessionEventKindFilter;
use redesmyn_protocol::session::{PermissionsMode, SessionEventKind};
use redesmyn_storage::StorageError;

use crate::session_events::SessionEvents;

pub(crate) async fn load_session_policy_snapshot(
    session_events: &SessionEvents,
    session_id: SessionId,
) -> Result<SessionPolicySnapshot, StorageError> {
    let permissions_mode = {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::PermissionsModeChanged],
            )
            .await?;
        events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::PermissionsModeChanged(changed) => Some(changed.mode),
                _ => None,
            })
            .unwrap_or(PermissionsMode::Ask)
    };

    let codex_approval_policy = {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::CodexApprovalPolicyChanged],
            )
            .await?;
        events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::CodexApprovalPolicyChanged(changed) => changed.approval_policy,
                _ => None,
            })
    };

    let codex_sandbox_policy = {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::CodexSandboxPolicyChanged],
            )
            .await?;
        events
            .into_iter()
            .next()
            .and_then(|event| match event.kind {
                SessionEventKind::CodexSandboxPolicyChanged(changed) => changed.sandbox_policy,
                _ => None,
            })
    };

    let (model_id, model_reasoning_effort) = {
        let (events, _) = session_events
            .get_session_events(
                session_id,
                None,
                1,
                &[SessionEventKindFilter::SessionModelChanged],
            )
            .await?;

        events
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
            .unwrap_or((None, None))
    };

    Ok(SessionPolicySnapshot {
        permissions_mode,
        codex_approval_policy,
        codex_sandbox_policy,
        model_id,
        model_reasoning_effort,
    })
}
