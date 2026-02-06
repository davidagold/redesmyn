use serde::Serialize;

use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::agent_commands::{
    AttachTaskAgentSessionCommand, ResumeByIdTaskAgentTurnCommand, SessionPolicySnapshot,
    StartTaskAgentSessionCommand, StopTaskAgentSessionCommand,
};
use redesmyn_protocol::client::AgentKind;
use redesmyn_protocol::{ErrorCategory, ErrorDetail, ErrorEnvelope, ExternalSessionRef};

fn encode_payload<T: Serialize>(payload: &T) -> Result<Vec<u8>, ErrorEnvelope> {
    serde_json::to_vec(payload).map_err(|err| {
        ErrorEnvelope::new(
            ErrorCategory::Internal,
            "Failed to encode agent command payload.",
        )
        .with_detail(ErrorDetail::from([("error".to_string(), err.to_string())]))
    })
}

pub(super) fn attach_session(session_id: SessionId) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&AttachTaskAgentSessionCommand { session_id })
}

pub(super) fn stop_task_sessions(
    task_id: TaskId,
    session_ids: Vec<SessionId>,
) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&StopTaskAgentSessionCommand {
        task_id,
        session_ids,
        reason: None,
    })
}

pub(super) fn start_task_session(
    session_id: SessionId,
    task_id: TaskId,
    agent_kind: AgentKind,
    initial_prompt: Option<String>,
    policy_snapshot: Option<SessionPolicySnapshot>,
    stop_session_ids: Vec<SessionId>,
) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&StartTaskAgentSessionCommand {
        session_id,
        task_id,
        agent_kind,
        initial_prompt,
        policy_snapshot,
        stop_session_ids,
    })
}

pub(super) fn resume_by_id_turn(
    session_id: SessionId,
    task_id: Option<TaskId>,
    prompt: String,
    external_session_ref: ExternalSessionRef,
    policy_snapshot: Option<SessionPolicySnapshot>,
    interrupt_turn: bool,
) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&ResumeByIdTaskAgentTurnCommand {
        session_id,
        task_id,
        prompt,
        image_attachments: Vec::new(),
        external_session_ref,
        policy_snapshot,
        interrupt_turn,
    })
}
