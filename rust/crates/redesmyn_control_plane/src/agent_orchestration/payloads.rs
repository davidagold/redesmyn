use serde::Serialize;

use redesmyn_ids::{SessionId, TaskId};
use redesmyn_protocol::agent_commands::{
    AttachTaskAgentSessionCommand, ResumeByIdTaskAgentTurnCommand, SendTaskAgentMessageCommand,
    StartTaskAgentSessionCommand, StopTaskAgentSessionCommand,
};
use redesmyn_protocol::client::{AgentInterfaceMode, AgentKind};
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
    interface_mode: AgentInterfaceMode,
    initial_prompt: Option<String>,
    stop_session_ids: Vec<SessionId>,
) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&StartTaskAgentSessionCommand {
        session_id,
        task_id,
        agent_kind,
        interface_mode,
        initial_prompt,
        stop_session_ids,
    })
}

pub(super) fn resume_by_id_turn(
    session_id: SessionId,
    prompt: String,
    external_session_ref: ExternalSessionRef,
    interrupt_turn: bool,
) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&ResumeByIdTaskAgentTurnCommand {
        session_id,
        prompt,
        external_session_ref,
        interrupt_turn,
    })
}

pub(super) fn send_message(
    session_id: SessionId,
    text: String,
    interrupt_turn: bool,
    submit: bool,
) -> Result<Vec<u8>, ErrorEnvelope> {
    encode_payload(&SendTaskAgentMessageCommand {
        session_id,
        text,
        interrupt_turn,
        submit,
    })
}
