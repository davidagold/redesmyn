use redesmyn_ids::SessionId;
use redesmyn_protocol::client::{
    AgentInterfaceMode, AgentKind, AgentMessageConflictAction,
    TaskAgentMessageConversationContinuity, TaskAgentMessageDelivery,
};
use redesmyn_protocol::{ErrorEnvelope, ExternalSessionRef};

use super::conflicts::{
    CONFLICT_CODE_SESSION_CONFLICT, CONFLICT_CODE_TURN_IN_PROGRESS, conflict_envelope,
    invalid_request,
};
use super::state::{
    ResumableStructuredSession, StorageAgentSessionRecord, is_active_task_session,
    protocol_agent_kind_from_storage, protocol_interface_mode_from_storage,
};

pub(super) fn desired_interface_mode(
    agent_kind: AgentKind,
    preferred: Option<AgentInterfaceMode>,
) -> AgentInterfaceMode {
    preferred.unwrap_or(match agent_kind {
        AgentKind::Codex | AgentKind::ClaudeCode => AgentInterfaceMode::StructuredExec,
        AgentKind::Shell => AgentInterfaceMode::ShellTmux,
    })
}

pub(super) fn effective_on_conflict(
    on_conflict: AgentMessageConflictAction,
    interrupt: Option<bool>,
) -> AgentMessageConflictAction {
    if on_conflict != AgentMessageConflictAction::Fail {
        return on_conflict;
    }
    if interrupt == Some(true) {
        return AgentMessageConflictAction::InterruptTurn;
    }
    AgentMessageConflictAction::Fail
}

pub(super) struct StartAgentPlan {
    pub stop_session_ids: Vec<SessionId>,
}

pub(super) fn plan_start_agent(
    active_sessions: Vec<SessionId>,
    on_conflict: AgentMessageConflictAction,
) -> Result<StartAgentPlan, ErrorEnvelope> {
    if active_sessions.is_empty() {
        return Ok(StartAgentPlan {
            stop_session_ids: Vec::new(),
        });
    }

    match on_conflict {
        AgentMessageConflictAction::StopSessionAndStartNew => Ok(StartAgentPlan {
            stop_session_ids: active_sessions,
        }),
        AgentMessageConflictAction::Fail | AgentMessageConflictAction::InterruptTurn => {
            Err(conflict_envelope(
                CONFLICT_CODE_SESSION_CONFLICT,
                "An agent session is currently running for this task.",
            ))
        }
    }
}

pub(super) struct StopAgentPlan {
    pub session_ids: Vec<SessionId>,
}

pub(super) fn plan_stop_agent(session_ids: Vec<SessionId>) -> StopAgentPlan {
    StopAgentPlan { session_ids }
}

pub(super) enum SendTaskAgentMessagePlan {
    StructuredResume(StructuredResumePlan),
    StructuredStart(NewSessionPlan),
    InteractiveStart(NewSessionPlan),
    InteractiveSendExisting(InteractiveSendExistingPlan),
}

pub(super) struct StructuredResumePlan {
    pub session_id: SessionId,
    pub external_session_ref: ExternalSessionRef,
    pub interrupt_turn: bool,
}

pub(super) struct NewSessionPlan {
    pub stop_session_ids: Vec<SessionId>,
    pub delivery: TaskAgentMessageDelivery,
    pub conversation_continuity: TaskAgentMessageConversationContinuity,
}

pub(super) struct InteractiveSendExistingPlan {
    pub session_id: SessionId,
    pub interrupt_turn: bool,
    pub warnings: Vec<String>,
    pub delivery: TaskAgentMessageDelivery,
    pub conversation_continuity: TaskAgentMessageConversationContinuity,
}

fn sessions_without_end(recent_sessions: &[StorageAgentSessionRecord]) -> Vec<SessionId> {
    recent_sessions
        .iter()
        .filter(|row| row.ended_at_ms.is_none())
        .map(|row| row.session_id)
        .collect()
}

pub(super) fn plan_send_task_agent_message(
    recent_sessions: &[StorageAgentSessionRecord],
    agent_kind: AgentKind,
    desired_interface_mode: AgentInterfaceMode,
    effective_on_conflict: AgentMessageConflictAction,
    resumable_structured: Option<&ResumableStructuredSession>,
) -> Result<SendTaskAgentMessagePlan, ErrorEnvelope> {
    let compatible_sessions: Vec<_> = recent_sessions
        .iter()
        .filter(|row| {
            is_active_task_session(row)
                && protocol_agent_kind_from_storage(row.agent_kind) == agent_kind
                && protocol_interface_mode_from_storage(row.interface_mode)
                    == desired_interface_mode
        })
        .collect();

    let active_any = recent_sessions
        .iter()
        .find(|row| is_active_task_session(row));

    if desired_interface_mode == AgentInterfaceMode::StructuredExec {
        if let Some(resumable) = resumable_structured {
            if effective_on_conflict == AgentMessageConflictAction::StopSessionAndStartNew {
                if resumable.turn_in_progress {
                    return Err(conflict_envelope(
                        CONFLICT_CODE_TURN_IN_PROGRESS,
                        "Agent turn in progress. Interrupt the current turn before starting a new session.",
                    ));
                }

                return Ok(SendTaskAgentMessagePlan::StructuredStart(NewSessionPlan {
                    stop_session_ids: sessions_without_end(recent_sessions),
                    delivery: TaskAgentMessageDelivery::StructuredStarted,
                    conversation_continuity: TaskAgentMessageConversationContinuity::Broken,
                }));
            }

            if resumable.turn_in_progress
                && effective_on_conflict == AgentMessageConflictAction::Fail
            {
                return Err(conflict_envelope(
                    CONFLICT_CODE_TURN_IN_PROGRESS,
                    "Agent turn in progress. Interrupt the current turn before sending a new structured message.",
                ));
            }

            return Ok(SendTaskAgentMessagePlan::StructuredResume(
                StructuredResumePlan {
                    session_id: resumable.session_id,
                    external_session_ref: resumable.external_session_ref.clone(),
                    interrupt_turn: resumable.turn_in_progress
                        && effective_on_conflict == AgentMessageConflictAction::InterruptTurn,
                },
            ));
        }

        if active_any.is_some()
            && effective_on_conflict == AgentMessageConflictAction::InterruptTurn
        {
            return Err(invalid_request(
                "Cannot interrupt a structured turn when no resumable structured session exists. Use on_conflict=stop_session_and_start_new instead.",
            ));
        }
        if active_any.is_some() && effective_on_conflict == AgentMessageConflictAction::Fail {
            return Err(conflict_envelope(
                CONFLICT_CODE_SESSION_CONFLICT,
                "A (non-resumable) agent session is running for this task. Stop it and start a new structured session to send this message?",
            ));
        }

        let stop_session_ids = if active_any.is_some()
            && effective_on_conflict == AgentMessageConflictAction::StopSessionAndStartNew
        {
            sessions_without_end(recent_sessions)
        } else {
            Vec::new()
        };

        return Ok(SendTaskAgentMessagePlan::StructuredStart(NewSessionPlan {
            stop_session_ids,
            delivery: TaskAgentMessageDelivery::StructuredStarted,
            conversation_continuity: TaskAgentMessageConversationContinuity::Broken,
        }));
    }

    // Interactive mode.
    let active_interactive = compatible_sessions.first().map(|row| row.session_id);
    let active_any_session_id = active_any.map(|row| row.session_id);

    if active_interactive.is_none() && active_any_session_id.is_some() {
        let active_any_session_id = active_any_session_id.expect("checked is_some above");
        let active_row = recent_sessions
            .iter()
            .find(|row| row.session_id == active_any_session_id)
            .expect("active_any points at a row in recent_sessions");

        let active_row_mode = protocol_interface_mode_from_storage(active_row.interface_mode);

        if active_row_mode == AgentInterfaceMode::StructuredExec
            && effective_on_conflict != AgentMessageConflictAction::StopSessionAndStartNew
        {
            return Err(conflict_envelope(
                CONFLICT_CODE_SESSION_CONFLICT,
                "An incompatible structured agent session is currently running for this task. Stop it and start a new interactive session to send this message?",
            ));
        }

        if effective_on_conflict == AgentMessageConflictAction::StopSessionAndStartNew {
            return Ok(SendTaskAgentMessagePlan::InteractiveStart(NewSessionPlan {
                stop_session_ids: sessions_without_end(recent_sessions),
                delivery: TaskAgentMessageDelivery::InteractiveStarted,
                conversation_continuity: TaskAgentMessageConversationContinuity::Broken,
            }));
        }

        return Ok(SendTaskAgentMessagePlan::InteractiveSendExisting(
            InteractiveSendExistingPlan {
                session_id: active_row.session_id,
                interrupt_turn: effective_on_conflict == AgentMessageConflictAction::InterruptTurn,
                delivery: TaskAgentMessageDelivery::InteractiveSent,
                conversation_continuity: TaskAgentMessageConversationContinuity::Kept,
                warnings: vec![
                    "Sent message to the currently running agent session (it does not match the configured harness command).".to_string(),
                ],
            },
        ));
    }

    if let Some(session_id) = active_interactive {
        return Ok(SendTaskAgentMessagePlan::InteractiveSendExisting(
            InteractiveSendExistingPlan {
                session_id,
                interrupt_turn: effective_on_conflict == AgentMessageConflictAction::InterruptTurn,
                delivery: TaskAgentMessageDelivery::InteractiveSent,
                conversation_continuity: TaskAgentMessageConversationContinuity::Kept,
                warnings: Vec::new(),
            },
        ));
    }

    Ok(SendTaskAgentMessagePlan::InteractiveStart(NewSessionPlan {
        stop_session_ids: Vec::new(),
        delivery: TaskAgentMessageDelivery::InteractiveStarted,
        conversation_continuity: TaskAgentMessageConversationContinuity::Broken,
    }))
}
