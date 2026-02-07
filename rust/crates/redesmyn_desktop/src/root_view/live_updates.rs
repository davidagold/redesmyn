use redesmyn_ids::{CommandId, TaskId};
use redesmyn_protocol::Timestamp;
use redesmyn_protocol::client::{CommandState, EventLogEvent, SubscriptionEvent, TaskState};
use redesmyn_protocol::sync_commands::{LOCAL_SYNC_EVENT_APPLIED, LOCAL_SYNC_EVENT_FAILED};
use redesmyn_protocol::task_events::TASK_STATE_CHANGED_EVENT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphRefreshReason {
    SyncApplied,
    SyncFailed,
    TaskStatePayloadInvalid,
    CommandPayloadInvalid,
    SubscriptionError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskStateUpdate {
    pub task_id: TaskId,
    pub state: TaskState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandStateUpdate {
    pub command_id: CommandId,
    pub target_task_id: Option<TaskId>,
    pub kind: Option<String>,
    pub state: CommandState,
    pub message: Option<String>,
    pub updated_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveUpdateAction {
    RefreshGraph(GraphRefreshReason),
    TaskStateChanged(TaskStateUpdate),
    CommandStateChanged(CommandStateUpdate),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LiveUpdateRouter;

impl LiveUpdateRouter {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    #[must_use]
    pub fn route_subscription_event(&self, event: &SubscriptionEvent) -> Vec<LiveUpdateAction> {
        match event {
            SubscriptionEvent::EventLog(event) => self.route_event_log(event),
            SubscriptionEvent::Error(_) => {
                vec![LiveUpdateAction::RefreshGraph(
                    GraphRefreshReason::SubscriptionError,
                )]
            }
            _ => Vec::new(),
        }
    }

    fn route_event_log(&self, event: &EventLogEvent) -> Vec<LiveUpdateAction> {
        if event.event_type == LOCAL_SYNC_EVENT_APPLIED {
            return vec![LiveUpdateAction::RefreshGraph(
                GraphRefreshReason::SyncApplied,
            )];
        }
        if event.event_type == LOCAL_SYNC_EVENT_FAILED {
            return vec![LiveUpdateAction::RefreshGraph(
                GraphRefreshReason::SyncFailed,
            )];
        }
        if event.event_type == TASK_STATE_CHANGED_EVENT {
            return self.route_task_state_event(event);
        }
        if event.event_type.starts_with("command.") {
            return self.route_command_state_event(event);
        }
        Vec::new()
    }

    fn route_task_state_event(&self, event: &EventLogEvent) -> Vec<LiveUpdateAction> {
        #[derive(serde::Deserialize)]
        struct TaskStateChangedPayload {
            task_id: TaskId,
            state: TaskState,
        }

        match serde_json::from_slice::<TaskStateChangedPayload>(&event.json_payload) {
            Ok(payload) => vec![LiveUpdateAction::TaskStateChanged(TaskStateUpdate {
                task_id: payload.task_id,
                state: payload.state,
            })],
            Err(_) => vec![LiveUpdateAction::RefreshGraph(
                GraphRefreshReason::TaskStatePayloadInvalid,
            )],
        }
    }

    fn route_command_state_event(&self, event: &EventLogEvent) -> Vec<LiveUpdateAction> {
        #[derive(serde::Deserialize)]
        struct CommandUpdatePayload {
            command_id: CommandId,
            #[serde(default)]
            state: Option<String>,
            #[serde(default)]
            kind: Option<String>,
            #[serde(default)]
            target_task_id: Option<TaskId>,
            #[serde(default)]
            message: Option<String>,
        }

        let Ok(payload) = serde_json::from_slice::<CommandUpdatePayload>(&event.json_payload)
        else {
            return vec![LiveUpdateAction::RefreshGraph(
                GraphRefreshReason::CommandPayloadInvalid,
            )];
        };

        let state = payload
            .state
            .as_deref()
            .and_then(parse_command_state)
            .or_else(|| {
                event
                    .event_type
                    .strip_prefix("command.")
                    .and_then(parse_command_state)
            });
        let Some(state) = state else {
            return vec![LiveUpdateAction::RefreshGraph(
                GraphRefreshReason::CommandPayloadInvalid,
            )];
        };

        vec![LiveUpdateAction::CommandStateChanged(CommandStateUpdate {
            command_id: payload.command_id,
            target_task_id: payload.target_task_id,
            kind: payload.kind,
            state,
            message: payload.message,
            updated_at: event.occurred_at,
        })]
    }
}

fn parse_command_state(value: &str) -> Option<CommandState> {
    match value {
        "unknown" => Some(CommandState::Unknown),
        "queued" => Some(CommandState::Queued),
        "accepted" => Some(CommandState::Accepted),
        "running" => Some(CommandState::Running),
        "blocked" => Some(CommandState::Blocked),
        "resumable" => Some(CommandState::Resumable),
        "succeeded" => Some(CommandState::Succeeded),
        "failed" => Some(CommandState::Failed),
        "canceled" => Some(CommandState::Canceled),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphRefreshReason, LiveUpdateAction, LiveUpdateRouter, parse_command_state};
    use redesmyn_ids::{CommandId, EventId, TaskId};
    use redesmyn_protocol::Timestamp;
    use redesmyn_protocol::client::{CommandState, EventLogEvent, SubscriptionEvent, TaskState};
    use redesmyn_protocol::task_events::TASK_STATE_CHANGED_EVENT;

    #[test]
    fn parses_command_states() {
        assert_eq!(parse_command_state("running"), Some(CommandState::Running));
        assert_eq!(parse_command_state("bad-state"), None);
    }

    #[test]
    fn routes_task_state_changed_event() {
        let router = LiveUpdateRouter::new();
        let event = EventLogEvent {
            event_id: EventId::from_bytes([1; 16]),
            occurred_at: Timestamp::from_unix_millis(0).expect("timestamp"),
            event_type: TASK_STATE_CHANGED_EVENT.to_string(),
            json_payload: serde_json::to_vec(&serde_json::json!({
                "task_id": TaskId::from_bytes([2; 16]),
                "state": "in_progress"
            }))
            .expect("json"),
        };
        let updates = router.route_subscription_event(&SubscriptionEvent::EventLog(event));
        assert!(matches!(
            updates.as_slice(),
            [LiveUpdateAction::TaskStateChanged(update)]
            if update.state == TaskState::InProgress
        ));
    }

    #[test]
    fn routes_command_state_event() {
        let router = LiveUpdateRouter::new();
        let command_id = CommandId::from_bytes([3; 16]);
        let task_id = TaskId::from_bytes([4; 16]);
        let event = EventLogEvent {
            event_id: EventId::from_bytes([5; 16]),
            occurred_at: Timestamp::from_unix_millis(1).expect("timestamp"),
            event_type: "command.running".to_string(),
            json_payload: serde_json::to_vec(&serde_json::json!({
                "command_id": command_id,
                "state": "running",
                "kind": "task.agent.start",
                "target_task_id": task_id,
                "message": "starting"
            }))
            .expect("json"),
        };

        let updates = router.route_subscription_event(&SubscriptionEvent::EventLog(event));
        assert!(matches!(
            updates.as_slice(),
            [LiveUpdateAction::CommandStateChanged(update)]
            if update.command_id == command_id
                && update.target_task_id == Some(task_id)
                && update.state == CommandState::Running
        ));
    }

    #[test]
    fn falls_back_to_refresh_on_invalid_command_payload() {
        let router = LiveUpdateRouter::new();
        let event = EventLogEvent {
            event_id: EventId::from_bytes([6; 16]),
            occurred_at: Timestamp::from_unix_millis(2).expect("timestamp"),
            event_type: "command.failed".to_string(),
            json_payload: br#"{"command_id":"not-a-command-id"}"#.to_vec(),
        };
        let updates = router.route_subscription_event(&SubscriptionEvent::EventLog(event));
        assert_eq!(
            updates,
            vec![LiveUpdateAction::RefreshGraph(
                GraphRefreshReason::CommandPayloadInvalid
            )]
        );
    }
}
