use std::collections::HashMap;

use tokio::sync::RwLock;

use redesmyn_ids::CommandId;
use redesmyn_logging::tracing;

use crate::event_log::EventLog;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandState {
    Accepted,
    Running,
    Succeeded,
    Failed,
    Canceled,
}

#[derive(Debug, Clone, serde::Serialize)]
struct CommandUpdatePayload {
    command_id: CommandId,
    state: CommandState,
    message: Option<String>,
}

#[derive(Debug)]
pub struct CommandRegistry {
    state: RwLock<HashMap<CommandId, CommandState>>,
    event_log: EventLog,
}

impl CommandRegistry {
    #[must_use]
    pub fn new(event_log: EventLog) -> Self {
        Self {
            state: RwLock::new(HashMap::new()),
            event_log,
        }
    }

    pub async fn create(&self, message: Option<String>) -> CommandId {
        let command_id = CommandId::new();
        self.set_state(command_id, CommandState::Accepted, message).await;
        command_id
    }

    pub async fn set_state(&self, command_id: CommandId, state: CommandState, message: Option<String>) {
        {
            let mut state_map = self.state.write().await;
            state_map.insert(command_id, state);
        }

        let payload = CommandUpdatePayload {
            command_id,
            state,
            message,
        };

        let json_payload = match serde_json::to_vec(&payload) {
            Ok(json) => json,
            Err(err) => {
                tracing::warn!(error = %err, "failed to encode command update payload");
                Vec::new()
            }
        };

        self.event_log.publish("command.update", json_payload);
    }

    pub async fn get_state(&self, command_id: CommandId) -> Option<CommandState> {
        let state = self.state.read().await;
        state.get(&command_id).copied()
    }
}

