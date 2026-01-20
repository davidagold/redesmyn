use tokio::sync::broadcast;

use redesmyn_protocol::{Timestamp, client::EventLogEvent};

#[derive(Debug, Clone)]
pub struct EventLog {
    tx: broadcast::Sender<EventLogEvent>,
}

impl EventLog {
    #[must_use]
    pub fn new(buffer: usize) -> Self {
        let (tx, _rx) = broadcast::channel(buffer);
        Self { tx }
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<EventLogEvent> {
        self.tx.subscribe()
    }

    pub fn publish(&self, event_type: impl Into<String>, json_payload: Vec<u8>) {
        let event = EventLogEvent {
            event_id: redesmyn_ids::EventId::new(),
            occurred_at: Timestamp::now_utc(),
            event_type: event_type.into(),
            json_payload,
        };

        // It's expected that the event log can be empty early in startup and
        // have no active subscribers. Dropping events in that case is fine.
        let _ = self.tx.send(event);
    }
}

