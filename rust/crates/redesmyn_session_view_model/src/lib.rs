//! Session viewer presentation model (pure logic; no GPUI).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use redesmyn_ids::{SessionEventId, SessionId};
use redesmyn_protocol::client::SessionEventCursor;
use redesmyn_protocol::session::{
    ArtifactEmitted, AssistantMessage, SessionEventKind, StatusUpdate, ToolInvocation, ToolResult,
    TurnCompleted, TurnStarted, UserMessage,
};
use redesmyn_protocol::session_live::{SessionLiveEvent, SessionLiveEventKind};
use redesmyn_protocol::{ArtifactRef, SessionEvent, Timestamp};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SessionMessageRole {
    User,
    Assistant,
    Tool,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SessionEventKindTag {
    SessionStarted,
    SessionEnded,
    TurnStarted,
    TurnCompleted,
    UserMessage,
    AssistantMessage,
    ToolInvocation,
    ToolResult,
    StatusUpdate,
    ArtifactEmitted,
    Unknown,
}

impl SessionEventKindTag {
    #[must_use]
    pub fn from_kind(kind: &SessionEventKind) -> Self {
        match kind {
            SessionEventKind::SessionStarted(_) => Self::SessionStarted,
            SessionEventKind::SessionEnded(_) => Self::SessionEnded,
            SessionEventKind::TurnStarted(_) => Self::TurnStarted,
            SessionEventKind::TurnCompleted(_) => Self::TurnCompleted,
            SessionEventKind::UserMessage(_) => Self::UserMessage,
            SessionEventKind::AssistantMessage(_) => Self::AssistantMessage,
            SessionEventKind::ToolInvocation(_) => Self::ToolInvocation,
            SessionEventKind::ToolResult(_) => Self::ToolResult,
            SessionEventKind::StatusUpdate(_) => Self::StatusUpdate,
            SessionEventKind::ArtifactEmitted(_) => Self::ArtifactEmitted,
            SessionEventKind::Unknown(_) => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEventRow {
    pub cursor: SessionEventCursor,
    pub event: SessionEvent,
    pub kind: SessionEventKindTag,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
}

impl SessionEventRow {
    #[must_use]
    pub fn from_event(event: SessionEvent) -> Self {
        let kind = SessionEventKindTag::from_kind(&event.kind);
        let preview = preview_from_event_kind(&event.kind);
        Self {
            cursor: SessionEventCursor {
                created_at: event.created_at,
                session_event_id: event.session_event_id,
            },
            event,
            kind,
            preview,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionTimelineItem {
    LoadOlder(LoadOlderRow),
    NewMessages(NewMessagesRow),
    Event(SessionEventItem),
    EphemeralText(EphemeralTextItem),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LoadOlderRow {
    pub enabled: bool,
    pub in_flight: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NewMessagesRow {
    pub count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEventItem {
    pub session_event_id: SessionEventId,
    pub created_at: Timestamp,
    pub turn_id: Option<String>,
    pub kind: SessionEventKindTag,
    pub content: SessionEventItemContent,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SessionEventItemContent {
    SessionStarted,
    SessionEnded,
    TurnStarted(TurnStarted),
    TurnCompleted(TurnCompleted),
    UserMessage(MessageItem),
    AssistantMessage(MessageItem),
    ToolInvocation(ToolInvocation),
    ToolResult(ToolResult),
    StatusUpdate(StatusUpdate),
    ArtifactEmitted(ArtifactEmittedItem),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MessageItem {
    pub role: SessionMessageRole,
    pub text: String,
    pub preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_artifact: Option<ArtifactRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactEmittedItem {
    pub artifact: ArtifactRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EphemeralTextItem {
    pub key: String,
    pub role: SessionMessageRole,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionHistoryState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<SessionEventCursor>,
    pub loading_older: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionLiveState {
    pub syncing: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionScrollState {
    pub at_bottom: bool,
    pub unseen_count: u32,
    pub pending_scroll_to_bottom: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_prepend_anchor: Option<SessionEventId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionFeedSummaryState {
    pub message_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_message_preview: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionComposerConflictPrompt {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionComposerState {
    pub draft: String,
    pub sending: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conflict_prompt: Option<SessionComposerConflictPrompt>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionFeedState {
    pub session_id: SessionId,
    events: Vec<SessionEventRow>,
    event_ids: BTreeSet<SessionEventId>,
    ephemeral: BTreeMap<String, EphemeralTextItem>,
    pub summary: SessionFeedSummaryState,
    pub composer: SessionComposerState,
    pub history: SessionHistoryState,
    pub live: SessionLiveState,
    pub scroll: SessionScrollState,
}

impl SessionFeedState {
    #[must_use]
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            events: Vec::new(),
            event_ids: BTreeSet::new(),
            ephemeral: BTreeMap::new(),
            summary: SessionFeedSummaryState::default(),
            composer: SessionComposerState::default(),
            history: SessionHistoryState::default(),
            live: SessionLiveState::default(),
            scroll: SessionScrollState::default(),
        }
    }

    #[must_use]
    pub fn timeline_items(&self) -> Vec<SessionTimelineItem> {
        let mut items = Vec::new();

        if self.history.next_cursor.is_some() || self.history.loading_older {
            items.push(SessionTimelineItem::LoadOlder(LoadOlderRow {
                enabled: self.history.next_cursor.is_some() && !self.history.loading_older,
                in_flight: self.history.loading_older,
            }));
        }

        items.extend(
            self.events
                .iter()
                .map(|row| SessionTimelineItem::Event(SessionEventItem::from_row(row))),
        );

        items.extend(
            self.ephemeral
                .values()
                .cloned()
                .map(SessionTimelineItem::EphemeralText),
        );

        if !self.scroll.at_bottom && self.scroll.unseen_count > 0 {
            items.push(SessionTimelineItem::NewMessages(NewMessagesRow {
                count: self.scroll.unseen_count,
            }));
        }

        items
    }

    pub fn set_at_bottom(&mut self, at_bottom: bool) {
        if self.scroll.at_bottom == at_bottom {
            return;
        }

        self.scroll.at_bottom = at_bottom;
        if at_bottom {
            self.scroll.unseen_count = 0;
        }
    }

    pub fn clear_scroll_intents(&mut self) {
        self.scroll.pending_scroll_to_bottom = false;
        self.scroll.pending_prepend_anchor = None;
    }

    pub fn start_loading_older(&mut self) {
        if self.history.loading_older {
            return;
        }
        self.history.loading_older = true;
        self.history.error = None;
    }

    pub fn apply_history_page(
        &mut self,
        events: Vec<SessionEvent>,
        next_cursor: Option<SessionEventCursor>,
    ) {
        let anchor = self.events.first().map(|row| row.event.session_event_id);

        for event in events {
            self.insert_event(SessionEventRow::from_event(event));
        }

        self.recompute_summary();
        self.history.loading_older = false;
        self.history.next_cursor = next_cursor;
        self.history.error = None;
        self.scroll.pending_prepend_anchor = anchor;
    }

    pub fn apply_history_error(&mut self, message: impl Into<String>) {
        self.history.loading_older = false;
        self.history.error = Some(message.into());
    }

    pub fn set_live_syncing(&mut self, syncing: bool) {
        self.live.syncing = syncing;
    }

    pub fn apply_live_error(&mut self, message: impl Into<String>) {
        self.live.error = Some(message.into());
        self.live.syncing = false;
    }

    pub fn apply_live_event(&mut self, event: SessionEvent) {
        let clear_assistant_ephemeral =
            matches!(&event.kind, SessionEventKind::AssistantMessage(_));
        let clear_tool_ephemeral = match &event.kind {
            SessionEventKind::ToolResult(ev) => Some(ev.tool_call_id.clone()),
            _ => None,
        };
        let inserted = self.insert_event(SessionEventRow::from_event(event));

        let Some(inserted_at_end) = inserted else {
            return;
        };

        if inserted_at_end {
            if clear_assistant_ephemeral {
                self.ephemeral
                    .retain(|_, item| item.role != SessionMessageRole::Assistant);
            }
            if let Some(tool_call_id) = clear_tool_ephemeral {
                match tool_call_id {
                    Some(tool_call_id) => {
                        self.ephemeral.remove(&tool_call_id);
                    }
                    None => {
                        self.ephemeral
                            .retain(|_, item| item.role != SessionMessageRole::Tool);
                    }
                }
            }
        }

        if inserted_at_end {
            if self.scroll.at_bottom {
                self.scroll.pending_scroll_to_bottom = true;
            } else {
                self.scroll.unseen_count = self.scroll.unseen_count.saturating_add(1);
            }
        }

        self.recompute_summary();
    }

    pub fn apply_live_session_event(&mut self, event: SessionLiveEvent) {
        if event.session_id != self.session_id {
            return;
        }

        match event.kind {
            SessionLiveEventKind::AssistantMessageDelta(delta) => {
                if delta.delta.is_empty() {
                    return;
                }

                let key = event
                    .item_id
                    .unwrap_or_else(|| "assistant_delta".to_owned());

                let entry = self
                    .ephemeral
                    .entry(key.clone())
                    .or_insert(EphemeralTextItem {
                        key,
                        role: SessionMessageRole::Assistant,
                        text: String::new(),
                    });

                entry.text.push_str(&delta.delta);
                if self.scroll.at_bottom {
                    self.scroll.pending_scroll_to_bottom = true;
                }
            }
            SessionLiveEventKind::ToolOutputDelta(delta) => {
                if delta.delta.is_empty() {
                    return;
                }

                let key = event
                    .item_id
                    .unwrap_or_else(|| "tool_output_delta".to_owned());

                let entry = self
                    .ephemeral
                    .entry(key.clone())
                    .or_insert(EphemeralTextItem {
                        key,
                        role: SessionMessageRole::Tool,
                        text: format!("{}\n", delta.tool_name),
                    });

                entry.text.push_str(&delta.delta);
                if self.scroll.at_bottom {
                    self.scroll.pending_scroll_to_bottom = true;
                }
            }
            SessionLiveEventKind::Unknown(_) => {}
        }
    }

    pub fn set_draft(&mut self, draft: impl Into<String>) {
        self.composer.draft = draft.into();
        self.composer.last_error = None;
        self.composer.conflict_prompt = None;
    }

    pub fn start_sending(&mut self) {
        self.composer.sending = true;
        self.composer.last_error = None;
        self.composer.conflict_prompt = None;
    }

    pub fn finish_sending_success(&mut self) {
        self.composer.sending = false;
        self.composer.draft.clear();
        self.composer.last_error = None;
        self.composer.conflict_prompt = None;
    }

    pub fn finish_sending_error(&mut self, message: impl Into<String>) {
        self.composer.sending = false;
        self.composer.last_error = Some(message.into());
        self.composer.conflict_prompt = None;
    }

    pub fn finish_sending_conflict(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.composer.sending = false;
        self.composer.last_error = None;
        self.composer.conflict_prompt = Some(SessionComposerConflictPrompt {
            code: code.into(),
            message: message.into(),
        });
    }

    pub fn upsert_ephemeral_text(
        &mut self,
        key: impl Into<String>,
        role: SessionMessageRole,
        text: impl Into<String>,
    ) {
        let key = key.into();
        self.ephemeral.insert(
            key.clone(),
            EphemeralTextItem {
                key,
                role,
                text: text.into(),
            },
        );
    }

    pub fn clear_ephemeral(&mut self) {
        self.ephemeral.clear();
    }

    fn insert_event(&mut self, row: SessionEventRow) -> Option<bool> {
        if row.event.session_id != self.session_id {
            return None;
        }

        if !self.event_ids.insert(row.event.session_event_id) {
            return None;
        }

        let pos = self
            .events
            .binary_search_by(|existing| existing.cursor.cmp(&row.cursor))
            .unwrap_or_else(|pos| pos);
        self.events.insert(pos, row);
        Some(pos + 1 == self.events.len())
    }

    fn recompute_summary(&mut self) {
        let mut message_count = 0_u32;
        let mut last_preview = None;

        for row in &self.events {
            match row.kind {
                SessionEventKindTag::UserMessage | SessionEventKindTag::AssistantMessage => {
                    message_count = message_count.saturating_add(1);
                    last_preview = row.preview.clone();
                }
                _ => {}
            }
        }

        self.summary.message_count = message_count;
        self.summary.last_message_preview = last_preview;
    }
}

impl SessionEventItem {
    #[must_use]
    pub fn from_row(row: &SessionEventRow) -> Self {
        let content = match &row.event.kind {
            SessionEventKind::SessionStarted(_) => SessionEventItemContent::SessionStarted,
            SessionEventKind::SessionEnded(_) => SessionEventItemContent::SessionEnded,
            SessionEventKind::TurnStarted(ev) => SessionEventItemContent::TurnStarted(ev.clone()),
            SessionEventKind::TurnCompleted(ev) => {
                SessionEventItemContent::TurnCompleted(ev.clone())
            }
            SessionEventKind::UserMessage(ev) => {
                SessionEventItemContent::UserMessage(MessageItem {
                    role: SessionMessageRole::User,
                    text: ev.text.clone(),
                    preview: ev.preview.clone(),
                    full_text_artifact: ev.full_text_artifact.clone(),
                })
            }
            SessionEventKind::AssistantMessage(ev) => {
                SessionEventItemContent::AssistantMessage(MessageItem {
                    role: SessionMessageRole::Assistant,
                    text: ev.text.clone(),
                    preview: ev.preview.clone(),
                    full_text_artifact: ev.full_text_artifact.clone(),
                })
            }
            SessionEventKind::ToolInvocation(ev) => {
                SessionEventItemContent::ToolInvocation(ev.clone())
            }
            SessionEventKind::ToolResult(ev) => SessionEventItemContent::ToolResult(ev.clone()),
            SessionEventKind::StatusUpdate(ev) => SessionEventItemContent::StatusUpdate(ev.clone()),
            SessionEventKind::ArtifactEmitted(ev) => {
                SessionEventItemContent::ArtifactEmitted(ArtifactEmittedItem {
                    artifact: ev.artifact.clone(),
                    label: ev.label.clone(),
                })
            }
            SessionEventKind::Unknown(_) => SessionEventItemContent::Unknown,
        };

        Self {
            session_event_id: row.event.session_event_id,
            created_at: row.event.created_at,
            turn_id: row.event.turn_id.clone(),
            kind: row.kind,
            content,
        }
    }
}

fn preview_from_event_kind(kind: &SessionEventKind) -> Option<String> {
    match kind {
        SessionEventKind::UserMessage(UserMessage { preview, .. }) => Some(preview.clone()),
        SessionEventKind::AssistantMessage(AssistantMessage { preview, .. }) => {
            Some(preview.clone())
        }
        SessionEventKind::ToolInvocation(ToolInvocation { input_preview, .. }) => {
            Some(input_preview.clone())
        }
        SessionEventKind::ToolResult(ToolResult { output_preview, .. }) => {
            Some(output_preview.clone())
        }
        SessionEventKind::StatusUpdate(StatusUpdate { message, .. }) => message.clone(),
        SessionEventKind::ArtifactEmitted(ArtifactEmitted { label, .. }) => label.clone(),
        SessionEventKind::TurnStarted(TurnStarted {
            idempotency_key, ..
        }) => idempotency_key.clone(),
        SessionEventKind::TurnCompleted(TurnCompleted { error, .. }) => {
            error.as_ref().map(|e| e.message.clone())
        }
        SessionEventKind::SessionStarted(_)
        | SessionEventKind::SessionEnded(_)
        | SessionEventKind::Unknown(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(seconds: i64) -> Timestamp {
        let dt = time::OffsetDateTime::from_unix_timestamp(seconds).unwrap();
        Timestamp::from_offset_date_time(dt)
    }

    fn user_event(
        session_id: SessionId,
        id: SessionEventId,
        created_at: Timestamp,
        preview: &str,
    ) -> SessionEvent {
        SessionEvent {
            session_event_id: id,
            created_at,
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: None,
            kind: SessionEventKind::UserMessage(UserMessage {
                text: preview.to_string(),
                preview: preview.to_string(),
                full_text_artifact: None,
            }),
        }
    }

    fn assistant_event(
        session_id: SessionId,
        id: SessionEventId,
        created_at: Timestamp,
        preview: &str,
    ) -> SessionEvent {
        SessionEvent {
            session_event_id: id,
            created_at,
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: None,
            kind: SessionEventKind::AssistantMessage(AssistantMessage {
                text: preview.to_string(),
                preview: preview.to_string(),
                full_text_artifact: None,
            }),
        }
    }

    fn tool_result_event(
        session_id: SessionId,
        id: SessionEventId,
        created_at: Timestamp,
        tool_call_id: &str,
        preview: &str,
    ) -> SessionEvent {
        SessionEvent {
            session_event_id: id,
            created_at,
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: None,
            kind: SessionEventKind::ToolResult(ToolResult {
                tool_name: "exec_command".to_owned(),
                tool_call_id: Some(tool_call_id.to_owned()),
                output_preview: preview.to_owned(),
                output_artifact: None,
                error: None,
            }),
        }
    }

    #[test]
    fn test_history_prepend_sets_anchor() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        state.history.next_cursor = Some(SessionEventCursor {
            created_at: ts(0),
            session_event_id: SessionEventId::new(),
        });

        let existing_id = SessionEventId::new();
        state.apply_live_event(user_event(session_id, existing_id, ts(10), "newer"));
        state.clear_scroll_intents();

        let older_id = SessionEventId::new();
        state.start_loading_older();
        state.apply_history_page(vec![user_event(session_id, older_id, ts(1), "older")], None);

        assert_eq!(state.scroll.pending_prepend_anchor, Some(existing_id));
        assert_eq!(state.events.len(), 2);
        assert_eq!(state.events[0].event.session_event_id, older_id);
        assert_eq!(state.events[1].event.session_event_id, existing_id);
    }

    #[test]
    fn test_summary_updates_message_count_and_last_preview() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        assert_eq!(state.summary.message_count, 0);
        assert_eq!(state.summary.last_message_preview, None);

        state.apply_live_event(user_event(
            session_id,
            SessionEventId::new(),
            ts(1),
            "hello",
        ));
        assert_eq!(state.summary.message_count, 1);
        assert_eq!(state.summary.last_message_preview.as_deref(), Some("hello"));

        state.apply_live_event(assistant_event(
            session_id,
            SessionEventId::new(),
            ts(2),
            "world",
        ));
        assert_eq!(state.summary.message_count, 2);
        assert_eq!(state.summary.last_message_preview.as_deref(), Some("world"));
    }

    #[test]
    fn test_composer_preserves_draft_on_error() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        state.set_draft("hi");
        state.start_sending();
        assert!(state.composer.sending);

        state.finish_sending_error("nope");
        assert!(!state.composer.sending);
        assert_eq!(state.composer.draft, "hi");
        assert_eq!(state.composer.last_error.as_deref(), Some("nope"));
    }

    #[test]
    fn test_composer_clears_draft_on_success() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        state.set_draft("hi");
        state.start_sending();
        state.finish_sending_success();
        assert_eq!(state.composer.draft, "");
        assert!(state.composer.last_error.is_none());
        assert!(state.composer.conflict_prompt.is_none());
    }

    #[test]
    fn test_composer_sets_conflict_prompt() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        state.set_draft("hi");
        state.start_sending();
        state.finish_sending_conflict("structured_turn_in_progress", "turn in progress");

        assert!(!state.composer.sending);
        assert_eq!(state.composer.draft, "hi");
        assert!(state.composer.last_error.is_none());
        assert_eq!(
            state
                .composer
                .conflict_prompt
                .as_ref()
                .map(|p| p.code.as_str()),
            Some("structured_turn_in_progress")
        );
    }

    #[test]
    fn test_live_event_unseen_counter_and_indicator() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        state.set_at_bottom(false);

        let id = SessionEventId::new();
        state.apply_live_event(user_event(session_id, id, ts(1), "hello"));

        assert_eq!(state.scroll.unseen_count, 1);
        let items = state.timeline_items();
        assert!(matches!(
            items.last(),
            Some(SessionTimelineItem::NewMessages(_))
        ));
    }

    #[test]
    fn test_live_event_scroll_to_bottom_intent() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        state.set_at_bottom(true);

        let id = SessionEventId::new();
        state.apply_live_event(user_event(session_id, id, ts(1), "hello"));

        assert!(state.scroll.pending_scroll_to_bottom);
        assert_eq!(state.scroll.unseen_count, 0);
    }

    #[test]
    fn test_live_session_event_appends_ephemeral_and_clears_on_assistant_message() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        state.set_at_bottom(true);

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(1),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantMessageDelta(
                redesmyn_protocol::session_live::AssistantMessageDelta {
                    delta: "hello".to_owned(),
                },
            ),
        });

        assert_eq!(state.ephemeral.len(), 1);
        assert_eq!(
            state.ephemeral.get("item_1").map(|item| item.text.as_str()),
            Some("hello")
        );

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(2),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantMessageDelta(
                redesmyn_protocol::session_live::AssistantMessageDelta {
                    delta: " world".to_owned(),
                },
            ),
        });

        assert_eq!(
            state.ephemeral.get("item_1").map(|item| item.text.as_str()),
            Some("hello world")
        );

        state.apply_live_event(assistant_event(
            session_id,
            SessionEventId::new(),
            ts(10),
            "final",
        ));

        assert!(state.ephemeral.is_empty());
    }

    #[test]
    fn test_live_tool_output_delta_appends_ephemeral_and_clears_on_tool_result() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        state.set_at_bottom(true);

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(1),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::ToolOutputDelta(
                redesmyn_protocol::session_live::ToolOutputDelta {
                    tool_name: "exec_command".to_owned(),
                    delta: "hello".to_owned(),
                },
            ),
        });

        assert_eq!(state.ephemeral.len(), 1);
        assert_eq!(
            state.ephemeral.get("item_1").map(|item| item.text.as_str()),
            Some("exec_command\nhello")
        );

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(2),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::ToolOutputDelta(
                redesmyn_protocol::session_live::ToolOutputDelta {
                    tool_name: "exec_command".to_owned(),
                    delta: " world".to_owned(),
                },
            ),
        });

        assert_eq!(
            state.ephemeral.get("item_1").map(|item| item.text.as_str()),
            Some("exec_command\nhello world")
        );

        state.apply_live_event(tool_result_event(
            session_id,
            SessionEventId::new(),
            ts(10),
            "item_1",
            "done",
        ));

        assert!(state.ephemeral.is_empty());
    }

    #[test]
    fn test_dedup_by_event_id() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        let id = SessionEventId::new();
        let event = user_event(session_id, id, ts(1), "hello");
        state.apply_live_event(event.clone());
        state.apply_live_event(event);
        assert_eq!(state.events.len(), 1);
    }

    #[test]
    fn test_out_of_order_insertion_keeps_sorting() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        let first = user_event(session_id, SessionEventId::new(), ts(10), "first");
        let second = user_event(session_id, SessionEventId::new(), ts(5), "second");
        state.apply_live_event(first);
        state.apply_live_event(second);

        assert_eq!(state.events.len(), 2);
        assert!(state.events[0].cursor.created_at < state.events[1].cursor.created_at);
    }
}
