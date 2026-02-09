//! Session viewer presentation model (pure logic; no GPUI).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use redesmyn_ids::{ArtifactId, SessionEventId, SessionId};
use redesmyn_protocol::client::SessionEventCursor;
use redesmyn_protocol::session::{
    ArtifactEmitted, AssistantMessage, CodexApprovalPolicy, CodexApprovalPolicyChanged,
    CodexSandboxPolicy, CodexSandboxPolicyChanged, ImageAttachment, PermissionDecided,
    PermissionRequested, PermissionsMode, PermissionsModeChanged, SessionEventKind, StatusUpdate,
    ToolInvocation, ToolResult, TurnCompleted, TurnStarted, UserMessage,
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
    AssistantReasoning,
    ToolInvocation,
    ToolResult,
    StatusUpdate,
    PermissionsModeChanged,
    CodexApprovalPolicyChanged,
    CodexSandboxPolicyChanged,
    PermissionRequested,
    PermissionDecided,
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
            SessionEventKind::AssistantReasoning(_) => Self::AssistantReasoning,
            SessionEventKind::ToolInvocation(_) => Self::ToolInvocation,
            SessionEventKind::ToolResult(_) => Self::ToolResult,
            SessionEventKind::StatusUpdate(_) => Self::StatusUpdate,
            SessionEventKind::TaskAgentMessageSent(_) => Self::Unknown,
            SessionEventKind::PermissionsModeChanged(_) => Self::PermissionsModeChanged,
            SessionEventKind::CodexApprovalPolicyChanged(_) => Self::CodexApprovalPolicyChanged,
            SessionEventKind::CodexSandboxPolicyChanged(_) => Self::CodexSandboxPolicyChanged,
            SessionEventKind::PermissionRequested(_) => Self::PermissionRequested,
            SessionEventKind::PermissionDecided(_) => Self::PermissionDecided,
            SessionEventKind::ArtifactEmitted(_) => Self::ArtifactEmitted,
            SessionEventKind::SessionModelChanged(_) => Self::Unknown,
            SessionEventKind::Unknown(_) => Self::Unknown,
        }
    }

    #[must_use]
    pub fn shows_in_timeline(self) -> bool {
        matches!(
            self,
            Self::UserMessage
                | Self::AssistantMessage
                | Self::AssistantReasoning
                | Self::ToolInvocation
                | Self::ToolResult
                | Self::PermissionRequested
                | Self::PermissionDecided
                | Self::ArtifactEmitted
        )
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
    EphemeralReasoning(EphemeralReasoningItem),
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
    AssistantReasoning(ReasoningItem),
    ToolInvocation(ToolInvocation),
    ToolResult(ToolResult),
    StatusUpdate(StatusUpdate),
    PermissionsModeChanged(PermissionsModeChanged),
    CodexApprovalPolicyChanged(CodexApprovalPolicyChanged),
    CodexSandboxPolicyChanged(CodexSandboxPolicyChanged),
    PermissionRequested(PermissionRequested),
    PermissionDecided(PermissionDecided),
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_attachments: Vec<ImageAttachment>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TextBlockItem {
    pub text: String,
    pub preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_text_artifact: Option<ArtifactRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReasoningItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    pub summary: TextBlockItem,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<TextBlockItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    pub role: SessionMessageRole,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EphemeralReasoningItem {
    pub key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub summary: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub raw: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EphemeralItem {
    Text(EphemeralTextItem),
    Reasoning(EphemeralReasoningItem),
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub image_attachments: Vec<ImageAttachment>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionFeedState {
    pub session_id: SessionId,
    #[serde(default = "default_permissions_mode")]
    pub permissions_mode: PermissionsMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_approval_policy: Option<CodexApprovalPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex_sandbox_policy: Option<CodexSandboxPolicy>,
    events: Vec<SessionEventRow>,
    event_ids: BTreeSet<SessionEventId>,
    ephemeral: BTreeMap<String, EphemeralItem>,
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
            permissions_mode: PermissionsMode::Ask,
            codex_approval_policy: None,
            codex_sandbox_policy: None,
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

        let mut event_items: Vec<SessionEventItem> =
            self.events.iter().map(SessionEventItem::from_row).collect();
        reorder_reasoning_before_assistant_messages(&mut event_items);

        let mut assistant_message_ix_by_turn: HashMap<String, usize> = HashMap::new();
        for event in event_items {
            if !event.kind.shows_in_timeline() {
                continue;
            }

            let ix = items.len();
            if event.kind == SessionEventKindTag::AssistantMessage
                && let Some(turn_id) = event.turn_id.as_ref()
            {
                assistant_message_ix_by_turn
                    .entry(turn_id.clone())
                    .or_insert(ix);
            }
            items.push(SessionTimelineItem::Event(event));
        }

        let mut tail_ephemeral_reasoning: Vec<SessionTimelineItem> = Vec::new();
        let mut tail_ephemeral_tool: Vec<SessionTimelineItem> = Vec::new();
        let mut tail_ephemeral_assistant: Vec<SessionTimelineItem> = Vec::new();
        let mut ephemeral_insertions: Vec<(usize, SessionTimelineItem)> = Vec::new();

        for ephemeral in self.ephemeral.values().cloned() {
            match ephemeral {
                EphemeralItem::Reasoning(item) => {
                    if let Some(turn_id) = item.turn_id.as_deref()
                        && let Some(&ix) = assistant_message_ix_by_turn.get(turn_id)
                    {
                        ephemeral_insertions
                            .push((ix, SessionTimelineItem::EphemeralReasoning(item)));
                    } else {
                        tail_ephemeral_reasoning
                            .push(SessionTimelineItem::EphemeralReasoning(item));
                    }
                }
                EphemeralItem::Text(item) => match item.role {
                    SessionMessageRole::Tool => {
                        tail_ephemeral_tool.push(SessionTimelineItem::EphemeralText(item));
                    }
                    SessionMessageRole::Assistant => {
                        tail_ephemeral_assistant.push(SessionTimelineItem::EphemeralText(item));
                    }
                    SessionMessageRole::User => {
                        tail_ephemeral_assistant.push(SessionTimelineItem::EphemeralText(item));
                    }
                },
            }
        }

        ephemeral_insertions.sort_by(|a, b| b.0.cmp(&a.0));
        for (ix, item) in ephemeral_insertions {
            items.insert(ix, item);
        }

        items.extend(tail_ephemeral_reasoning);
        items.extend(tail_ephemeral_tool);
        items.extend(tail_ephemeral_assistant);

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

    pub fn request_scroll_to_bottom(&mut self) {
        self.scroll.at_bottom = true;
        self.scroll.unseen_count = 0;
        self.scroll.pending_scroll_to_bottom = true;
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

        self.recompute_permissions_mode();
        self.recompute_codex_policies();
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
        match &event.kind {
            SessionEventKind::PermissionsModeChanged(ev) => {
                self.permissions_mode = ev.mode;
            }
            SessionEventKind::CodexApprovalPolicyChanged(ev) => {
                self.codex_approval_policy = ev.approval_policy;
            }
            SessionEventKind::CodexSandboxPolicyChanged(ev) => {
                self.codex_sandbox_policy = ev.sandbox_policy.clone();
            }
            _ => {}
        }

        let clear_assistant_ephemeral =
            matches!(&event.kind, SessionEventKind::AssistantMessage(_));
        let clear_reasoning_ephemeral = match &event.kind {
            SessionEventKind::AssistantReasoning(ev) => Some(ev.item_id.clone()),
            _ => None,
        };
        let clear_tool_ephemeral = match &event.kind {
            SessionEventKind::ToolResult(ev) => Some(ev.tool_call_id.clone()),
            _ => None,
        };
        let clear_all_ephemeral = matches!(&event.kind, SessionEventKind::TurnCompleted(_));
        let clear_ephemeral_turn_id = clear_all_ephemeral.then(|| event.turn_id.clone());
        let inserted = self.insert_event(SessionEventRow::from_event(event));

        let Some(inserted_at_end) = inserted else {
            return;
        };

        if clear_all_ephemeral {
            if let Some(turn_id) = clear_ephemeral_turn_id.flatten() {
                self.ephemeral.retain(|_, item| match item {
                    EphemeralItem::Text(item) => item.turn_id.as_deref() != Some(turn_id.as_str()),
                    EphemeralItem::Reasoning(item) => {
                        item.turn_id.as_deref() != Some(turn_id.as_str())
                    }
                });
            } else {
                self.ephemeral.clear();
            }
        }

        if inserted_at_end {
            if clear_assistant_ephemeral {
                self.ephemeral.retain(|_, item| match item {
                    EphemeralItem::Text(item) => item.role != SessionMessageRole::Assistant,
                    EphemeralItem::Reasoning(_) => true,
                });
            }
            if let Some(tool_call_id) = clear_tool_ephemeral {
                match tool_call_id {
                    Some(tool_call_id) => {
                        self.ephemeral.remove(&tool_call_id);
                    }
                    None => {
                        self.ephemeral.retain(|_, item| match item {
                            EphemeralItem::Text(item) => item.role != SessionMessageRole::Tool,
                            EphemeralItem::Reasoning(_) => true,
                        });
                    }
                }
            }
        }

        if let Some(reasoning_item_id) = clear_reasoning_ephemeral {
            match reasoning_item_id {
                Some(item_id) => {
                    self.ephemeral.remove(&item_id);
                }
                None if inserted_at_end => {
                    self.ephemeral.retain(|_, item| match item {
                        EphemeralItem::Reasoning(_) => false,
                        EphemeralItem::Text(_) => true,
                    });
                }
                None => {}
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

    fn normalize_reasoning_ephemeral_keys_for_turn(&mut self, key: &str, turn_id: Option<&str>) {
        let Some(turn_id) = turn_id else {
            return;
        };

        if key != "assistant_reasoning"
            && let Some(EphemeralItem::Reasoning(mut placeholder)) =
                self.ephemeral.remove("assistant_reasoning")
        {
            if placeholder.turn_id.as_deref() == Some(turn_id) {
                placeholder.key = key.to_owned();

                if let Some(EphemeralItem::Reasoning(existing)) = self.ephemeral.get_mut(key) {
                    if existing.turn_id.is_none() {
                        existing.turn_id = Some(turn_id.to_owned());
                    }

                    if existing.summary.len() < placeholder.summary.len() {
                        existing
                            .summary
                            .resize(placeholder.summary.len(), String::new());
                    }
                    for (ix, chunk) in placeholder.summary.into_iter().enumerate() {
                        existing.summary[ix].push_str(&chunk);
                    }

                    if existing.raw.len() < placeholder.raw.len() {
                        existing.raw.resize(placeholder.raw.len(), String::new());
                    }
                    for (ix, chunk) in placeholder.raw.into_iter().enumerate() {
                        existing.raw[ix].push_str(&chunk);
                    }

                    if existing.signature.is_none() {
                        existing.signature = placeholder.signature;
                    }
                } else {
                    self.ephemeral
                        .insert(key.to_owned(), EphemeralItem::Reasoning(placeholder));
                }
            } else {
                self.ephemeral.insert(
                    "assistant_reasoning".to_owned(),
                    EphemeralItem::Reasoning(placeholder),
                );
            }
        }

        self.ephemeral.retain(|item_key, item| match item {
            EphemeralItem::Reasoning(entry) => {
                entry.turn_id.as_deref() != Some(turn_id) || item_key.as_str() == key
            }
            EphemeralItem::Text(_) => true,
        });
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

                let needs_insert = !self
                    .ephemeral
                    .get(&key)
                    .is_some_and(|item| matches!(item, EphemeralItem::Text(_)));
                if needs_insert {
                    self.ephemeral.insert(
                        key.clone(),
                        EphemeralItem::Text(EphemeralTextItem {
                            key: key.clone(),
                            turn_id: event.turn_id.clone(),
                            role: SessionMessageRole::Assistant,
                            text: String::new(),
                        }),
                    );
                }

                if let Some(EphemeralItem::Text(entry)) = self.ephemeral.get_mut(&key) {
                    if entry.turn_id.is_none() {
                        entry.turn_id = event.turn_id.clone();
                    }
                    entry.text.push_str(&delta.delta);
                }
                if self.scroll.at_bottom {
                    self.scroll.pending_scroll_to_bottom = true;
                }
            }
            SessionLiveEventKind::AssistantReasoningSummaryPartAdded(part) => {
                let Some(summary_index) = usize::try_from(part.summary_index).ok() else {
                    return;
                };

                let key = event
                    .item_id
                    .clone()
                    .unwrap_or_else(|| "assistant_reasoning".to_owned());
                self.normalize_reasoning_ephemeral_keys_for_turn(&key, event.turn_id.as_deref());

                let needs_insert = !self
                    .ephemeral
                    .get(&key)
                    .is_some_and(|item| matches!(item, EphemeralItem::Reasoning(_)));
                if needs_insert {
                    self.ephemeral.insert(
                        key.clone(),
                        EphemeralItem::Reasoning(EphemeralReasoningItem {
                            key: key.clone(),
                            turn_id: event.turn_id.clone(),
                            summary: Vec::new(),
                            raw: Vec::new(),
                            signature: None,
                        }),
                    );
                }

                if let Some(EphemeralItem::Reasoning(entry)) = self.ephemeral.get_mut(&key) {
                    if entry.turn_id.is_none() {
                        entry.turn_id = event.turn_id.clone();
                    }
                    if entry.summary.len() <= summary_index {
                        entry.summary.resize(summary_index + 1, String::new());
                    }
                }
                if self.scroll.at_bottom {
                    self.scroll.pending_scroll_to_bottom = true;
                }
            }
            SessionLiveEventKind::AssistantReasoningSummaryDelta(delta) => {
                if delta.delta.is_empty() {
                    return;
                }

                let Some(summary_index) = usize::try_from(delta.summary_index).ok() else {
                    return;
                };

                let key = event
                    .item_id
                    .clone()
                    .unwrap_or_else(|| "assistant_reasoning".to_owned());
                self.normalize_reasoning_ephemeral_keys_for_turn(&key, event.turn_id.as_deref());

                let needs_insert = !self
                    .ephemeral
                    .get(&key)
                    .is_some_and(|item| matches!(item, EphemeralItem::Reasoning(_)));
                if needs_insert {
                    self.ephemeral.insert(
                        key.clone(),
                        EphemeralItem::Reasoning(EphemeralReasoningItem {
                            key: key.clone(),
                            turn_id: event.turn_id.clone(),
                            summary: Vec::new(),
                            raw: Vec::new(),
                            signature: None,
                        }),
                    );
                }

                if let Some(EphemeralItem::Reasoning(entry)) = self.ephemeral.get_mut(&key) {
                    if entry.turn_id.is_none() {
                        entry.turn_id = event.turn_id.clone();
                    }
                    if entry.summary.len() <= summary_index {
                        entry.summary.resize(summary_index + 1, String::new());
                    }
                    entry.summary[summary_index].push_str(&delta.delta);
                }
                if self.scroll.at_bottom {
                    self.scroll.pending_scroll_to_bottom = true;
                }
            }
            SessionLiveEventKind::AssistantReasoningRawDelta(delta) => {
                if delta.delta.is_empty() {
                    return;
                }

                let Some(content_index) = usize::try_from(delta.content_index).ok() else {
                    return;
                };

                let key = event
                    .item_id
                    .clone()
                    .unwrap_or_else(|| "assistant_reasoning".to_owned());
                self.normalize_reasoning_ephemeral_keys_for_turn(&key, event.turn_id.as_deref());

                let needs_insert = !self
                    .ephemeral
                    .get(&key)
                    .is_some_and(|item| matches!(item, EphemeralItem::Reasoning(_)));
                if needs_insert {
                    self.ephemeral.insert(
                        key.clone(),
                        EphemeralItem::Reasoning(EphemeralReasoningItem {
                            key: key.clone(),
                            turn_id: event.turn_id.clone(),
                            summary: Vec::new(),
                            raw: Vec::new(),
                            signature: None,
                        }),
                    );
                }

                if let Some(EphemeralItem::Reasoning(entry)) = self.ephemeral.get_mut(&key) {
                    if entry.turn_id.is_none() {
                        entry.turn_id = event.turn_id.clone();
                    }
                    if entry.raw.len() <= content_index {
                        entry.raw.resize(content_index + 1, String::new());
                    }
                    entry.raw[content_index].push_str(&delta.delta);
                }
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

                let needs_insert = !self
                    .ephemeral
                    .get(&key)
                    .is_some_and(|item| matches!(item, EphemeralItem::Text(_)));
                if needs_insert {
                    self.ephemeral.insert(
                        key.clone(),
                        EphemeralItem::Text(EphemeralTextItem {
                            key: key.clone(),
                            turn_id: event.turn_id.clone(),
                            role: SessionMessageRole::Tool,
                            text: format!("{}\n", delta.tool_name),
                        }),
                    );
                }

                if let Some(EphemeralItem::Text(entry)) = self.ephemeral.get_mut(&key) {
                    if entry.turn_id.is_none() {
                        entry.turn_id = event.turn_id.clone();
                    }
                    entry.text.push_str(&delta.delta);
                }
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

    pub fn add_composer_image_attachment(&mut self, attachment: ImageAttachment) {
        self.composer.image_attachments.push(attachment);
        self.composer.last_error = None;
        self.composer.conflict_prompt = None;
    }

    pub fn remove_composer_image_attachment(
        &mut self,
        artifact_id: ArtifactId,
    ) -> Option<ImageAttachment> {
        let index = self
            .composer
            .image_attachments
            .iter()
            .position(|attachment| attachment.artifact.artifact_id == artifact_id)?;
        self.composer.last_error = None;
        self.composer.conflict_prompt = None;
        Some(self.composer.image_attachments.remove(index))
    }

    pub fn start_sending(&mut self) {
        self.composer.sending = true;
        self.composer.last_error = None;
        self.composer.conflict_prompt = None;
    }

    pub fn finish_sending_success(&mut self) {
        self.composer.sending = false;
        self.composer.draft.clear();
        self.composer.image_attachments.clear();
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
            EphemeralItem::Text(EphemeralTextItem {
                key,
                turn_id: None,
                role,
                text: text.into(),
            }),
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

    fn recompute_permissions_mode(&mut self) {
        for row in self.events.iter().rev() {
            if let SessionEventKind::PermissionsModeChanged(ev) = &row.event.kind {
                self.permissions_mode = ev.mode;
                return;
            }
        }

        self.permissions_mode = PermissionsMode::Ask;
    }

    fn recompute_codex_policies(&mut self) {
        let mut found_approval = false;
        let mut found_sandbox = false;
        let mut approval = None;
        let mut sandbox = None;

        for row in self.events.iter().rev() {
            match &row.event.kind {
                SessionEventKind::CodexApprovalPolicyChanged(ev) if !found_approval => {
                    approval = ev.approval_policy;
                    found_approval = true;
                }
                SessionEventKind::CodexSandboxPolicyChanged(ev) if !found_sandbox => {
                    sandbox = ev.sandbox_policy.clone();
                    found_sandbox = true;
                }
                _ => {}
            }

            if found_approval && found_sandbox {
                break;
            }
        }

        self.codex_approval_policy = approval;
        self.codex_sandbox_policy = sandbox;
    }
}

fn default_permissions_mode() -> PermissionsMode {
    PermissionsMode::Ask
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
                    image_attachments: ev.image_attachments.clone(),
                })
            }
            SessionEventKind::AssistantMessage(ev) => {
                SessionEventItemContent::AssistantMessage(MessageItem {
                    role: SessionMessageRole::Assistant,
                    text: ev.text.clone(),
                    preview: ev.preview.clone(),
                    full_text_artifact: ev.full_text_artifact.clone(),
                    image_attachments: Vec::new(),
                })
            }
            SessionEventKind::AssistantReasoning(ev) => {
                let summary = TextBlockItem {
                    text: ev.summary.text.clone(),
                    preview: ev.summary.preview.clone(),
                    full_text_artifact: ev.summary.full_text_artifact.clone(),
                };
                let raw = ev.raw.as_ref().map(|raw| TextBlockItem {
                    text: raw.text.clone(),
                    preview: raw.preview.clone(),
                    full_text_artifact: raw.full_text_artifact.clone(),
                });

                SessionEventItemContent::AssistantReasoning(ReasoningItem {
                    item_id: ev.item_id.clone(),
                    summary,
                    raw,
                    signature: ev.signature.clone(),
                })
            }
            SessionEventKind::ToolInvocation(ev) => {
                SessionEventItemContent::ToolInvocation(ev.clone())
            }
            SessionEventKind::ToolResult(ev) => SessionEventItemContent::ToolResult(ev.clone()),
            SessionEventKind::StatusUpdate(ev) => SessionEventItemContent::StatusUpdate(ev.clone()),
            SessionEventKind::TaskAgentMessageSent(_) => SessionEventItemContent::Unknown,
            SessionEventKind::PermissionsModeChanged(ev) => {
                SessionEventItemContent::PermissionsModeChanged(ev.clone())
            }
            SessionEventKind::CodexApprovalPolicyChanged(ev) => {
                SessionEventItemContent::CodexApprovalPolicyChanged(ev.clone())
            }
            SessionEventKind::CodexSandboxPolicyChanged(ev) => {
                SessionEventItemContent::CodexSandboxPolicyChanged(ev.clone())
            }
            SessionEventKind::PermissionRequested(ev) => {
                SessionEventItemContent::PermissionRequested(ev.clone())
            }
            SessionEventKind::PermissionDecided(ev) => {
                SessionEventItemContent::PermissionDecided(ev.clone())
            }
            SessionEventKind::ArtifactEmitted(ev) => {
                SessionEventItemContent::ArtifactEmitted(ArtifactEmittedItem {
                    artifact: ev.artifact.clone(),
                    label: ev.label.clone(),
                })
            }
            SessionEventKind::SessionModelChanged(_) => SessionEventItemContent::Unknown,
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

fn reorder_reasoning_before_assistant_messages(items: &mut Vec<SessionEventItem>) {
    let mut assistant_message_ix_by_turn: HashMap<&str, usize> = HashMap::new();
    for (ix, item) in items.iter().enumerate() {
        if item.kind == SessionEventKindTag::AssistantMessage
            && let Some(turn_id) = item.turn_id.as_deref()
        {
            assistant_message_ix_by_turn.entry(turn_id).or_insert(ix);
        }
    }

    let mut deferred: HashMap<&str, Vec<SessionEventItem>> = HashMap::new();
    for (ix, item) in items.iter().enumerate() {
        if item.kind != SessionEventKindTag::AssistantReasoning {
            continue;
        }
        let Some(turn_id) = item.turn_id.as_deref() else {
            continue;
        };
        let Some(message_ix) = assistant_message_ix_by_turn.get(turn_id) else {
            continue;
        };
        if ix > *message_ix {
            deferred.entry(turn_id).or_default().push(item.clone());
        }
    }

    if deferred.is_empty() {
        return;
    }

    let mut out = Vec::with_capacity(items.len());
    for (ix, item) in items.iter().enumerate() {
        if item.kind == SessionEventKindTag::AssistantMessage
            && let Some(turn_id) = item.turn_id.as_deref()
            && let Some(reasoning) = deferred.remove(turn_id)
        {
            out.extend(reasoning);
        }

        if item.kind == SessionEventKindTag::AssistantReasoning
            && let Some(turn_id) = item.turn_id.as_deref()
            && let Some(message_ix) = assistant_message_ix_by_turn.get(turn_id)
            && ix > *message_ix
        {
            continue;
        }

        out.push(item.clone());
    }

    for remaining in deferred.into_values() {
        out.extend(remaining);
    }

    *items = out;
}

fn preview_from_event_kind(kind: &SessionEventKind) -> Option<String> {
    match kind {
        SessionEventKind::UserMessage(UserMessage { preview, .. }) => Some(preview.clone()),
        SessionEventKind::AssistantMessage(AssistantMessage { preview, .. }) => {
            Some(preview.clone())
        }
        SessionEventKind::AssistantReasoning(ev) => Some(ev.summary.preview.clone()),
        SessionEventKind::ToolInvocation(ToolInvocation { input_preview, .. }) => {
            Some(input_preview.clone())
        }
        SessionEventKind::ToolResult(ToolResult { output_preview, .. }) => {
            Some(output_preview.clone())
        }
        SessionEventKind::StatusUpdate(StatusUpdate { message, .. }) => message.clone(),
        SessionEventKind::TaskAgentMessageSent(ev) => Some(ev.message_preview.clone()),
        SessionEventKind::PermissionsModeChanged(ev) => Some(
            match ev.mode {
                redesmyn_protocol::session::PermissionsMode::Ask => "ask",
                redesmyn_protocol::session::PermissionsMode::AutoApprove => "auto_approve",
                redesmyn_protocol::session::PermissionsMode::Deny => "deny",
                redesmyn_protocol::session::PermissionsMode::Unknown => "unknown",
            }
            .to_owned(),
        ),
        SessionEventKind::CodexApprovalPolicyChanged(ev) => Some(
            match ev.approval_policy {
                None => "default",
                Some(redesmyn_protocol::session::CodexApprovalPolicy::UnlessTrusted) => "untrusted",
                Some(redesmyn_protocol::session::CodexApprovalPolicy::OnFailure) => "on_failure",
                Some(redesmyn_protocol::session::CodexApprovalPolicy::OnRequest) => "on_request",
                Some(redesmyn_protocol::session::CodexApprovalPolicy::Never) => "never",
                Some(redesmyn_protocol::session::CodexApprovalPolicy::Unknown) => "unknown",
            }
            .to_owned(),
        ),
        SessionEventKind::CodexSandboxPolicyChanged(ev) => Some(
            match ev.sandbox_policy.as_ref() {
                None => "default",
                Some(redesmyn_protocol::session::CodexSandboxPolicy::DangerFullAccess) => {
                    "danger_full_access"
                }
                Some(redesmyn_protocol::session::CodexSandboxPolicy::ReadOnly) => "read_only",
                Some(redesmyn_protocol::session::CodexSandboxPolicy::ExternalSandbox {
                    ..
                }) => "external_sandbox",
                Some(redesmyn_protocol::session::CodexSandboxPolicy::WorkspaceWrite { .. }) => {
                    "workspace_write"
                }
                Some(redesmyn_protocol::session::CodexSandboxPolicy::Unknown) => "unknown",
            }
            .to_owned(),
        ),
        SessionEventKind::PermissionRequested(ev) => Some(ev.summary.clone()),
        SessionEventKind::PermissionDecided(ev) => Some(
            match ev.decision {
                redesmyn_protocol::session::PermissionDecision::Approve => "approve",
                redesmyn_protocol::session::PermissionDecision::Deny => "deny",
                redesmyn_protocol::session::PermissionDecision::Unknown => "unknown",
            }
            .to_owned(),
        ),
        SessionEventKind::SessionModelChanged(ev) => Some(format!(
            "{}:{}",
            ev.model_id.as_deref().unwrap_or("default"),
            match ev.reasoning_effort {
                None => "default",
                Some(redesmyn_protocol::session::SessionModelReasoningEffort::Minimal) => {
                    "minimal"
                }
                Some(redesmyn_protocol::session::SessionModelReasoningEffort::Low) => "low",
                Some(redesmyn_protocol::session::SessionModelReasoningEffort::Medium) => "medium",
                Some(redesmyn_protocol::session::SessionModelReasoningEffort::High) => "high",
                Some(redesmyn_protocol::session::SessionModelReasoningEffort::Xhigh) => "xhigh",
                Some(redesmyn_protocol::session::SessionModelReasoningEffort::Unknown) => {
                    "unknown"
                }
            }
        )),
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

    fn sample_image_attachment() -> ImageAttachment {
        let artifact_id = ArtifactId::new();
        ImageAttachment {
            artifact: ArtifactRef {
                artifact_id,
                kind: redesmyn_protocol::ArtifactKind::Image,
                content_hash: None,
                byte_len: Some(128),
                mime: Some("image/png".to_string()),
                storage_hint: Some(redesmyn_protocol::StorageHint::BlobKey {
                    blob_key: format!("artifact/{artifact_id}"),
                }),
            },
            label: Some("image.png".to_string()),
        }
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
                image_attachments: Vec::new(),
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

    fn assistant_event_with_turn(
        session_id: SessionId,
        id: SessionEventId,
        created_at: Timestamp,
        turn_id: &str,
        preview: &str,
    ) -> SessionEvent {
        SessionEvent {
            session_event_id: id,
            created_at,
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id.to_owned()),
            kind: SessionEventKind::AssistantMessage(AssistantMessage {
                text: preview.to_string(),
                preview: preview.to_string(),
                full_text_artifact: None,
            }),
        }
    }

    fn assistant_reasoning_event_with_turn(
        session_id: SessionId,
        id: SessionEventId,
        created_at: Timestamp,
        turn_id: &str,
        item_id: &str,
        summary_preview: &str,
    ) -> SessionEvent {
        SessionEvent {
            session_event_id: id,
            created_at,
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id.to_owned()),
            kind: SessionEventKind::AssistantReasoning(
                redesmyn_protocol::session::AssistantReasoning {
                    item_id: Some(item_id.to_owned()),
                    summary: redesmyn_protocol::session::AssistantReasoningText {
                        text: summary_preview.to_owned(),
                        preview: summary_preview.to_owned(),
                        full_text_artifact: None,
                    },
                    raw: None,
                    signature: None,
                },
            ),
        }
    }

    fn turn_completed_event_with_turn(
        session_id: SessionId,
        id: SessionEventId,
        created_at: Timestamp,
        turn_id: &str,
    ) -> SessionEvent {
        SessionEvent {
            session_event_id: id,
            created_at,
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: Some(turn_id.to_owned()),
            kind: SessionEventKind::TurnCompleted(TurnCompleted {
                interface_mode: redesmyn_protocol::session::InterfaceMode::Structured,
                external_session_ref: None,
                exit_code: Some(0),
                error: None,
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
        state.add_composer_image_attachment(sample_image_attachment());
        state.start_sending();
        state.finish_sending_success();
        assert_eq!(state.composer.draft, "");
        assert!(state.composer.image_attachments.is_empty());
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
    fn test_composer_remove_image_attachment() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        let attachment = sample_image_attachment();
        let artifact_id = attachment.artifact.artifact_id;

        state.add_composer_image_attachment(attachment.clone());
        assert_eq!(state.composer.image_attachments.len(), 1);

        let removed = state.remove_composer_image_attachment(artifact_id);
        assert_eq!(removed, Some(attachment));
        assert!(state.composer.image_attachments.is_empty());
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
        let Some(EphemeralItem::Text(item)) = state.ephemeral.get("item_1") else {
            panic!("expected ephemeral text item");
        };
        assert_eq!(item.text, "hello");

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

        let Some(EphemeralItem::Text(item)) = state.ephemeral.get("item_1") else {
            panic!("expected ephemeral text item");
        };
        assert_eq!(item.text, "hello world");

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
        let Some(EphemeralItem::Text(item)) = state.ephemeral.get("item_1") else {
            panic!("expected ephemeral text item");
        };
        assert_eq!(item.text, "exec_command\nhello");

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

        let Some(EphemeralItem::Text(item)) = state.ephemeral.get("item_1") else {
            panic!("expected ephemeral text item");
        };
        assert_eq!(item.text, "exec_command\nhello world");

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
    fn test_live_reasoning_deltas_append_ephemeral_and_clear_on_durable_reasoning() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);
        state.set_at_bottom(true);

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(1),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryPartAdded(
                redesmyn_protocol::session_live::AssistantReasoningSummaryPartAdded {
                    summary_index: 0,
                },
            ),
        });

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(2),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryDelta(
                redesmyn_protocol::session_live::AssistantReasoningSummaryDelta {
                    summary_index: 0,
                    delta: "thinking".to_owned(),
                },
            ),
        });

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(3),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningRawDelta(
                redesmyn_protocol::session_live::AssistantReasoningRawDelta {
                    content_index: 0,
                    delta: "raw".to_owned(),
                },
            ),
        });

        let Some(EphemeralItem::Reasoning(item)) = state.ephemeral.get("item_1") else {
            panic!("expected ephemeral reasoning item");
        };
        assert_eq!(item.turn_id.as_deref(), Some("turn_1"));
        assert_eq!(item.summary, vec!["thinking".to_string()]);
        assert_eq!(item.raw, vec!["raw".to_string()]);

        state.apply_live_event(SessionEvent {
            session_event_id: SessionEventId::new(),
            created_at: ts(10),
            scope: redesmyn_protocol::SessionScope::Chat,
            session_id,
            turn_id: Some("turn_1".to_owned()),
            kind: SessionEventKind::AssistantReasoning(
                redesmyn_protocol::session::AssistantReasoning {
                    item_id: Some("item_1".to_owned()),
                    summary: redesmyn_protocol::session::AssistantReasoningText {
                        text: "summary".to_owned(),
                        preview: "summary".to_owned(),
                        full_text_artifact: None,
                    },
                    raw: None,
                    signature: None,
                },
            ),
        });

        assert!(state.ephemeral.is_empty());
    }

    #[test]
    fn test_timeline_orders_reasoning_before_assistant_message_in_turn() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        state.apply_live_event(assistant_event_with_turn(
            session_id,
            SessionEventId::new(),
            ts(1),
            "turn_1",
            "assistant",
        ));
        state.apply_live_event(assistant_reasoning_event_with_turn(
            session_id,
            SessionEventId::new(),
            ts(2),
            "turn_1",
            "item_1",
            "thinking",
        ));

        let items = state.timeline_items();
        let reasoning_ix = items
            .iter()
            .position(|item| matches!(item, SessionTimelineItem::Event(ev) if ev.kind == SessionEventKindTag::AssistantReasoning))
            .expect("reasoning event missing");
        let message_ix = items
            .iter()
            .position(|item| matches!(item, SessionTimelineItem::Event(ev) if ev.kind == SessionEventKindTag::AssistantMessage))
            .expect("assistant message missing");

        assert!(
            reasoning_ix < message_ix,
            "expected reasoning before message, got {reasoning_ix} >= {message_ix}"
        );
    }

    #[test]
    fn test_timeline_inserts_ephemeral_reasoning_before_assistant_message_in_turn() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(1),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryPartAdded(
                redesmyn_protocol::session_live::AssistantReasoningSummaryPartAdded {
                    summary_index: 0,
                },
            ),
        });

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(2),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryDelta(
                redesmyn_protocol::session_live::AssistantReasoningSummaryDelta {
                    summary_index: 0,
                    delta: "thinking".to_owned(),
                },
            ),
        });

        state.apply_live_event(assistant_event_with_turn(
            session_id,
            SessionEventId::new(),
            ts(10),
            "turn_1",
            "assistant",
        ));

        let items = state.timeline_items();
        let ephemeral_reasoning_ix = items
            .iter()
            .position(|item| matches!(item, SessionTimelineItem::EphemeralReasoning(ephemeral) if ephemeral.key == "item_1"))
            .expect("ephemeral reasoning missing");
        let message_ix = items
            .iter()
            .position(|item| matches!(item, SessionTimelineItem::Event(ev) if ev.kind == SessionEventKindTag::AssistantMessage && ev.turn_id.as_deref() == Some("turn_1")))
            .expect("assistant message missing");

        assert!(
            ephemeral_reasoning_ix < message_ix,
            "expected ephemeral reasoning before message, got {ephemeral_reasoning_ix} >= {message_ix}"
        );
    }

    #[test]
    fn test_turn_completed_clears_ephemeral_even_if_inserted_out_of_order() {
        let session_id = SessionId::new();
        let mut state = SessionFeedState::new(session_id);

        state.apply_live_event(assistant_event_with_turn(
            session_id,
            SessionEventId::new(),
            ts(10),
            "turn_1",
            "assistant",
        ));

        state.apply_live_session_event(redesmyn_protocol::session_live::SessionLiveEvent {
            created_at: ts(9),
            session_id,
            turn_id: Some("turn_1".to_owned()),
            item_id: Some("item_1".to_owned()),
            kind: redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryPartAdded(
                redesmyn_protocol::session_live::AssistantReasoningSummaryPartAdded {
                    summary_index: 0,
                },
            ),
        });

        assert!(
            !state.ephemeral.is_empty(),
            "expected ephemeral state before TurnCompleted"
        );

        // Insert TurnCompleted *out of order* (cursor earlier than the current tail).
        state.apply_live_event(turn_completed_event_with_turn(
            session_id,
            SessionEventId::new(),
            ts(5),
            "turn_1",
        ));

        assert!(
            state.ephemeral.is_empty(),
            "expected TurnCompleted to clear ephemeral state"
        );
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
