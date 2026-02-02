//! Session viewer scaffolding for GPUI (Domain 7).

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::{
    App, AsyncApp, ClickEvent, ClipboardItem, Context, ElementId, Entity, FocusHandle, Focusable,
    ListOffset, ListState, Render, ScrollHandle, SharedString, Subscription, Task, WeakEntity,
    Window, div, list, px, relative,
};

use gpui::prelude::*;

use redesmyn_client_api::Client;
use redesmyn_ids::{SessionEventId, SessionId, SubscriptionId};
use redesmyn_markdown::{MarkdownDoc, MarkdownParseOptions, parse_markdown};
use redesmyn_protocol::client::{
    AgentMessageConflictAction, GetSessionEventsRequest, GetSessionEventsResponse, RequestPayload,
    ResponseResult, SendSessionMessageRequest, SendSessionMessageResponse, SessionEventCursor,
    SubscriptionEvent,
};
use redesmyn_protocol::session::{SessionEventKind, ToolResult};
use redesmyn_protocol::ui_driver::UiComposerState;
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, SessionEvent};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;

use redesmyn_session_view_model::{SessionEventItemContent, SessionFeedState, SessionTimelineItem};
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, MarkdownView, TextArea, TextButton, TextInput,
    TextInputEvent,
};
use redesmyn_ui::styles::ThemeMode;
use redesmyn_ui::utils::{
    UiActivityGuard, theme_for_window, ui_idle_tracker, ui_test_mode_animation_duration,
};

use serde_json::Value as JsonValue;

fn session_event_id_key(id: SessionEventId) -> u64 {
    let bytes = id.to_bytes();
    u64::from_be_bytes(bytes[8..16].try_into().expect("slice length"))
}

fn stable_str_key(input: &str) -> u64 {
    // Deterministic hashing for element IDs (avoid RandomState).
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn ease_out_cubic(t: f32) -> f32 {
    1.0 - (1.0 - t).powi(3)
}

fn should_autoscroll_to_bottom(handle: &ScrollHandle) -> bool {
    let max = handle.max_offset().height;
    if max <= px(0.0) {
        return true;
    }

    let offset = handle.offset().y;
    offset <= (-max + px(4.0))
}

fn chain_scroll_wheel_to_timeline_list_if_needed(
    event: &gpui::ScrollWheelEvent,
    window: &mut Window,
    cx: &mut App,
    scroll_handle: &ScrollHandle,
    timeline_view: Entity<SessionView>,
) {
    let delta_y = event.delta.pixel_delta(window.line_height()).y;
    if delta_y == px(0.0) {
        return;
    }

    let max = scroll_handle.max_offset().height;
    if max <= px(0.0) {
        timeline_view.update(cx, move |this, cx| {
            this.timeline_follow_bottom.set(false);
            this.timeline_list_state.scroll_by(-delta_y);
            this.on_timeline_scrolled(0, 0, cx);
            cx.notify();
        });
        return;
    }

    // `div`'s built-in scroll listener runs earlier in the same bubble phase and updates the
    // tracked `ScrollHandle` by adding `delta_y` to its offset. Use that to reconstruct the
    // pre-scroll offset so we only chain to the parent list once this scroll view was already
    // at the boundary.
    let offset_y_after = scroll_handle.offset().y;
    let offset_y_before = (offset_y_after - delta_y).clamp(-max, px(0.0));
    let threshold = px(1.0);

    let allow_parent_scroll = if delta_y > px(0.0) {
        // Scroll up: allow the parent list to scroll only if this scroll view is already at top.
        offset_y_before >= -threshold
    } else {
        // Scroll down: allow the parent list to scroll only if this scroll view is already at bottom.
        offset_y_before <= (-max + threshold)
    };

    if !allow_parent_scroll {
        return;
    }

    // Keep the inner scroll view pinned at the boundary so the chained scroll doesn't produce
    // a visible "rubber band" as the offset snaps back during the next paint.
    let clamp_y = if delta_y > px(0.0) { px(0.0) } else { -max };
    let offset = scroll_handle.offset();
    scroll_handle.set_offset(gpui::point(offset.x, clamp_y));

    timeline_view.update(cx, move |this, cx| {
        this.timeline_follow_bottom.set(false);
        this.timeline_list_state.scroll_by(-delta_y);
        this.on_timeline_scrolled(0, 0, cx);
        cx.notify();
    });
}

struct AutoscrollMarker {
    enabled: bool,
}

impl AutoscrollMarker {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }
}

impl gpui::IntoElement for AutoscrollMarker {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl gpui::Element for AutoscrollMarker {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&gpui::GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        _cx: &mut App,
    ) -> (gpui::LayoutId, Self::RequestLayoutState) {
        let height = px(1.0);
        let layout_id =
            window.request_measured_layout(Default::default(), move |known, avail, _, _| {
                let width = known.width.or(match avail.width {
                    gpui::AvailableSpace::Definite(width) => Some(width),
                    _ => None,
                });
                gpui::size(width.unwrap_or(px(0.0)), height)
            });
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&gpui::GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        bounds: gpui::Bounds<gpui::Pixels>,
        _: &mut Self::RequestLayoutState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        if self.enabled {
            window.request_autoscroll(bounds);
        }
    }

    fn paint(
        &mut self,
        _id: Option<&gpui::GlobalElementId>,
        _inspector_id: Option<&gpui::InspectorElementId>,
        _bounds: gpui::Bounds<gpui::Pixels>,
        _: &mut Self::RequestLayoutState,
        _: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

#[derive(Clone)]
struct RawThoughtScrollState {
    handle: ScrollHandle,
    follow_bottom: bool,
    last_offset_y: gpui::Pixels,
}

impl RawThoughtScrollState {
    fn new() -> Self {
        Self {
            handle: ScrollHandle::new(),
            follow_bottom: true,
            last_offset_y: px(0.0),
        }
    }
}

#[derive(Clone)]
struct ReasoningScrollState {
    summary: RawThoughtScrollState,
    raw: RawThoughtScrollState,
}

impl ReasoningScrollState {
    fn new() -> Self {
        Self {
            summary: RawThoughtScrollState::new(),
            raw: RawThoughtScrollState::new(),
        }
    }
}

const CONFLICT_CODE_KEY: &str = "conflict_code";
const CONFLICT_CODE_TURN_IN_PROGRESS: &str = "structured_turn_in_progress";
// NOTE: `conflict_code` values are part of the client-visible contract.
// `structured_session_conflict` is currently returned for any concurrent task session (even if the
// conflicting session isn't structured). Consider renaming if we need semantic precision.
const CONFLICT_CODE_TASK_SESSION_CONFLICT: &str = "structured_session_conflict";

fn extract_json_string_field(preview: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = preview.find(&needle)?;
    let mut i = start + needle.len();

    let bytes = preview.as_bytes();
    while i < bytes.len() && bytes[i] != b':' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    i += 1;

    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'"' {
        return None;
    }
    i += 1;

    let mut out = String::new();
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(out),
            b'\\' => {
                i += 1;
                if i >= bytes.len() {
                    break;
                }
                match bytes[i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if i + 4 < bytes.len() {
                            let hex = &preview[i + 1..i + 5];
                            if let Ok(value) = u16::from_str_radix(hex, 16)
                                && let Some(ch) = char::from_u32(value.into())
                            {
                                out.push(ch);
                                i += 4;
                            }
                        }
                    }
                    other => out.push(other as char),
                }
            }
            other => out.push(other as char),
        }
        i += 1;
    }

    if out.is_empty() { None } else { Some(out) }
}

fn parse_exec_command_input_preview(preview: &str) -> (Option<String>, Option<String>) {
    let trimmed = preview.trim();
    let candidate = trimmed
        .find('{')
        .map(|start| &trimmed[start..])
        .unwrap_or(trimmed);

    if let Ok(json) = serde_json::from_str::<JsonValue>(candidate) {
        let command = json
            .get("command")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned);
        let cwd = json
            .get("cwd")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned);

        return (command, cwd);
    }

    (
        extract_json_string_field(candidate, "command"),
        extract_json_string_field(candidate, "cwd"),
    )
}

fn tidy_shell_command(command: &str) -> String {
    let command = command.trim();

    let Some(rest) = command.split_once(" -lc ").map(|(_, rest)| rest) else {
        return command.to_string();
    };

    rest.trim().trim_matches(&['\'', '"'][..]).to_string()
}

fn split_exec_command_exit_code(preview: &str) -> (Option<i32>, &str) {
    let trimmed = preview.trim_start();

    let Some(rest) = trimmed
        .strip_prefix("exit_code:")
        .or_else(|| trimmed.strip_prefix("exit_code="))
    else {
        return (None, trimmed);
    };

    let rest = rest.trim_start();
    let end = rest
        .find(|ch: char| !ch.is_ascii_digit() && ch != '-')
        .unwrap_or(rest.len());
    let (code_str, remainder) = rest.split_at(end);
    let code = code_str.trim().parse::<i32>().ok();
    (code, remainder.trim_start())
}

fn start_client(conn: ClientInProcEndpoint, cx: &mut Context<SessionView>) -> (Client, Task<()>) {
    let (client, client_task) = Client::connect(conn, 64);
    let task = cx.spawn(
        move |_: WeakEntity<SessionView>, _cx: &mut AsyncApp| async move {
            client_task.run().await;
        },
    );
    (client, task)
}

async fn get_session_events(
    client: &Client,
    session_id: SessionId,
    before: Option<SessionEventCursor>,
    limit: u32,
) -> Result<GetSessionEventsResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::GetSessionEvents(GetSessionEventsRequest {
            session_id,
            before,
            limit,
            kinds: Vec::new(),
        }))
        .await?;

    match response {
        ResponseResult::GetSessionEvents(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

async fn send_session_message(
    client: &Client,
    session_id: SessionId,
    message: String,
    on_conflict: AgentMessageConflictAction,
) -> Result<SendSessionMessageResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::SendSessionMessage(
            SendSessionMessageRequest {
                session_id,
                message,
                on_conflict,
            },
        ))
        .await?;

    match response {
        ResponseResult::SendSessionMessage(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

fn sync_timeline_list_state(
    list_state: &ListState,
    old_items: &[SessionTimelineItem],
    new_items: &[SessionTimelineItem],
) {
    fn item_matches(
        old_items: &[SessionTimelineItem],
        new_items: &[SessionTimelineItem],
        old_ix: usize,
        new_ix: usize,
    ) -> bool {
        match (old_items.get(old_ix), new_items.get(new_ix)) {
            (Some(old), Some(new)) => old == new,
            (None, None) => true, // Follow marker.
            _ => false,
        }
    }

    let old_len = old_items.len() + 1;
    let new_len = new_items.len() + 1;

    if old_items == new_items {
        if list_state.item_count() != new_len {
            list_state.reset(new_len);
        }
        return;
    }

    let mut prefix = 0;
    while prefix < old_len && prefix < new_len && item_matches(old_items, new_items, prefix, prefix)
    {
        prefix += 1;
    }

    let mut suffix = 0;
    while suffix < old_len.saturating_sub(prefix)
        && suffix < new_len.saturating_sub(prefix)
        && item_matches(
            old_items,
            new_items,
            old_len - 1 - suffix,
            new_len - 1 - suffix,
        )
    {
        suffix += 1;
    }

    let old_range = prefix..old_len.saturating_sub(suffix);
    let new_count = new_len.saturating_sub(prefix + suffix);
    list_state.splice(old_range, new_count);
}

#[derive(Debug, Default, Clone, Copy)]
struct MarkdownCacheStats {
    message_events: usize,
    cached: usize,
    truncated: usize,
    total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolEventGroupKind {
    ExecCommands,
    ToolActivity,
}

impl ToolEventGroupKind {
    fn title(self) -> &'static str {
        match self {
            Self::ExecCommands => "Commands",
            Self::ToolActivity => "Tool activity",
        }
    }
}

#[derive(Debug, Clone)]
struct ToolEventGroup {
    start_ix: usize,
    end_ix: usize,
    count: usize,
    kind: ToolEventGroupKind,
    last_summary: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct ToolEventGroupMembership {
    group_id: SessionEventId,
    is_first: bool,
}

#[derive(Debug, Clone, Copy)]
struct ToolGroupTransition {
    started_at: Instant,
    from: f32,
    to: f32,
    duration: Duration,
}

impl ToolGroupTransition {
    fn value(&self) -> f32 {
        if self.duration == Duration::from_millis(0) {
            return self.to;
        }

        let elapsed = self.started_at.elapsed().as_secs_f32();
        let total = self.duration.as_secs_f32();
        let t = (elapsed / total).clamp(0.0, 1.0);
        let eased = ease_out_cubic(t);
        self.from + (self.to - self.from) * eased
    }

    fn is_done(&self) -> bool {
        self.duration == Duration::from_millis(0) || self.started_at.elapsed() >= self.duration
    }
}

fn cache_markdown_for_events(
    cache: &mut HashMap<SessionEventId, Arc<MarkdownDoc>>,
    events: &[SessionEvent],
) -> MarkdownCacheStats {
    let mut stats = MarkdownCacheStats::default();
    let options = MarkdownParseOptions::default();

    for event in events {
        let text = match &event.kind {
            SessionEventKind::UserMessage(msg) => msg.text.as_str(),
            SessionEventKind::AssistantMessage(msg) => msg.text.as_str(),
            _ => continue,
        };

        stats.message_events = stats.message_events.saturating_add(1);
        stats.total_bytes = stats.total_bytes.saturating_add(text.len());

        if cache.contains_key(&event.session_event_id) {
            continue;
        }

        let span = redesmyn_logging::redesmyn_info_span!(
            "parse_markdown",
            session_event_id = %event.session_event_id,
            bytes = text.len(),
            duration_ms = redesmyn_logging::tracing::field::Empty,
        );
        let start = Instant::now();
        let doc = {
            let _guard = span.enter();
            parse_markdown(text, options)
        };
        span.record(
            "duration_ms",
            redesmyn_logging::tracing::field::display(start.elapsed().as_millis()),
        );
        if doc.truncation.is_some() {
            stats.truncated = stats.truncated.saturating_add(1);
        }

        cache.insert(event.session_event_id, Arc::new(doc));
        stats.cached = stats.cached.saturating_add(1);
    }

    stats
}

pub struct SessionView {
    focus_handle: FocusHandle,
    timeline_list_state: ListState,
    timeline_items: Rc<Vec<SessionTimelineItem>>,
    timeline_follow_bottom: Rc<Cell<bool>>,
    timeline_last_scroll_offset: gpui::Pixels,
    timeline_scroll_handler_installed: bool,
    timeline_viewport_width: Option<gpui::Pixels>,
    timeline_list_reset_scheduled: bool,
    timeline_autoload_scheduled: bool,
    reasoning_scroll_states: Rc<RefCell<HashMap<String, ReasoningScrollState>>>,
    reasoning_shimmer_phase: u8,
    reasoning_shimmer_task: Option<Task<()>>,
    show_debug_controls: bool,
    session_id_input: Entity<TextInput>,
    composer_input: Entity<TextArea>,
    pending_focus_composer: bool,
    feed: Option<SessionFeedState>,
    collapsed_reasoning: HashSet<String>,
    expanded_tool_events: HashSet<SessionEventId>,
    expanded_tool_groups: HashSet<SessionEventId>,
    tool_group_transitions: Rc<HashMap<SessionEventId, ToolGroupTransition>>,
    tool_group_transition_guards: HashMap<SessionEventId, UiActivityGuard>,
    exec_command_result_by_invocation: Rc<HashMap<SessionEventId, (SessionEventId, ToolResult)>>,
    grouped_exec_command_result_event_ids: Rc<HashSet<SessionEventId>>,
    tool_event_groups: Rc<HashMap<SessionEventId, ToolEventGroup>>,
    tool_event_group_membership: Rc<HashMap<SessionEventId, ToolEventGroupMembership>>,
    markdown_cache: Rc<RefCell<HashMap<SessionEventId, Arc<MarkdownDoc>>>>,
    client: Option<Client>,
    _client_task: Option<Task<()>>,
    subscription_task: Option<Task<()>>,
    subscription_id: Option<SubscriptionId>,
    load_task: Option<Task<()>>,
    load_older_task: Option<Task<()>>,
    send_task: Option<Task<()>>,
    error: Option<SharedString>,
    _subscriptions: Vec<Subscription>,
}

impl Focusable for SessionView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl SessionView {
    pub fn new(
        control_plane_client: Option<ClientInProcEndpoint>,
        initial_session_id: Option<SessionId>,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        let timeline_list_state = ListState::new(1, gpui::ListAlignment::Top, px(400.0));
        let show_debug_controls = std::env::var("REDESMYN_SESSION_VIEWER_DEBUG_CONTROLS")
            .ok()
            .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true"));
        let session_id_input = cx.new(|cx| TextInput::new(cx).placeholder("Session id…"));
        let composer_input = cx.new(|cx| TextArea::new(cx).placeholder("Message…"));

        let mut subscriptions = Vec::new();
        subscriptions.push(cx.subscribe(&session_id_input, |this, _, event, cx| {
            if let TextInputEvent::Submitted(text) = event {
                this.load_session_id(text.as_ref(), cx);
            }
        }));
        subscriptions.push(
            cx.subscribe(&composer_input, |this, _, event, cx| match event {
                TextInputEvent::Changed(text) => {
                    if let Some(feed) = this.feed.as_mut() {
                        feed.set_draft(text.as_ref());
                        cx.notify();
                    }
                }
                TextInputEvent::Submitted(_text) => {
                    this.send_message(AgentMessageConflictAction::Fail, cx);
                }
            }),
        );

        let (client, client_task) = match control_plane_client {
            Some(conn) => {
                let (client, task) = start_client(conn, cx);
                (Some(client), Some(task))
            }
            None => (None, None),
        };

        let initial_id = initial_session_id.map(|id| id.to_string()).or_else(|| {
            std::env::var("REDESMYN_SESSION_VIEWER_SESSION_ID")
                .ok()
                .filter(|_| show_debug_controls)
        });
        if let Some(id) = initial_id {
            session_id_input.update(cx, move |input, cx| input.set_text(id, cx));
        }

        let mut this = Self {
            focus_handle,
            timeline_list_state,
            timeline_items: Rc::new(Vec::new()),
            timeline_follow_bottom: Rc::new(Cell::new(false)),
            timeline_last_scroll_offset: px(0.0),
            timeline_scroll_handler_installed: false,
            timeline_viewport_width: None,
            timeline_list_reset_scheduled: false,
            timeline_autoload_scheduled: false,
            reasoning_scroll_states: Rc::new(RefCell::new(HashMap::new())),
            reasoning_shimmer_phase: 0,
            reasoning_shimmer_task: None,
            show_debug_controls,
            session_id_input,
            composer_input,
            pending_focus_composer: false,
            feed: None,
            collapsed_reasoning: HashSet::new(),
            expanded_tool_events: HashSet::new(),
            expanded_tool_groups: HashSet::new(),
            tool_group_transitions: Rc::new(HashMap::new()),
            tool_group_transition_guards: HashMap::new(),
            exec_command_result_by_invocation: Rc::new(HashMap::new()),
            grouped_exec_command_result_event_ids: Rc::new(HashSet::new()),
            tool_event_groups: Rc::new(HashMap::new()),
            tool_event_group_membership: Rc::new(HashMap::new()),
            markdown_cache: Rc::new(RefCell::new(HashMap::new())),
            client,
            _client_task: client_task,
            subscription_task: None,
            subscription_id: None,
            load_task: None,
            load_older_task: None,
            send_task: None,
            error: None,
            _subscriptions: subscriptions,
        };

        let initial = this.session_id_input.read(cx).text().clone();
        if !initial.as_ref().trim().is_empty() {
            this.load_session_id(initial.as_ref(), cx);
        }

        this
    }

    pub fn set_session_id(&mut self, session_id: Option<SessionId>, cx: &mut Context<Self>) {
        let Some(session_id) = session_id else {
            self.subscription_id = None;
            self.subscription_task = None;
            self.load_task = None;
            self.load_older_task = None;
            self.error = None;
            self.feed = None;
            self.timeline_follow_bottom.set(false);
            self.timeline_last_scroll_offset = px(0.0);
            self.collapsed_reasoning.clear();
            self.expanded_tool_events.clear();
            self.expanded_tool_groups.clear();
            self.tool_group_transitions = Rc::new(HashMap::new());
            self.tool_group_transition_guards.clear();
            self.exec_command_result_by_invocation = Rc::new(HashMap::new());
            self.grouped_exec_command_result_event_ids = Rc::new(HashSet::new());
            self.tool_event_groups = Rc::new(HashMap::new());
            self.tool_event_group_membership = Rc::new(HashMap::new());
            self.markdown_cache.borrow_mut().clear();
            self.set_timeline_items(Vec::new());
            self.timeline_list_reset_scheduled = false;
            self.timeline_autoload_scheduled = false;
            self.reasoning_shimmer_phase = 0;
            self.reasoning_shimmer_task = None;
            self.pending_focus_composer = false;
            self.composer_input
                .update(cx, |input, cx| input.set_text("", cx));
            self.session_id_input
                .update(cx, |input, cx| input.set_text("", cx));
            cx.notify();
            return;
        };

        self.session_id_input
            .update(cx, |input, cx| input.set_text(session_id.to_string(), cx));
        self.load_session_id(&session_id.to_string(), cx);
    }

    pub fn request_focus_composer(&mut self, cx: &mut Context<Self>) {
        self.pending_focus_composer = true;
        cx.notify();
    }

    #[must_use]
    pub fn ui_composer_state(&self) -> UiComposerState {
        let Some(feed) = self.feed.as_ref() else {
            return UiComposerState::default();
        };

        UiComposerState {
            sending: feed.composer.sending,
            error: feed.composer.last_error.clone(),
        }
    }

    fn load_session_id(&mut self, raw: &str, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else {
            self.error = Some("Control plane client is unavailable.".into());
            cx.notify();
            return;
        };

        if let Some(subscription_id) = self.subscription_id.take() {
            let client = client.clone();
            cx.spawn(move |_: WeakEntity<Self>, _cx: &mut AsyncApp| async move {
                let _ = client.unsubscribe(subscription_id).await;
            })
            .detach();
        }

        let raw = raw.trim();
        let session_id = match SessionId::from_str(raw) {
            Ok(id) => id,
            Err(_) => {
                self.error = Some("Invalid session id.".into());
                cx.notify();
                return;
            }
        };

        self.error = None;
        self.subscription_task = None;
        self.subscription_id = None;
        self.load_task = None;
        self.load_older_task = None;
        self.send_task = None;
        self.collapsed_reasoning.clear();
        self.expanded_tool_events.clear();
        self.expanded_tool_groups.clear();
        self.tool_group_transitions = Rc::new(HashMap::new());
        self.tool_group_transition_guards.clear();
        self.exec_command_result_by_invocation = Rc::new(HashMap::new());
        self.grouped_exec_command_result_event_ids = Rc::new(HashSet::new());
        self.tool_event_groups = Rc::new(HashMap::new());
        self.tool_event_group_membership = Rc::new(HashMap::new());
        self.markdown_cache.borrow_mut().clear();
        self.feed = Some(SessionFeedState::new(session_id));
        self.composer_input
            .update(cx, |input, cx| input.set_text("", cx));
        self.set_timeline_items(Vec::new());
        cx.notify();

        let view = cx.entity();
        self.load_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let client = client.clone();
            let cx = cx.clone();
            async move {
                let response = get_session_events(&client, session_id, None, 50).await;
                let _ = cx
                    .update(|cx| view.update(cx, |this, cx| this.on_history_loaded(response, cx)));
            }
        }));
    }

    fn send_message(&mut self, on_conflict: AgentMessageConflictAction, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else {
            if let Some(feed) = self.feed.as_mut() {
                feed.finish_sending_error("Control plane client is unavailable.");
            } else {
                self.error = Some("Control plane client is unavailable.".into());
            }
            cx.notify();
            return;
        };

        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        if feed.composer.sending {
            return;
        }

        let draft = self.composer_input.read(cx).text().clone();
        let message = draft.as_ref().trim().to_string();
        if message.is_empty() {
            return;
        }

        let session_id = feed.session_id;
        feed.start_sending();
        feed.set_at_bottom(true);
        self.timeline_follow_bottom.set(true);
        self.refresh_timeline_items();
        cx.notify();

        let view = cx.entity();
        self.send_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let result = send_session_message(&client, session_id, message, on_conflict).await;
                let _ =
                    cx.update(|cx| view.update(cx, |this, cx| this.on_send_completed(result, cx)));
            }
        }));
    }

    fn on_send_completed(
        &mut self,
        result: Result<SendSessionMessageResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        self.send_task = None;

        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        match result {
            Ok(resp) => {
                let previous = feed.session_id;
                feed.finish_sending_success();
                self.composer_input
                    .update(cx, |input, cx| input.set_text("", cx));

                if resp.session_id != previous {
                    let raw = resp.session_id.to_string();
                    self.session_id_input
                        .update(cx, |input, cx| input.set_text(raw.clone(), cx));
                    self.load_session_id(raw.as_ref(), cx);
                    return;
                }

                let stats = cache_markdown_for_events(
                    &mut *self.markdown_cache.borrow_mut(),
                    std::slice::from_ref(&resp.event),
                );
                if stats.truncated > 0 {
                    redesmyn_logging::tracing::warn!(
                        session_event_id = %resp.event.session_event_id,
                        total_bytes = stats.total_bytes,
                        "markdown input truncated for rendering"
                    );
                }

                feed.apply_live_event(resp.event);
                self.refresh_timeline_items();
            }
            Err(err) => {
                if err.category == ErrorCategory::Conflict {
                    let code = err
                        .detail
                        .as_ref()
                        .and_then(|detail| detail.get(CONFLICT_CODE_KEY))
                        .cloned();
                    if let Some(code) = code {
                        feed.finish_sending_conflict(code, err.message);
                    } else {
                        feed.finish_sending_error(err.message);
                    }
                } else {
                    feed.finish_sending_error(err.message);
                }
            }
        }

        cx.notify();
    }

    fn on_history_loaded(
        &mut self,
        response: Result<GetSessionEventsResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        match response {
            Ok(resp) => {
                let after = resp.events.last().map(|event| SessionEventCursor {
                    created_at: event.created_at,
                    session_event_id: event.session_event_id,
                });

                let stats =
                    cache_markdown_for_events(&mut *self.markdown_cache.borrow_mut(), &resp.events);
                if stats.cached > 0 {
                    let span = redesmyn_logging::redesmyn_info_span!(
                        "session_markdown_cache_history",
                        session_id = %feed.session_id,
                        events = resp.events.len(),
                        message_events = stats.message_events,
                        cached = stats.cached,
                        truncated = stats.truncated,
                        total_bytes = stats.total_bytes
                    );
                    let _guard = span.enter();
                    redesmyn_logging::tracing::info!("cached markdown docs");
                }

                feed.apply_history_page(resp.events, resp.next_cursor);
                feed.set_at_bottom(true);
                feed.clear_scroll_intents();
                self.refresh_timeline_items();
                self.timeline_follow_bottom.set(true);
                self.start_subscription(after, cx);
            }
            Err(err) => {
                self.error = Some(err.message.into());
            }
        }

        cx.notify();
    }

    fn start_subscription(&mut self, after: Option<SessionEventCursor>, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else {
            return;
        };
        let Some(feed) = self.feed.as_ref() else {
            return;
        };

        let session_id = feed.session_id;
        let view = cx.entity();

        self.subscription_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let subscribed = client.subscribe_session_events(session_id, after).await;
                let (subscription_id, mut events_rx) = match subscribed {
                    Ok((id, rx)) => (id, rx),
                    Err(err) => {
                        let _ = cx.update(|cx| {
                            view.update(cx, |this, cx| {
                                if let Some(feed) = this.feed.as_mut() {
                                    feed.apply_live_error(err.message);
                                }
                                cx.notify();
                            })
                        });
                        return;
                    }
                };

                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.subscription_id = Some(subscription_id);
                        if let Some(feed) = this.feed.as_mut() {
                            feed.set_live_syncing(true);
                        }
                        cx.notify();
                    })
                });

                while let Some(event) = events_rx.recv().await {
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            this.on_subscription_event(event, cx);
                        })
                    });
                }
            }
        }));
    }

    fn on_subscription_event(&mut self, event: SubscriptionEvent, cx: &mut Context<Self>) {
        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        match event {
            SubscriptionEvent::SessionEvent(ev) => {
                if ev.session_id != feed.session_id {
                    return;
                }
                let stats = cache_markdown_for_events(
                    &mut *self.markdown_cache.borrow_mut(),
                    std::slice::from_ref(&ev),
                );
                if stats.truncated > 0 {
                    redesmyn_logging::tracing::warn!(
                        session_event_id = %ev.session_event_id,
                        total_bytes = stats.total_bytes,
                        "markdown input truncated for rendering"
                    );
                }
                feed.apply_live_event(ev);
                self.refresh_timeline_items();
            }
            SubscriptionEvent::SessionLiveEvent(ev) => {
                if ev.session_id != feed.session_id {
                    return;
                }
                feed.apply_live_session_event(ev);
                self.refresh_timeline_items();
            }
            SubscriptionEvent::Error(err) => {
                feed.apply_live_error(err.message);
            }
            SubscriptionEvent::Subscribed(_) | SubscriptionEvent::EventLog(_) => {}
        }

        cx.notify();
    }

    fn maybe_autoload_older(&mut self, cx: &mut Context<Self>) {
        let Some(feed) = self.feed.as_ref() else {
            return;
        };

        if self.timeline_autoload_scheduled {
            return;
        }

        if feed.history.loading_older || feed.history.next_cursor.is_none() {
            return;
        }

        let scroll_top = self.timeline_list_state.logical_scroll_top();
        if scroll_top.item_ix != 0 || scroll_top.offset_in_item > px(24.0) {
            return;
        }

        self.timeline_autoload_scheduled = true;
        let view = cx.entity();
        cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.timeline_autoload_scheduled = false;
                        this.start_loading_older(cx);
                    })
                });
            }
        })
        .detach();
    }

    fn start_loading_older(&mut self, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else {
            return;
        };

        if self
            .feed
            .as_ref()
            .is_some_and(|feed| feed.history.loading_older)
        {
            return;
        }

        let Some(before) = self.feed.as_ref().and_then(|feed| feed.history.next_cursor) else {
            return;
        };

        if let Some(feed) = self.feed.as_mut() {
            feed.start_loading_older();
        }
        self.refresh_timeline_items();
        cx.notify();

        let Some(session_id) = self.feed.as_ref().map(|feed| feed.session_id) else {
            return;
        };
        let view = cx.entity();
        self.load_older_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let response = get_session_events(&client, session_id, Some(before), 50).await;
                let _ =
                    cx.update(|cx| view.update(cx, |this, cx| this.on_older_loaded(response, cx)));
            }
        }));
    }

    fn on_older_loaded(
        &mut self,
        response: Result<GetSessionEventsResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        match response {
            Ok(resp) => {
                let stats =
                    cache_markdown_for_events(&mut *self.markdown_cache.borrow_mut(), &resp.events);
                if stats.cached > 0 {
                    let span = redesmyn_logging::redesmyn_info_span!(
                        "session_markdown_cache_older",
                        session_id = %feed.session_id,
                        events = resp.events.len(),
                        message_events = stats.message_events,
                        cached = stats.cached,
                        truncated = stats.truncated,
                        total_bytes = stats.total_bytes
                    );
                    let _guard = span.enter();
                    redesmyn_logging::tracing::info!("cached markdown docs");
                }

                feed.apply_history_page(resp.events, resp.next_cursor);
                self.refresh_timeline_items();
            }
            Err(err) => {
                feed.apply_history_error(err.message);
            }
        }

        cx.notify();
    }

    fn schedule_list_reset(&mut self, scroll_top: ListOffset, cx: &mut Context<Self>) {
        if self.timeline_list_reset_scheduled {
            return;
        }

        self.timeline_list_reset_scheduled = true;
        let view = cx.entity();
        cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.timeline_list_reset_scheduled = false;
                        this.timeline_list_state.reset(this.timeline_items.len() + 1);
                        this.timeline_list_state.scroll_to(scroll_top);
                        cx.notify();
                    })
                });
            }
        })
        .detach();
    }

    fn on_timeline_scrolled(
        &mut self,
        _visible_range_end: usize,
        _count: usize,
        cx: &mut Context<Self>,
    ) {
        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        let max_offset = self.timeline_list_state.max_offset_for_scrollbar().height;
        let scroll_offset = -self.timeline_list_state.scroll_px_offset_for_scrollbar().y;
        let threshold = px(4.0);
        let follow_zone = px(96.0);

        let was_at_bottom = feed.scroll.at_bottom;
        let distance_to_bottom = (max_offset - scroll_offset).max(px(0.0));
        let at_bottom = distance_to_bottom <= threshold;
        feed.set_at_bottom(at_bottom);

        let delta = scroll_offset - self.timeline_last_scroll_offset;
        self.timeline_last_scroll_offset = scroll_offset;

        if delta < px(-0.5) && distance_to_bottom > threshold {
            self.timeline_follow_bottom.set(false);
        } else if distance_to_bottom <= follow_zone {
            self.timeline_follow_bottom.set(true);
        }

        if feed.scroll.at_bottom != was_at_bottom {
            self.refresh_timeline_items();
        }

        cx.notify();
    }

    fn toggle_tool_event(&mut self, session_event_id: SessionEventId, cx: &mut Context<Self>) {
        if self.expanded_tool_events.contains(&session_event_id) {
            self.expanded_tool_events.remove(&session_event_id);
        } else {
            self.expanded_tool_events.insert(session_event_id);
        }
        self.invalidate_timeline_item(session_event_id);
        cx.notify();
    }

    fn toggle_reasoning(&mut self, key: &str, cx: &mut Context<Self>) {
        if self.collapsed_reasoning.contains(key) {
            self.collapsed_reasoning.remove(key);
        } else {
            self.collapsed_reasoning.insert(key.to_owned());
        }
        self.invalidate_reasoning_item(key);
        cx.notify();
    }

    fn toggle_tool_group(&mut self, group_id: SessionEventId, cx: &mut Context<Self>) {
        let was_expanded = self.expanded_tool_groups.contains(&group_id);
        if was_expanded {
            self.expanded_tool_groups.remove(&group_id);
        } else {
            self.expanded_tool_groups.insert(group_id);
        }

        let target = if was_expanded { 0.0 } else { 1.0 };
        let current = self
            .tool_group_transitions
            .get(&group_id)
            .map(|transition| transition.value())
            .unwrap_or_else(|| if was_expanded { 1.0 } else { 0.0 });

        let duration = ui_test_mode_animation_duration(Duration::from_millis(180));
        let mut transitions: HashMap<SessionEventId, ToolGroupTransition> =
            self.tool_group_transitions.as_ref().clone();

        if duration == Duration::from_millis(0) || (current - target).abs() < 1e-3 {
            transitions.remove(&group_id);
            self.tool_group_transition_guards.remove(&group_id);
        } else {
            transitions.insert(
                group_id,
                ToolGroupTransition {
                    started_at: Instant::now(),
                    from: current,
                    to: target,
                    duration,
                },
            );

            if !self.tool_group_transition_guards.contains_key(&group_id)
                && let Some(tracker) = ui_idle_tracker(cx)
            {
                self.tool_group_transition_guards
                    .insert(group_id, tracker.begin_transition());
            }
        }

        self.tool_group_transitions = Rc::new(transitions);
        self.invalidate_tool_group(group_id);
        cx.notify();
    }

    fn jump_to_bottom(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        feed.set_at_bottom(true);
        feed.clear_scroll_intents();
        self.timeline_follow_bottom.set(true);
        self.refresh_timeline_items();
        cx.notify();
    }

    fn set_timeline_items(&mut self, items: Vec<SessionTimelineItem>) {
        let old_items = std::mem::replace(&mut self.timeline_items, Rc::new(items));
        self.rebuild_exec_command_groups();
        self.rebuild_tool_event_groups();
        sync_timeline_list_state(
            &self.timeline_list_state,
            old_items.as_ref(),
            self.timeline_items.as_ref(),
        );
    }

    fn rebuild_exec_command_groups(&mut self) {
        let mut result_by_invocation = HashMap::new();
        let mut grouped_result_ids = HashSet::new();
        let mut invocations_by_call_id: HashMap<String, SessionEventId> = HashMap::new();

        for item in self.timeline_items.iter() {
            let SessionTimelineItem::Event(event) = item else {
                continue;
            };

            match &event.content {
                SessionEventItemContent::ToolInvocation(invocation)
                    if invocation.tool_name == "exec_command" =>
                {
                    if let Some(call_id) = invocation.tool_call_id.as_ref() {
                        invocations_by_call_id.insert(call_id.clone(), event.session_event_id);
                    }
                }
                SessionEventItemContent::ToolResult(result)
                    if result.tool_name == "exec_command" =>
                {
                    if let Some(call_id) = result.tool_call_id.as_ref() {
                        if let Some(invocation_id) = invocations_by_call_id.get(call_id) {
                            result_by_invocation.insert(
                                invocation_id.clone(),
                                (event.session_event_id, result.clone()),
                            );
                            grouped_result_ids.insert(event.session_event_id);
                        }
                    }
                }
                _ => {}
            }
        }

        self.exec_command_result_by_invocation = Rc::new(result_by_invocation);
        self.grouped_exec_command_result_event_ids = Rc::new(grouped_result_ids);
    }

    fn rebuild_tool_event_groups(&mut self) {
        const MIN_GROUP_LEN: usize = 4;

        let mut groups: HashMap<SessionEventId, ToolEventGroup> = HashMap::new();
        let mut membership: HashMap<SessionEventId, ToolEventGroupMembership> = HashMap::new();

        let mut run_start_ix: Option<usize> = None;
        let mut run_end_ix: Option<usize> = None;
        let mut run_event_ids: Vec<SessionEventId> = Vec::new();
        let mut run_kind = ToolEventGroupKind::ExecCommands;
        let mut last_summary: Option<String> = None;

        let mut flush_run = |run_start_ix: &mut Option<usize>,
                             run_end_ix: &mut Option<usize>,
                             run_event_ids: &mut Vec<SessionEventId>,
                             run_kind: &mut ToolEventGroupKind,
                             last_summary: &mut Option<String>| {
            let Some(start_ix) = run_start_ix.take() else {
                return;
            };
            let Some(end_ix) = run_end_ix.take() else {
                return;
            };

            if run_event_ids.len() >= MIN_GROUP_LEN {
                let group_id = run_event_ids[0];

                groups.insert(
                    group_id,
                    ToolEventGroup {
                        start_ix,
                        end_ix,
                        count: run_event_ids.len(),
                        kind: *run_kind,
                        last_summary: match *run_kind {
                            ToolEventGroupKind::ExecCommands => None,
                            ToolEventGroupKind::ToolActivity => last_summary.clone(),
                        },
                    },
                );

                for (ix, event_id) in run_event_ids.iter().copied().enumerate() {
                    membership.insert(
                        event_id,
                        ToolEventGroupMembership {
                            group_id,
                            is_first: ix == 0,
                        },
                    );
                }
            }

            run_event_ids.clear();
            *run_kind = ToolEventGroupKind::ExecCommands;
            *last_summary = None;
        };

        for (ix, item) in self.timeline_items.iter().enumerate() {
            let SessionTimelineItem::Event(event) = item else {
                flush_run(
                    &mut run_start_ix,
                    &mut run_end_ix,
                    &mut run_event_ids,
                    &mut run_kind,
                    &mut last_summary,
                );
                continue;
            };

            match &event.content {
                SessionEventItemContent::ToolInvocation(tool) => {
                    let is_exec_command = tool.tool_name == "exec_command";
                    let summary_main = if is_exec_command {
                        let (command, _cwd) = parse_exec_command_input_preview(&tool.input_preview);
                        command
                            .as_deref()
                            .map(tidy_shell_command)
                            .unwrap_or_else(|| tool.input_preview.clone())
                    } else {
                        tool.input_preview.clone()
                    };

                    if run_start_ix.is_none() {
                        run_start_ix = Some(ix);
                    }
                    run_end_ix = Some(ix + 1);
                    run_event_ids.push(event.session_event_id);
                    last_summary = Some(summary_main);

                    if !is_exec_command {
                        run_kind = ToolEventGroupKind::ToolActivity;
                    }
                }
                SessionEventItemContent::ToolResult(_tool)
                    if self
                        .grouped_exec_command_result_event_ids
                        .contains(&event.session_event_id) =>
                {
                    // `exec_command` results are already surfaced via their invocation rows; keep
                    // groups contiguous across the invocation/result pairs.
                    if run_start_ix.is_none() {
                        run_start_ix = Some(ix);
                    }
                    run_end_ix = Some(ix + 1);
                }
                SessionEventItemContent::ToolResult(tool) => {
                    if run_start_ix.is_none() {
                        run_start_ix = Some(ix);
                    }
                    run_end_ix = Some(ix + 1);
                    run_event_ids.push(event.session_event_id);
                    last_summary = Some(tool.output_preview.clone());

                    if tool.tool_name != "exec_command" {
                        run_kind = ToolEventGroupKind::ToolActivity;
                    }
                }
                _ => {
                    flush_run(
                        &mut run_start_ix,
                        &mut run_end_ix,
                        &mut run_event_ids,
                        &mut run_kind,
                        &mut last_summary,
                    );
                }
            }
        }

        flush_run(
            &mut run_start_ix,
            &mut run_end_ix,
            &mut run_event_ids,
            &mut run_kind,
            &mut last_summary,
        );

        self.tool_event_groups = Rc::new(groups);
        self.tool_event_group_membership = Rc::new(membership);
    }

    fn refresh_timeline_items(&mut self) {
        let items = self
            .feed
            .as_ref()
            .map(SessionFeedState::timeline_items)
            .unwrap_or_default();
        self.set_timeline_items(items);
    }

    fn ensure_reasoning_shimmer_task(&mut self, cx: &mut Context<Self>) {
        if self.reasoning_shimmer_task.is_some() {
            return;
        }

        let interval = ui_test_mode_animation_duration(Duration::from_millis(90));
        if interval == Duration::from_millis(0) {
            return;
        }

        self.reasoning_shimmer_task =
            Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    loop {
                        gpui::Timer::after(interval).await;
                        let Some(view) = weak.upgrade() else {
                            break;
                        };
                        match cx.update(|cx| {
                            view.update(cx, |this, cx| {
                                let has_ephemeral_reasoning =
                                    this.timeline_items.iter().any(|item| {
                                        matches!(item, SessionTimelineItem::EphemeralReasoning(_))
                                    });

                                if !has_ephemeral_reasoning {
                                    this.reasoning_shimmer_phase = 0;
                                    this.reasoning_shimmer_task = None;
                                    cx.notify();
                                    return false;
                                }

                                this.reasoning_shimmer_phase =
                                    this.reasoning_shimmer_phase.wrapping_add(1);
                                cx.notify();
                                true
                            })
                        }) {
                            Ok(true) => {}
                            Ok(false) => break,
                            Err(_) => break,
                        }
                    }
                }
            }));
    }

    fn reasoning_shimmer_alpha(&self) -> f32 {
        let t = (self.reasoning_shimmer_phase as f32) * 0.18;
        let pulse = (t.sin() * 0.5) + 0.5;
        (0.35 + 0.65 * pulse).clamp(0.0, 1.0)
    }

    fn invalidate_timeline_item(&mut self, session_event_id: SessionEventId) {
        if let Some(ix) = self.timeline_items.iter().position(|item| {
            matches!(
                item,
                SessionTimelineItem::Event(ev) if ev.session_event_id == session_event_id
            )
        }) {
            self.timeline_list_state.splice(ix..ix + 1, 1);
        }
    }

    fn invalidate_reasoning_item(&mut self, key: &str) {
        let Some(ix) = self.timeline_items.iter().position(|item| match item {
            SessionTimelineItem::Event(ev) => match &ev.content {
                SessionEventItemContent::AssistantReasoning(reasoning) => {
                    if let Some(item_id) = reasoning.item_id.as_deref() {
                        item_id == key
                    } else {
                        ev.session_event_id.to_string() == key
                    }
                }
                _ => false,
            },
            SessionTimelineItem::EphemeralReasoning(item) => item.key == key,
            _ => false,
        }) else {
            return;
        };

        self.timeline_list_state.splice(ix..ix + 1, 1);
    }

    fn invalidate_tool_group(&mut self, group_id: SessionEventId) {
        let Some(group) = self.tool_event_groups.get(&group_id) else {
            return;
        };
        let count = group.end_ix.saturating_sub(group.start_ix);
        if count == 0 {
            return;
        }
        self.timeline_list_state
            .splice(group.start_ix..group.end_ix, count);
    }

    fn tick_tool_group_transitions_for_render(&mut self, window: &mut Window) {
        if self.tool_group_transitions.is_empty() {
            return;
        }

        let transitions = Rc::clone(&self.tool_group_transitions);
        // While a group is animating we need to invalidate its whole range every frame so GPUI's
        // list re-measures the evolving row heights. Otherwise the list will keep the initial
        // cached heights and the animation will "jump" once at the end.
        let active_group_ids: Vec<SessionEventId> = transitions.keys().copied().collect();
        let mut next: HashMap<SessionEventId, ToolGroupTransition> = HashMap::new();

        for (group_id, transition) in transitions.iter() {
            if transition.is_done() {
                self.tool_group_transition_guards.remove(group_id);
            } else {
                next.insert(*group_id, *transition);
            }
        }

        if next.len() != self.tool_group_transitions.len() {
            self.tool_group_transitions = Rc::new(next);
        } else {
            // Keep the same Rc when nothing changes so list closures don't churn.
        }

        for group_id in active_group_ids {
            self.invalidate_tool_group(group_id);
        }

        if !self.tool_group_transitions.is_empty() {
            window.request_animation_frame();
        }
    }
}

impl Render for SessionView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let theme = theme_for_window(window, cx);
        let view = cx.entity();

        if !self.timeline_scroll_handler_installed {
            let view = view.clone();
            let follow_bottom = Rc::clone(&self.timeline_follow_bottom);
            self.timeline_list_state
                .set_scroll_handler(move |event, _window, cx| {
                    // Disable follow-bottom immediately on user scroll so we don't "fight" the
                    // user between this handler and the async update that reads list offsets.
                    follow_bottom.set(false);
                    let visible_end = event.visible_range.end;
                    let count = event.count;
                    let view = view.clone();

                    cx.spawn(move |cx: &mut AsyncApp| {
                        let cx = cx.clone();
                        async move {
                            let _ = cx.update(|cx| {
                                view.update(cx, |this, cx| {
                                    this.on_timeline_scrolled(visible_end, count, cx);
                                })
                            });
                        }
                    })
                    .detach();
                });
            self.timeline_scroll_handler_installed = true;
        }

        let max_offset = self.timeline_list_state.max_offset_for_scrollbar().height;
        let scroll_offset = -self.timeline_list_state.scroll_px_offset_for_scrollbar().y;
        let distance_to_bottom = (max_offset - scroll_offset).max(px(0.0));
        let threshold = px(4.0);

        let assistant_generating = self.timeline_items.iter().any(|item| match item {
            SessionTimelineItem::EphemeralReasoning(_) => true,
            SessionTimelineItem::EphemeralText(text) => matches!(
                text.role,
                redesmyn_session_view_model::SessionMessageRole::Assistant
                    | redesmyn_session_view_model::SessionMessageRole::Tool
            ),
            _ => false,
        });

        if assistant_generating && self.timeline_follow_bottom.get() && distance_to_bottom > threshold
        {
            self.timeline_list_state
                .scroll_to_reveal_item(self.timeline_items.len());
        }

        if self.pending_focus_composer && self.feed.is_some() {
            self.pending_focus_composer = false;
            window.focus(&self.composer_input.focus_handle(cx));
        }

        let viewport_width = self.timeline_list_state.viewport_bounds().size.width;
        if viewport_width > px(0.0) {
            if self
                .timeline_viewport_width
                .is_some_and(|prev| prev != viewport_width)
            {
                let scroll_top = self.timeline_list_state.logical_scroll_top();
                self.schedule_list_reset(scroll_top, cx);
            }
            self.timeline_viewport_width = Some(viewport_width);
        }

        if self
            .timeline_items
            .iter()
            .any(|item| matches!(item, SessionTimelineItem::EphemeralReasoning(_)))
        {
            self.ensure_reasoning_shimmer_task(cx);
        }

        // Note: do not call `ListState` methods from within the scroll handler (it runs while the
        // list state is mutably borrowed). We instead evaluate autoloading here during render.
        self.maybe_autoload_older(cx);
        self.tick_tool_group_transitions_for_render(window);

        let mut content = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .w_full()
            .gap(theme.spacing.sm);

        if let Some(error) = self.error.clone() {
            content = content.child(
                Callout::new(error)
                    .kind(CalloutKind::Warning)
                    .title("Session viewer"),
            );
        }

        if self.client.is_none() {
            content = content.child(
                Callout::new("Control plane client is unavailable in this mode.")
                    .kind(CalloutKind::Info)
                    .title("Session viewer"),
            );
        }

        if self.show_debug_controls {
            let load_view = view.clone();
            let controls = div()
                .flex()
                .flex_row()
                .gap(theme.spacing.sm)
                .items_center()
                .w_full()
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(self.session_id_input.clone()),
                )
                .child(
                    TextButton::new(("session_load", cx.entity_id()), "Load").on_click(
                        move |_, _, cx| {
                            let id = load_view.read(cx).session_id_input.read(cx).text().clone();
                            load_view.update(cx, |this, cx| this.load_session_id(id.as_ref(), cx));
                        },
                    ),
                );

            content = content.child(controls);
        }

        let items = Rc::clone(&self.timeline_items);
        let markdown_cache = Rc::clone(&self.markdown_cache);
        let expanded_tool_events = self.expanded_tool_events.clone();
        let expanded_tool_groups = self.expanded_tool_groups.clone();
        let collapsed_reasoning = self.collapsed_reasoning.clone();
        let reasoning_shimmer_alpha = self.reasoning_shimmer_alpha();
        let exec_command_results = Rc::clone(&self.exec_command_result_by_invocation);
        let grouped_exec_command_result_ids =
            Rc::clone(&self.grouped_exec_command_result_event_ids);
        let tool_event_groups = Rc::clone(&self.tool_event_groups);
        let tool_event_group_membership = Rc::clone(&self.tool_event_group_membership);
        let tool_group_transitions = Rc::clone(&self.tool_group_transitions);
        let reasoning_scroll_states = Rc::clone(&self.reasoning_scroll_states);
        let follow_bottom = Rc::clone(&self.timeline_follow_bottom);
        let entity_id = cx.entity_id();
        let timeline_view = view.clone();

        let feed_list = list(self.timeline_list_state.clone(), move |ix, window, cx| {
            let theme = theme_for_window(window, cx);
            let list = div().w_full().min_w_0();

            if ix == items.len() {
                return list
                    .child(AutoscrollMarker::new(follow_bottom.get()))
                    .into_any_element();
            }

            let Some(item) = items.get(ix).cloned() else {
                return div().into_any_element();
            };

            let rendered = match item {
                SessionTimelineItem::LoadOlder(row) => {
                    let label = if row.in_flight {
                        "Loading older…"
                    } else if row.enabled {
                        "Scroll up to load older"
                    } else {
                        ""
                    };

                    list.child(
                        div()
                            .id(("session_item_load_older", ix))
                            .w_full()
                            .flex()
                            .justify_center()
                            .py(theme.spacing.xs)
                            .text_xs()
                            .text_color(theme.colors.foreground_muted)
                            .child(label),
                    )
                }
                SessionTimelineItem::NewMessages(row) => list.child(
                    div().id(("session_item_new_messages", ix)).w_full().child(
                        TextButton::new(
                            ("session_jump_bottom", entity_id),
                            format!("New messages ({}) — Jump to bottom", row.count),
                        )
                        .on_click({
                            let view = timeline_view.clone();
                            move |event, window, cx| {
                                view.update(cx, |this, cx| this.jump_to_bottom(event, window, cx))
                            }
                        }),
                    ),
                ),
                SessionTimelineItem::EphemeralText(item) => list.child(
                    div()
                        .id(("session_item_ephemeral", ix))
                        .w_full()
                        .min_w_0()
                        .px(theme.spacing.sm)
                        .py(theme.spacing.sm)
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .opacity(0.75)
                        .child(item.text),
                ),
                SessionTimelineItem::EphemeralReasoning(item) => {
                    let key = item.key.clone();
                    let is_expanded = !collapsed_reasoning.contains(&key);
                    let chevron = if is_expanded { "▾" } else { "▸" };
                    let summary_text = (!item.summary.is_empty()).then(|| item.summary.join("\n"));
                    let raw_text = (!item.raw.is_empty()).then(|| item.raw.join("\n"));

                    let toggle_view = timeline_view.clone();
                    let toggle_key = key.clone();
                    let mut container = div()
                        .id(("session_item_ephemeral_reasoning", ix))
                        .w_full()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(theme.spacing.sm)
                        .px(theme.spacing.sm)
                        .py(theme.spacing.sm)
                        .rounded_md();

                    container = container.child(
                        div()
                            .id(("session_item_ephemeral_reasoning_header", ix))
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(theme.spacing.sm)
                            .cursor_pointer()
                            .focusable()
                            .on_click(move |event, _window, cx| {
                                if event.standard_click() {
                                    toggle_view.update(cx, |this, cx| {
                                        this.toggle_reasoning(&toggle_key, cx);
                                    });
                                }
                            })
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .font(theme.typography.mono.font.clone())
                                    .text_size(theme.typography.caption.size)
                                    .text_color(theme.colors.foreground_muted)
                                    .child(chevron),
                            )
                            .child(
                                div()
                                    .text_size(theme.typography.caption.size)
                                    .text_color(
                                        theme
                                            .colors
                                            .foreground_muted
                                            .opacity(reasoning_shimmer_alpha),
                                    )
                                    .child("Thinking"),
                            ),
                    );

                    if is_expanded {
                        let (summary_scroll_handle, follow_summary_bottom) = {
                            let mut states = reasoning_scroll_states.borrow_mut();
                            let state = states
                                .entry(key.clone())
                                .or_insert_with(ReasoningScrollState::new);

                            let handle = state.summary.handle.clone();
                            let offset_y = handle.offset().y;

                            if state.summary.follow_bottom {
                                if offset_y > state.summary.last_offset_y {
                                    state.summary.follow_bottom = false;
                                }
                            } else if should_autoscroll_to_bottom(&handle) {
                                state.summary.follow_bottom = true;
                            }

                            state.summary.last_offset_y = offset_y;
                            (handle, state.summary.follow_bottom)
                        };

                        if follow_summary_bottom {
                            summary_scroll_handle.scroll_to_bottom();
                        }

                        container = container.child(
                            div()
                                .id(("session_item_ephemeral_reasoning_summary", ix))
                                .max_h(px(160.0))
                                .overflow_y_scroll()
                                .track_scroll(&summary_scroll_handle)
                                .occlude()
                                .on_scroll_wheel({
                                    let handle = summary_scroll_handle.clone();
                                    let timeline_view = timeline_view.clone();
                                    move |event, window, cx| {
                                        chain_scroll_wheel_to_timeline_list_if_needed(
                                            event,
                                            window,
                                            cx,
                                            &handle,
                                            timeline_view.clone(),
                                        );
                                    }
                                })
                                .child(if let Some(summary_text) = summary_text {
                                    div()
                                        .text_size(theme.typography.caption.size)
                                        .text_color(theme.colors.foreground)
                                        .child(summary_text)
                                        .into_any_element()
                                } else {
                                    let bar_color = theme
                                        .colors
                                        .foreground_muted
                                        .opacity(
                                            (0.12 + 0.12 * reasoning_shimmer_alpha).clamp(0.0, 1.0),
                                        );
                                    let line_height = px(10.0);

                                    div()
                                        .flex()
                                        .flex_col()
                                        .gap(theme.spacing.xs)
                                        .child(
                                            div()
                                                .h(line_height)
                                                .w(relative(0.92))
                                                .rounded_sm()
                                                .bg(bar_color),
                                        )
                                        .child(
                                            div()
                                                .h(line_height)
                                                .w(relative(0.78))
                                                .rounded_sm()
                                                .bg(bar_color),
                                        )
                                        .child(
                                            div()
                                                .h(line_height)
                                                .w(relative(0.64))
                                                .rounded_sm()
                                                .bg(bar_color),
                                        )
                                        .into_any_element()
                                }),
                        );

                        if let Some(raw_text) = raw_text {
                            let (raw_scroll_handle, follow_raw_bottom) = {
                                let mut states = reasoning_scroll_states.borrow_mut();
                                let state = states
                                    .entry(key.clone())
                                    .or_insert_with(ReasoningScrollState::new);

                                let handle = state.raw.handle.clone();
                                let offset_y = handle.offset().y;

                                if state.raw.follow_bottom {
                                    if offset_y > state.raw.last_offset_y {
                                        state.raw.follow_bottom = false;
                                    }
                                } else if should_autoscroll_to_bottom(&handle) {
                                    state.raw.follow_bottom = true;
                                }

                                state.raw.last_offset_y = offset_y;
                                (handle, state.raw.follow_bottom)
                            };

                            if follow_raw_bottom {
                                raw_scroll_handle.scroll_to_bottom();
                            }

                            container = container.child(
                                div()
                                    .h(px(1.0))
                                    .bg(theme.colors.border.opacity(0.2)),
                            );
                            container = container.child(
                                div()
                                    .id((
                                        "session_item_ephemeral_reasoning_raw",
                                        stable_str_key(key.as_str()),
                                    ))
                                    .max_h(px(200.0))
                                    .overflow_y_scroll()
                                    .track_scroll(&raw_scroll_handle)
                                    .occlude()
                                    .on_scroll_wheel({
                                        let handle = raw_scroll_handle.clone();
                                        let timeline_view = timeline_view.clone();
                                        move |event, window, cx| {
                                            chain_scroll_wheel_to_timeline_list_if_needed(
                                                event,
                                                window,
                                                cx,
                                                &handle,
                                                timeline_view.clone(),
                                            );
                                        }
                                    })
                                    .font(theme.typography.mono.font.clone())
                                    .text_size(theme.typography.caption.size)
                                    .text_color(theme.colors.foreground)
                                    .child(raw_text),
                            );
                        }
                    }

                    list.child(container)
                }
                SessionTimelineItem::Event(item) => {
                    let session_event_id = item.session_event_id;
                    let event_key = session_event_id_key(item.session_event_id);
                    let bubble_id: ElementId = ("session_event", event_key).into();
                    let timeline_item_gap_y = theme.spacing.xs * 2.0;

                    let render_tool_group_row = |chevron: &'static str,
                                                 group_id: SessionEventId,
                                                 title: &'static str,
                                                 count: usize,
                                                 last: Option<String>| {
                        let toggle_view = timeline_view.clone();
                        let is_expanded = chevron == "▾";
                        div()
                            .id(("session_tool_group", session_event_id_key(group_id)))
                            .w_full()
                            .min_w_0()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(theme.spacing.sm)
                            .px(theme.spacing.sm)
                            .py(timeline_item_gap_y)
                            .rounded_sm()
                            .when(is_expanded, |this| {
                                this.bg(theme.colors.surface_elevated.opacity(0.18))
                            })
                            .font(theme.typography.mono.font.clone())
                            .text_size(theme.typography.caption.size)
                            .cursor_pointer()
                            .focusable()
                            .on_click(move |event, _window, cx| {
                                if event.standard_click() {
                                    toggle_view.update(cx, |this, cx| {
                                        this.toggle_tool_group(group_id, cx);
                                    });
                                }
                            })
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .whitespace_nowrap()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(chevron),
                            )
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .whitespace_nowrap()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(title),
                            )
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .whitespace_nowrap()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(count.to_string()),
                            )
                            .when_some(last, |this, last| {
                                this.child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .text_color(theme.colors.foreground)
                                        .truncate()
                                        .child(last),
                                )
                            })
                    };

                    let tool_group_member_row_height = {
                        let caption_line_height = window
                            .text_style()
                            .line_height
                            .to_pixels(theme.typography.caption.size, window.rem_size())
                            .round();
                        caption_line_height + timeline_item_gap_y * 2.0
                    };

                    match item.content {
                        SessionEventItemContent::AssistantReasoning(reasoning) => {
                            let key = reasoning
                                .item_id
                                .clone()
                                .unwrap_or_else(|| session_event_id.to_string());
                            let is_expanded = !collapsed_reasoning.contains(&key);
                            let chevron = if is_expanded { "▾" } else { "▸" };

                            let toggle_view = timeline_view.clone();
                            let toggle_key = key.clone();
                            let mut container = div()
                                .id(bubble_id.clone())
                                .w_full()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(theme.spacing.sm)
                                .px(theme.spacing.sm)
                                .py(theme.spacing.sm)
                                .rounded_md();

                            container = container.child(
                                div()
                                    .id((bubble_id.clone(), "reasoning_header"))
                                    .flex()
                                    .flex_row()
                                    .items_center()
                                    .gap(theme.spacing.sm)
                                    .cursor_pointer()
                                    .focusable()
                                    .on_click(move |event, _window, cx| {
                                        if event.standard_click() {
                                            toggle_view.update(cx, |this, cx| {
                                                this.toggle_reasoning(&toggle_key, cx);
                                            });
                                        }
                                    })
                                    .child(
                                        div()
                                            .flex_shrink_0()
                                            .font(theme.typography.mono.font.clone())
                                            .text_size(theme.typography.caption.size)
                                            .text_color(theme.colors.foreground_muted)
                                            .child(chevron),
                                    )
                                    .child(
                                        div()
                                            .text_size(theme.typography.caption.size)
                                            .text_color(theme.colors.foreground_muted)
                                            .child("Thought"),
                                    ),
                            );

                            if is_expanded {
                                let cached = { markdown_cache.borrow().get(&session_event_id).cloned() };
                                let summary_doc = cached.unwrap_or_else(|| {
                                    let doc = Arc::new(parse_markdown(
                                        reasoning.summary.text.as_str(),
                                        MarkdownParseOptions::default(),
                                    ));
                                    markdown_cache
                                        .borrow_mut()
                                        .insert(session_event_id, doc.clone());
                                    doc
                                });

                                let show_truncation_notice =
                                    reasoning.summary.full_text_artifact.is_none();

                                let (summary_scroll_handle, follow_summary_bottom) = {
                                    let mut states = reasoning_scroll_states.borrow_mut();
                                    let state = states
                                        .entry(key.clone())
                                        .or_insert_with(ReasoningScrollState::new);

                                    let handle = state.summary.handle.clone();
                                    let offset_y = handle.offset().y;

                                    if state.summary.follow_bottom {
                                        if offset_y > state.summary.last_offset_y {
                                            state.summary.follow_bottom = false;
                                        }
                                    } else if should_autoscroll_to_bottom(&handle) {
                                        state.summary.follow_bottom = true;
                                    }

                                    state.summary.last_offset_y = offset_y;
                                    (handle, state.summary.follow_bottom)
                                };

                                if follow_summary_bottom {
                                    summary_scroll_handle.scroll_to_bottom();
                                }

                                container = container.child(
                                    div()
                                        .id((bubble_id.clone(), "reasoning_summary_scroll"))
                                        .max_h(px(180.0))
                                        .overflow_y_scroll()
                                        .track_scroll(&summary_scroll_handle)
                                        .occlude()
                                        .on_scroll_wheel({
                                            let handle = summary_scroll_handle.clone();
                                            let timeline_view = timeline_view.clone();
                                            move |event, window, cx| {
                                                chain_scroll_wheel_to_timeline_list_if_needed(
                                                    event,
                                                    window,
                                                    cx,
                                                    &handle,
                                                    timeline_view.clone(),
                                                );
                                            }
                                        })
                                        .child(
                                            MarkdownView::new(
                                                (bubble_id.clone(), "reasoning_summary"),
                                                summary_doc,
                                            )
                                            .show_truncation_notice(show_truncation_notice)
                                            .text_size(theme.typography.caption.size)
                                            .into_any_element(),
                                        ),
                                );

                                if let Some(raw) = reasoning.raw {
                                    let (raw_scroll_handle, follow_raw_bottom) = {
                                        let mut states = reasoning_scroll_states.borrow_mut();
                                        let state = states
                                            .entry(key.clone())
                                            .or_insert_with(ReasoningScrollState::new);

                                        let handle = state.raw.handle.clone();
                                        let offset_y = handle.offset().y;

                                        if state.raw.follow_bottom {
                                            if offset_y > state.raw.last_offset_y {
                                                state.raw.follow_bottom = false;
                                            }
                                        } else if should_autoscroll_to_bottom(&handle) {
                                            state.raw.follow_bottom = true;
                                        }

                                        state.raw.last_offset_y = offset_y;
                                        (handle, state.raw.follow_bottom)
                                    };

                                    if follow_raw_bottom {
                                        raw_scroll_handle.scroll_to_bottom();
                                    }

                                    container = container.child(
                                        div()
                                            .h(px(1.0))
                                            .bg(theme.colors.border.opacity(0.2)),
                                    );

                                    container = container.child(
                                        div()
                                            .id((bubble_id.clone(), "reasoning_raw_scroll"))
                                            .max_h(px(240.0))
                                            .overflow_y_scroll()
                                            .track_scroll(&raw_scroll_handle)
                                            .occlude()
                                            .on_scroll_wheel({
                                                let handle = raw_scroll_handle.clone();
                                                let timeline_view = timeline_view.clone();
                                                move |event, window, cx| {
                                                    chain_scroll_wheel_to_timeline_list_if_needed(
                                                        event,
                                                        window,
                                                        cx,
                                                        &handle,
                                                        timeline_view.clone(),
                                                    );
                                                }
                                            })
                                            .font(theme.typography.mono.font.clone())
                                            .text_size(theme.typography.caption.size)
                                            .text_color(theme.colors.foreground)
                                            .child(raw.text),
                                    );
                                }
                            }

                            list.child(container)
                        }
                        redesmyn_session_view_model::SessionEventItemContent::UserMessage(msg)
                        | redesmyn_session_view_model::SessionEventItemContent::AssistantMessage(
                            msg,
                        ) => {
                            let redesmyn_session_view_model::MessageItem {
                                role,
                                text,
                                preview: _,
                                full_text_artifact,
                            } = msg;

                            let (bg, text_color, align_right) = match role {
                                redesmyn_session_view_model::SessionMessageRole::User => {
                                    let (bg, text_color) = match theme.mode {
                                        ThemeMode::Dark => (
                                            theme.colors.accent_foreground.opacity(0.9),
                                            theme.colors.surface,
                                        ),
                                        ThemeMode::Light => (
                                            theme.colors.accent.opacity(0.9),
                                            theme.colors.foreground,
                                        ),
                                    };
                                    (Some(bg), Some(text_color), true)
                                }
                                redesmyn_session_view_model::SessionMessageRole::Assistant => {
                                    (None, None, false)
                                }
                                redesmyn_session_view_model::SessionMessageRole::Tool => {
                                    (Some(theme.colors.surface_elevated.opacity(0.45)), None, false)
                                }
                            };

                            let show_truncation_notice = full_text_artifact.is_none();
                            let cached =
                                { markdown_cache.borrow().get(&session_event_id).cloned() };
                            let doc = cached.unwrap_or_else(|| {
                                let doc = Arc::new(parse_markdown(
                                    text.as_str(),
                                    MarkdownParseOptions::default(),
                                ));
                                markdown_cache
                                    .borrow_mut()
                                    .insert(session_event_id, doc.clone());
                                doc
                            });

                            let bubble_max_width = if align_right {
                                viewport_width * 0.85
                            } else {
                                px(560.0)
                            };
                            let bubble_padding_x = if align_right {
                                theme.spacing.md
                            } else {
                                theme.spacing.sm
                            };

                            let mut bubble = div()
                                .id(bubble_id.clone())
                                .flex()
                                .flex_col()
                                .gap(theme.spacing.sm)
                                .w_full()
                                .max_w(bubble_max_width)
                                .min_w_0()
                                .px(bubble_padding_x)
                                .py(theme.spacing.sm)
                                .rounded(theme.radius.xl)
                                .when_some(bg, |this, bg| this.bg(bg))
                                .child({
                                    let mut view =
                                        MarkdownView::new((bubble_id.clone(), "markdown"), doc)
                                            .show_truncation_notice(show_truncation_notice)
                                            .text_size(theme.typography.caption.size);
                                    if let Some(text_color) = text_color {
                                        view = view.text_color(text_color);
                                    }
                                    view.into_any_element()
                                });

                            if let Some(artifact) = full_text_artifact {
                                let artifact_id = artifact.artifact_id.to_string();
                                let artifact_copy = artifact_id.clone();

                                bubble = bubble.child(
                                    div()
                                        .id((bubble_id.clone(), "artifact"))
                                        .flex()
                                        .flex_row()
                                        .gap(theme.spacing.sm)
                                        .items_center()
                                        .justify_between()
                                        .px(theme.spacing.md)
                                        .py(theme.spacing.sm)
                                        .rounded_sm()
                                        .bg(theme.colors.surface_elevated.opacity(0.35))
                                        .child(
                                            div()
                                                .flex_1()
                                                .min_w_0()
                                                .text_xs()
                                                .text_color(theme.colors.foreground_muted)
                                                .child(format!(
                                                    "Full message stored as artifact {artifact_id}."
                                                )),
                                        )
                                        .child(
                                            TextButton::new(
                                                (bubble_id.clone(), "copy_artifact"),
                                                "Copy artifact id",
                                            )
                                            .kind(ButtonKind::Ghost)
                                            .on_click(move |event, _window, cx| {
                                                if event.standard_click() {
                                                    cx.write_to_clipboard(
                                                        ClipboardItem::new_string(
                                                            artifact_copy.clone(),
                                                        ),
                                                    );
                                                }
                                            }),
                                        ),
                                );
                            }

                            let mut row = div()
                                .id((bubble_id.clone(), "row"))
                                .w_full()
                                .min_w_0()
                                .flex()
                                .flex_row()
                                .px(theme.spacing.sm);
                            if align_right {
                                row = row.justify_end();
                            } else {
                                row = row.justify_start();
                            }

                            let mut row_pad_top = timeline_item_gap_y;
                            let mut row_pad_bottom = timeline_item_gap_y;

                            if matches!(
                                role,
                                redesmyn_session_view_model::SessionMessageRole::User
                            ) {
                                let tool_event_hidden_in_collapsed_group =
                                    |session_event_id: SessionEventId| {
                                        let Some(member) =
                                            tool_event_group_membership.get(&session_event_id)
                                        else {
                                            return false;
                                        };

                                        let group_id = member.group_id;
                                        let is_expanded = expanded_tool_groups.contains(&group_id);
                                        let progress = tool_group_transitions
                                            .get(&group_id)
                                            .map(|transition| transition.value())
                                            .unwrap_or_else(|| {
                                                if is_expanded {
                                                    1.0
                                                } else {
                                                    0.0
                                                }
                                            });
                                        !is_expanded && progress <= 1e-3 && !member.is_first
                                    };

                                let visible_neighbor_is_tool_call = |neighbor_ix: isize,
                                                                     step: isize| {
                                    let mut cursor = neighbor_ix;
                                    let len = items.len() as isize;
                                    while cursor >= 0 && cursor < len {
                                        let candidate_ix = cursor as usize;
                                        let Some(item) = items.get(candidate_ix) else {
                                            break;
                                        };

                                        let visible_kind = match item {
                                            SessionTimelineItem::Event(event) => {
                                                let session_event_id = event.session_event_id;
                                                match &event.content {
                                                    SessionEventItemContent::ToolInvocation(_tool)
                                                        if tool_event_hidden_in_collapsed_group(
                                                            session_event_id,
                                                        ) =>
                                                    {
                                                        None
                                                    }
                                                    SessionEventItemContent::ToolResult(_tool)
                                                        if grouped_exec_command_result_ids
                                                            .contains(&session_event_id)
                                                            || tool_event_hidden_in_collapsed_group(
                                                                session_event_id,
                                                            ) =>
                                                    {
                                                        None
                                                    }
                                                    SessionEventItemContent::ToolInvocation(_)
                                                    | SessionEventItemContent::ToolResult(_) => {
                                                        Some(true)
                                                    }
                                                    SessionEventItemContent::UserMessage(_)
                                                    | SessionEventItemContent::AssistantMessage(_)
                                                    | SessionEventItemContent::ArtifactEmitted(
                                                        _,
                                                    ) => Some(false),
                                                    _ => None,
                                                }
                                            }
                                            _ => Some(false),
                                        };

                                        if let Some(is_tool_call) = visible_kind {
                                            return is_tool_call;
                                        }

                                        cursor += step;
                                    }

                                    false
                                };

                                let extra_gap = theme.spacing.sm;
                                if visible_neighbor_is_tool_call(ix as isize - 1, -1) {
                                    row_pad_top += extra_gap;
                                }
                                if visible_neighbor_is_tool_call(ix as isize + 1, 1) {
                                    row_pad_bottom += extra_gap;
                                }
                            }

                            row = row.pt(row_pad_top).pb(row_pad_bottom);

                            list.child(row.child(bubble))
                        }
                        other => {
                            // Hide session lifecycle and status noise in the primary timeline.
                            // (These remain available via the semantic snapshot/debug surfaces.)
                            match other {
                                SessionEventItemContent::ToolInvocation(tool) => {
                                    let group_context = tool_event_group_membership
                                        .get(&item.session_event_id)
                                        .and_then(|member| {
                                            tool_event_groups
                                                .get(&member.group_id)
                                                .map(|group| (*member, group))
                                        });
                                    let group_progress = group_context.as_ref().map(|(member, _)| {
                                        tool_group_transitions
                                            .get(&member.group_id)
                                            .map(|transition| transition.value())
                                            .unwrap_or_else(|| {
                                                if expanded_tool_groups.contains(&member.group_id) {
                                                    1.0
                                                } else {
                                                    0.0
                                                }
                                            })
                                    });
                                    let group_state = group_context.map(|(member, group)| {
                                        let group_id = member.group_id;
                                        let is_expanded = expanded_tool_groups.contains(&group_id);
                                        let progress = group_progress
                                            .unwrap_or_else(|| if is_expanded { 1.0 } else { 0.0 });
                                        (member, group_id, group, progress, is_expanded)
                                    });

                                    if let Some((member, group_id, group, progress, is_expanded)) =
                                        group_state
                                        && !is_expanded
                                        && progress <= 1e-3
                                    {
                                        if member.is_first {
                                            list.child(render_tool_group_row(
                                                "▸",
                                                group_id,
                                                group.kind.title(),
                                                group.count,
                                                group.last_summary.clone(),
                                            ))
                                        } else {
                                            div()
                                        }
                                    } else {
                                        let expanded =
                                            expanded_tool_events.contains(&item.session_event_id);
                                        let chevron = if expanded { "▾" } else { "▸" };
                                        let toggle_view = timeline_view.clone();
                                        let session_event_id = item.session_event_id;

                                        let is_exec_command = tool.tool_name == "exec_command";
                                        let exec_command_result = is_exec_command
                                            .then(|| exec_command_results.get(&session_event_id))
                                            .flatten();

                                        let (exec_command, exec_cwd) = if is_exec_command {
                                            parse_exec_command_input_preview(&tool.input_preview)
                                        } else {
                                            (None, None)
                                        };

                                        let summary_main = if is_exec_command {
                                            exec_command
                                                .as_deref()
                                                .map(tidy_shell_command)
                                                .unwrap_or_else(|| tool.input_preview.clone())
                                        } else {
                                            tool.input_preview.clone()
                                        };

                                        let exit_code = exec_command_result
                                            .and_then(|(_id, result)| {
                                                split_exec_command_exit_code(
                                                    &result.output_preview,
                                                )
                                                .0
                                            });

                                        let summary = div()
                                            .id((bubble_id.clone(), "summary"))
                                            .flex()
                                            .flex_row()
                                            .items_center()
                                            .gap(theme.spacing.sm)
                                            .font(theme.typography.mono.font.clone())
                                            .text_size(theme.typography.caption.size)
                                            .cursor_pointer()
                                            .focusable()
                                            .on_click(move |event, _window, cx| {
                                                if event.standard_click() {
                                                    toggle_view.update(cx, |this, cx| {
                                                        this.toggle_tool_event(session_event_id, cx);
                                                    });
                                                }
                                            })
                                            .child(
                                                div()
                                                    .flex_shrink_0()
                                                    .whitespace_nowrap()
                                                    .text_color(theme.colors.foreground_muted)
                                                    .child(chevron),
                                            )
                                            .child(
                                                div()
                                                    .flex_shrink_0()
                                                    .whitespace_nowrap()
                                                    .text_color(theme.colors.foreground_muted)
                                                    .child(tool.tool_name.clone()),
                                            )
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .text_color(theme.colors.foreground)
                                                    .truncate()
                                                    .child(summary_main),
                                            )
                                            .when_some(exit_code, |this, exit_code| {
                                                let color = if exit_code == 0 {
                                                    theme.colors.foreground_muted
                                                } else {
                                                    theme.colors.danger
                                                };
                                                this.child(
                                                    div()
                                                        .flex_shrink_0()
                                                        .whitespace_nowrap()
                                                        .text_color(color)
                                                        .child(format!("exit {exit_code}")),
                                                )
                                            });

                                        let mut block = div()
                                            .id(bubble_id.clone())
                                            .w_full()
                                            .min_w_0()
                                            .flex()
                                            .flex_col()
                                            .gap(theme.spacing.xs)
                                            .px(theme.spacing.sm)
                                            .py(timeline_item_gap_y)
                                            .child(summary);

                                        if expanded {
                                            let mut details = div()
                                                .id((bubble_id.clone(), "details"))
                                                .pl(theme.spacing.lg)
                                                .flex()
                                                .flex_col()
                                                .gap(theme.spacing.xs)
                                                .font(theme.typography.mono.font.clone())
                                                .text_size(theme.typography.caption.size);

                                            if is_exec_command {
                                                if let Some(cwd) = exec_cwd.as_deref() {
                                                    details = details.child(
                                                        div()
                                                            .text_color(theme.colors.foreground_muted)
                                                            .child(format!("cwd: {cwd}")),
                                                    );
                                                }

                                                details = details.child(
                                                    div()
                                                        .text_color(theme.colors.foreground)
                                                        .child(
                                                            exec_command
                                                                .as_deref()
                                                                .map(tidy_shell_command)
                                                                .unwrap_or_else(|| tool.input_preview.clone()),
                                                        ),
                                                );

                                                if let Some((_result_id, result)) = exec_command_result {
                                                    let (exit_code, remainder) =
                                                        split_exec_command_exit_code(
                                                            &result.output_preview,
                                                        );
                                                    if let Some(exit_code) = exit_code {
                                                        let color = if exit_code == 0 {
                                                            theme.colors.foreground_muted
                                                        } else {
                                                            theme.colors.danger
                                                        };
                                                        details = details.child(
                                                            div()
                                                                .text_color(color)
                                                                .child(format!("exit {exit_code}")),
                                                        );
                                                    }

                                                    if let Some(error) = result.error.as_ref() {
                                                        details = details.child(
                                                            div()
                                                                .text_color(theme.colors.danger)
                                                                .child(error.message.clone()),
                                                        );
                                                    }

                                                    if !remainder.is_empty() {
                                                        details = details.child(
                                                            div()
                                                                .text_color(theme.colors.foreground)
                                                                .child(remainder.to_string()),
                                                        );
                                                    }
                                                }
                                            } else {
                                                details = details.child(
                                                    div()
                                                        .text_color(theme.colors.foreground)
                                                        .child(tool.input_preview.clone()),
                                                );
                                            }

                                            block = block.child(details);
                                        }

                                        if let Some((member, group_id, group, progress, _)) =
                                            group_state
                                        {
                                            let opacity = progress;
                                            let is_animating =
                                                tool_group_transitions.contains_key(&group_id);
                                            let max_h = tool_group_member_row_height * opacity;

                                            if member.is_first {
                                                let group_key = session_event_id_key(group_id);
                                                let gap = if is_animating {
                                                    theme.spacing.xs * opacity
                                                } else {
                                                    theme.spacing.xs
                                                };
                                                let group_header = render_tool_group_row(
                                                    "▾",
                                                    group_id,
                                                    group.kind.title(),
                                                    group.count,
                                                    group.last_summary.clone(),
                                                );

                                                let mut member_body = div()
                                                    .w_full()
                                                    .min_w_0()
                                                    .pl(theme.spacing.lg)
                                                    .opacity(opacity);
                                                if is_animating {
                                                    member_body = member_body
                                                        .overflow_hidden()
                                                        .max_h(max_h);
                                                }

                                                list.child(
                                                    div()
                                                        .id(("session_tool_group_container", group_key))
                                                        .w_full()
                                                        .min_w_0()
                                                        .flex()
                                                        .flex_col()
                                                        .gap(gap)
                                                        .child(group_header)
                                                        .child(member_body.child(block)),
                                                )
                                            } else {
                                                let mut member_row = div()
                                                    .id((bubble_id.clone(), "group_member"))
                                                    .w_full()
                                                    .min_w_0()
                                                    .pl(theme.spacing.lg)
                                                    .opacity(opacity);
                                                if is_animating {
                                                    member_row =
                                                        member_row.overflow_hidden().max_h(max_h);
                                                }
                                                list.child(member_row.child(block))
                                            }
                                        } else {
                                            list.child(block)
                                        }
                                    }
                                }
                                SessionEventItemContent::ToolResult(tool) => {
                                    if grouped_exec_command_result_ids.contains(&item.session_event_id) {
                                        div()
                                    } else {
                                        let group_context = tool_event_group_membership
                                            .get(&item.session_event_id)
                                            .and_then(|member| {
                                                tool_event_groups
                                                    .get(&member.group_id)
                                                    .map(|group| (*member, group))
                                            });
                                        let group_progress = group_context.as_ref().map(|(member, _)| {
                                            tool_group_transitions
                                                .get(&member.group_id)
                                                .map(|transition| transition.value())
                                                .unwrap_or_else(|| {
                                                    if expanded_tool_groups.contains(&member.group_id) {
                                                        1.0
                                                    } else {
                                                        0.0
                                                    }
                                                })
                                        });
                                        let group_state = group_context.map(|(member, group)| {
                                            let group_id = member.group_id;
                                            let is_expanded = expanded_tool_groups.contains(&group_id);
                                            let progress = group_progress
                                                .unwrap_or_else(|| if is_expanded { 1.0 } else { 0.0 });
                                            (member, group_id, group, progress, is_expanded)
                                        });

                                        if let Some((member, group_id, group, progress, is_expanded)) =
                                            group_state
                                            && !is_expanded
                                            && progress <= 1e-3
                                        {
                                            if member.is_first {
                                                list.child(render_tool_group_row(
                                                    "▸",
                                                    group_id,
                                                    group.kind.title(),
                                                    group.count,
                                                    group.last_summary.clone(),
                                                ))
                                            } else {
                                                div()
                                            }
                                        } else {
                                            let expanded =
                                                expanded_tool_events.contains(&item.session_event_id);
                                            let chevron = if expanded { "▾" } else { "▸" };
                                            let toggle_view = timeline_view.clone();
                                            let session_event_id = item.session_event_id;

                                            let has_error = tool.error.is_some();

                                            let summary = div()
                                                .id((bubble_id.clone(), "summary"))
                                                .flex()
                                                .flex_row()
                                                .items_center()
                                                .gap(theme.spacing.sm)
                                                .font(theme.typography.mono.font.clone())
                                                .text_size(theme.typography.caption.size)
                                                .cursor_pointer()
                                                .focusable()
                                                .on_click(move |event, _window, cx| {
                                                    if event.standard_click() {
                                                        toggle_view.update(cx, |this, cx| {
                                                            this.toggle_tool_event(session_event_id, cx);
                                                        });
                                                    }
                                                })
                                                .child(
                                                    div()
                                                        .flex_shrink_0()
                                                        .whitespace_nowrap()
                                                        .text_color(theme.colors.foreground_muted)
                                                        .child(chevron),
                                                )
                                                .child(
                                                    div()
                                                        .flex_shrink_0()
                                                        .whitespace_nowrap()
                                                        .text_color(theme.colors.foreground_muted)
                                                        .child(tool.tool_name.clone()),
                                                )
                                                .child(
                                                    div()
                                                        .flex_1()
                                                        .min_w_0()
                                                        .text_color(theme.colors.foreground)
                                                        .truncate()
                                                        .child(tool.output_preview.clone()),
                                                )
                                                .when(has_error, |this| {
                                                    this.child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(theme.colors.danger)
                                                            .child("error"),
                                                    )
                                                });

                                            let mut block = div()
                                                .id(bubble_id.clone())
                                                .w_full()
                                                .min_w_0()
                                                .flex()
                                                .flex_col()
                                                .gap(theme.spacing.xs)
                                                .px(theme.spacing.sm)
                                                .py(timeline_item_gap_y)
                                                .child(summary);

                                            if expanded {
                                                let mut details = div()
                                                    .id((bubble_id.clone(), "details"))
                                                    .pl(theme.spacing.lg)
                                                    .flex()
                                                    .flex_col()
                                                    .gap(theme.spacing.xs)
                                                    .font(theme.typography.mono.font.clone())
                                                    .text_size(theme.typography.caption.size);

                                                if let Some(error) = tool.error.as_ref() {
                                                    details = details.child(
                                                        div()
                                                            .text_color(theme.colors.danger)
                                                            .child(error.message.clone()),
                                                    );
                                                }

                                                details = details.child(
                                                    div()
                                                        .text_color(theme.colors.foreground)
                                                        .child(tool.output_preview.clone()),
                                                );

                                                block = block.child(details);
                                            }

                                            if let Some((member, group_id, group, progress, _)) =
                                                group_state
                                            {
                                                let opacity = progress;
                                                let is_animating =
                                                    tool_group_transitions.contains_key(&group_id);
                                                let max_h = tool_group_member_row_height * opacity;

                                                if member.is_first {
                                                    let group_key = session_event_id_key(group_id);
                                                    let gap = if is_animating {
                                                        theme.spacing.xs * opacity
                                                    } else {
                                                        theme.spacing.xs
                                                    };
                                                    let group_header = render_tool_group_row(
                                                        "▾",
                                                        group_id,
                                                        group.kind.title(),
                                                        group.count,
                                                        group.last_summary.clone(),
                                                    );

                                                    let mut member_body = div()
                                                        .w_full()
                                                        .min_w_0()
                                                        .pl(theme.spacing.lg)
                                                        .opacity(opacity);
                                                    if is_animating {
                                                        member_body = member_body
                                                            .overflow_hidden()
                                                            .max_h(max_h);
                                                    }

                                                    list.child(
                                                        div()
                                                            .id(("session_tool_group_container", group_key))
                                                            .w_full()
                                                            .min_w_0()
                                                            .flex()
                                                            .flex_col()
                                                            .gap(gap)
                                                            .child(group_header)
                                                            .child(member_body.child(block)),
                                                    )
                                                } else {
                                                    let mut member_row = div()
                                                        .id((bubble_id.clone(), "group_member"))
                                                        .w_full()
                                                        .min_w_0()
                                                        .pl(theme.spacing.lg)
                                                        .opacity(opacity);
                                                    if is_animating {
                                                        member_row = member_row
                                                            .overflow_hidden()
                                                            .max_h(max_h);
                                                    }
                                                    list.child(member_row.child(block))
                                                }
                                            } else {
                                                list.child(block)
                                            }
                                        }
                                    }
                                }
                                redesmyn_session_view_model::SessionEventItemContent::ArtifactEmitted(
                                    artifact,
                                ) => list.child(
                                    div()
                                        .id(bubble_id)
                                        .w_full()
                                        .min_w_0()
                                        .px(theme.spacing.sm)
                                        .py(theme.spacing.sm)
                                        .rounded_sm()
                                        .bg(theme.colors.surface_elevated.opacity(0.35))
                                        .text_sm()
                                        .text_color(theme.colors.foreground)
                                        .child(
                                            artifact
                                                .label
                                                .as_deref()
                                                .unwrap_or("Artifact emitted")
                                                .to_string(),
                                        ),
                                ),
                                _ => div(),
                            }
                        }
                    }
                }
            };

            rendered.into_any_element()
        })
        .flex_1()
        .min_h(px(0.0))
        .w_full()
        .min_w_0()
        .bg(theme.colors.surface);

        let mut composer_callout = None;
        if let Some(feed) = self.feed.as_ref() {
            if let Some(prompt) = feed.composer.conflict_prompt.clone() {
                let (message, action): (SharedString, _) = match prompt.code.as_str() {
                    CONFLICT_CODE_TURN_IN_PROGRESS => (
                        "A structured agent turn is currently in progress. Interrupt it and send your message?".into(),
                        div()
                            .flex()
                            .gap(theme.spacing.sm)
                            .child(
                                TextButton::new(
                                    ("session_conflict_cancel", cx.entity_id()),
                                    "Cancel",
                                )
                                .kind(ButtonKind::Secondary)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| {
                                            if let Some(feed) = this.feed.as_mut() {
                                                feed.composer.conflict_prompt = None;
                                            }
                                            cx.notify();
                                        });
                                    }
                                }),
                            )
                            .child(
                                TextButton::new(
                                    ("session_conflict_interrupt", cx.entity_id()),
                                    "Interrupt & send",
                                )
                                .kind(ButtonKind::Primary)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| {
                                            this.send_message(
                                                AgentMessageConflictAction::InterruptTurn,
                                                cx,
                                            );
                                        });
                                    }
                                }),
                            ),
                    ),
                    CONFLICT_CODE_TASK_SESSION_CONFLICT => (
                        "Another agent session is already running for this task. Stop it and send your message?".into(),
                        div()
                            .flex()
                            .gap(theme.spacing.sm)
                            .child(
                                TextButton::new(
                                    ("session_conflict_cancel", cx.entity_id()),
                                    "Cancel",
                                )
                                .kind(ButtonKind::Secondary)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| {
                                            if let Some(feed) = this.feed.as_mut() {
                                                feed.composer.conflict_prompt = None;
                                            }
                                            cx.notify();
                                        });
                                    }
                                }),
                            )
                            .child(
                                TextButton::new(
                                    ("session_conflict_stop", cx.entity_id()),
                                    "Stop session & send",
                                )
                                .kind(ButtonKind::Danger)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| {
                                            this.send_message(
                                                AgentMessageConflictAction::StopSessionAndStartNew,
                                                cx,
                                            );
                                        });
                                    }
                                }),
                            ),
                    ),
                    _ => (
                        prompt.message.clone().into(),
                        div().flex().gap(theme.spacing.sm).child(
                            TextButton::new(("session_conflict_cancel", cx.entity_id()), "Close")
                                .kind(ButtonKind::Secondary)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| {
                                            if let Some(feed) = this.feed.as_mut() {
                                                feed.composer.conflict_prompt = None;
                                            }
                                            cx.notify();
                                        });
                                    }
                                }),
                        ),
                    ),
                };
                composer_callout = Some(
                    Callout::new(message)
                        .kind(CalloutKind::Warning)
                        .title("Send message")
                        .action(action)
                        .into_any_element(),
                );
            } else if let Some(error) = feed.composer.last_error.clone() {
                composer_callout = Some(
                    Callout::new(error)
                        .kind(CalloutKind::Danger)
                        .title("Send message")
                        .into_any_element(),
                );
            }
        }

        let composer_draft = self
            .feed
            .as_ref()
            .map(|feed| feed.composer.draft.as_str())
            .unwrap_or_default();
        let composer_sending = self.feed.as_ref().is_some_and(|feed| feed.composer.sending);
        let (composer_can_send, disabled_reason) = if composer_sending {
            (false, "Sending…")
        } else if self.client.is_none() || self.feed.is_none() {
            (false, "Chat is unavailable")
        } else if composer_draft.trim().is_empty() {
            (false, "Message is empty")
        } else {
            (true, "")
        };

        let send_button =
            IconButton::new(("session_send_message", cx.entity_id()), div().child("↑"))
                .tooltip("Send (⌘⏎ / ⇧⏎)")
                .disabled(!composer_can_send)
                .disabled_reason(disabled_reason)
                .on_click({
                    let view = view.clone();
                    move |_, _, cx| {
                        view.update(cx, |this, cx| {
                            this.send_message(AgentMessageConflictAction::Fail, cx)
                        });
                    }
                });

        let composer = div()
            .relative()
            .w_full()
            .child(self.composer_input.clone())
            .child(
                div()
                    .absolute()
                    .right(theme.spacing.sm)
                    .bottom(theme.spacing.sm)
                    .child(send_button),
            );

        let mut timeline = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .child(feed_list);

        if let Some(callout) = composer_callout {
            timeline = timeline.child(callout);
        }

        timeline = timeline.child(composer);
        content = content.child(timeline);

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .size_full()
            .child(content)
            .track_focus(&self.focus_handle(cx))
    }
}
