//! Session viewer scaffolding for GPUI (Domain 7).

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Instant;

use gpui::{
    App, AsyncApp, ClickEvent, ClipboardItem, Context, ElementId, Entity, FocusHandle, Focusable,
    ListState, Render, SharedString, Subscription, Task, WeakEntity, Window, div, list, px,
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
use redesmyn_protocol::session::SessionEventKind;
use redesmyn_protocol::ui_driver::UiComposerState;
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, SessionEvent};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;

use redesmyn_session_view_model::{SessionFeedState, SessionTimelineItem};
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, MarkdownView, TextArea, TextButton, TextInput,
    TextInputEvent,
};
use redesmyn_ui::utils::theme_for_window;

fn session_event_id_key(id: SessionEventId) -> u64 {
    let bytes = id.to_bytes();
    u64::from_be_bytes(bytes[0..8].try_into().expect("slice length"))
}

const CONFLICT_CODE_KEY: &str = "conflict_code";
const CONFLICT_CODE_TURN_IN_PROGRESS: &str = "structured_turn_in_progress";
// NOTE: `conflict_code` values are part of the client-visible contract.
// `structured_session_conflict` is currently returned for any concurrent task session (even if the
// conflicting session isn't structured). Consider renaming if we need semantic precision.
const CONFLICT_CODE_TASK_SESSION_CONFLICT: &str = "structured_session_conflict";

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

fn sync_list_state(
    list_state: &ListState,
    old_items: &[SessionTimelineItem],
    new_items: &[SessionTimelineItem],
) {
    if old_items == new_items {
        return;
    }

    let old_len = old_items.len();
    let new_len = new_items.len();

    let mut prefix = 0;
    while prefix < old_len && prefix < new_len && old_items[prefix] == new_items[prefix] {
        prefix += 1;
    }

    let mut suffix = 0;
    while suffix < old_len.saturating_sub(prefix)
        && suffix < new_len.saturating_sub(prefix)
        && old_items[old_len - 1 - suffix] == new_items[new_len - 1 - suffix]
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
    timeline_scroll_handler_installed: bool,
    timeline_viewport_width: Option<gpui::Pixels>,
    timeline_needs_refresh: bool,
    show_debug_controls: bool,
    session_id_input: Entity<TextInput>,
    composer_input: Entity<TextArea>,
    pending_focus_composer: bool,
    feed: Option<SessionFeedState>,
    expanded_tool_events: HashSet<SessionEventId>,
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
        let timeline_list_state = ListState::new(0, gpui::ListAlignment::Top, px(400.0));
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
            timeline_scroll_handler_installed: false,
            timeline_viewport_width: None,
            timeline_needs_refresh: false,
            show_debug_controls,
            session_id_input,
            composer_input,
            pending_focus_composer: false,
            feed: None,
            expanded_tool_events: HashSet::new(),
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
            self.expanded_tool_events.clear();
            self.markdown_cache.borrow_mut().clear();
            self.set_timeline_items(Vec::new());
            self.timeline_needs_refresh = false;
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
        self.expanded_tool_events.clear();
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
                self.apply_scroll_intents();
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
                self.scroll_to_bottom();
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

    fn on_subscription_event(
        &mut self,
        event: SubscriptionEvent,
        cx: &mut Context<Self>,
    ) {
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
                self.apply_scroll_intents();
            }
            SubscriptionEvent::SessionLiveEvent(ev) => {
                if ev.session_id != feed.session_id {
                    return;
                }
                feed.apply_live_session_event(ev);
                self.refresh_timeline_items();
                self.apply_scroll_intents();
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

        if feed.history.loading_older || feed.history.next_cursor.is_none() {
            return;
        }

        let scroll_top = self.timeline_list_state.logical_scroll_top();
        if scroll_top.item_ix != 0 || scroll_top.offset_in_item > px(24.0) {
            return;
        }

        self.start_loading_older(cx);
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

    fn toggle_tool_event(&mut self, session_event_id: SessionEventId, cx: &mut Context<Self>) {
        if self.expanded_tool_events.contains(&session_event_id) {
            self.expanded_tool_events.remove(&session_event_id);
        } else {
            self.expanded_tool_events.insert(session_event_id);
        }
        self.invalidate_timeline_item(session_event_id);
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
        self.scroll_to_bottom();
        cx.notify();
    }

    fn set_timeline_items(&mut self, items: Vec<SessionTimelineItem>) {
        let old_items = std::mem::replace(&mut self.timeline_items, Rc::new(items));
        sync_list_state(
            &self.timeline_list_state,
            old_items.as_ref(),
            self.timeline_items.as_ref(),
        );
    }

    fn refresh_timeline_items(&mut self) {
        let items = self
            .feed
            .as_ref()
            .map(SessionFeedState::timeline_items)
            .unwrap_or_default();
        self.set_timeline_items(items);
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

    fn apply_scroll_intents(&mut self) {
        let should_scroll = self
            .feed
            .as_ref()
            .is_some_and(|feed| feed.scroll.pending_scroll_to_bottom);
        if !should_scroll {
            return;
        }

        self.scroll_to_bottom();
        if let Some(feed) = self.feed.as_mut() {
            feed.clear_scroll_intents();
        }
    }

    fn scroll_to_bottom(&mut self) {
        let Some(ix) = self.timeline_items.len().checked_sub(1) else {
            return;
        };
        self.timeline_list_state.scroll_to_reveal_item(ix);
    }
}

impl Render for SessionView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let theme = theme_for_window(window, cx);
        let view = cx.entity();

        if !self.timeline_scroll_handler_installed {
            let view = view.clone();
            self.timeline_list_state
                .set_scroll_handler(move |event, _window, cx| {
                    view.update(cx, |this, cx| {
                        let Some(feed) = this.feed.as_mut() else {
                            return;
                        };

                        let was_at_bottom = feed.scroll.at_bottom;
                        let at_bottom = event.visible_range.end >= event.count;
                        feed.set_at_bottom(at_bottom);

                        if feed.scroll.at_bottom != was_at_bottom {
                            this.timeline_needs_refresh = true;
                        }
                        cx.notify();
                    });
                });
            self.timeline_scroll_handler_installed = true;
        }

        if self.timeline_needs_refresh {
            self.timeline_needs_refresh = false;
            self.refresh_timeline_items();
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
                self.timeline_list_state.reset(self.timeline_items.len());
                self.timeline_list_state.scroll_to(scroll_top);
            }
            self.timeline_viewport_width = Some(viewport_width);
        }

        // Note: do not call `ListState` methods from within the scroll handler (it runs while the
        // list state is mutably borrowed). We instead evaluate autoloading here during render.
        self.maybe_autoload_older(cx);

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
        let entity_id = cx.entity_id();
        let timeline_view = view.clone();

        let feed_list = list(self.timeline_list_state.clone(), move |ix, window, cx| {
            let theme = theme_for_window(window, cx);
            let list = div().w_full().min_w_0().pb(theme.spacing.sm);

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
                SessionTimelineItem::Event(item) => {
                    let session_event_id = item.session_event_id;
                    let event_key = session_event_id_key(item.session_event_id);
                    let bubble_id: ElementId = ("session_event", event_key).into();

                    match item.content {
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

                            let (bg, align_right) = match role {
                                redesmyn_session_view_model::SessionMessageRole::User => {
                                    (Some(theme.colors.accent.opacity(0.4)), true)
                                }
                                redesmyn_session_view_model::SessionMessageRole::Assistant => {
                                    (None, false)
                                }
                                redesmyn_session_view_model::SessionMessageRole::Tool => {
                                    (Some(theme.colors.surface_elevated.opacity(0.45)), false)
                                }
                            };

                            let show_truncation_notice = full_text_artifact.is_none();
                            let cached = { markdown_cache.borrow().get(&session_event_id).cloned() };
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

                            let mut bubble = div()
                                .id(bubble_id.clone())
                                .flex()
                                .flex_col()
                                .gap(theme.spacing.sm)
                                .w_full()
                                .max_w(px(560.0))
                                .min_w_0()
                                .px(theme.spacing.md)
                                .py(theme.spacing.md)
                                .rounded_md()
                                .when_some(bg, |this, bg| this.bg(bg))
                                .child(
                                    MarkdownView::new((bubble_id.clone(), "markdown"), doc)
                                        .show_truncation_notice(show_truncation_notice)
                                        .into_any_element(),
                                );

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

                            list.child(row.child(bubble))
                        }
                        other => {
                            // Hide session lifecycle and status noise in the primary timeline.
                            // (These remain available via the semantic snapshot/debug surfaces.)
                            match other {
                                redesmyn_session_view_model::SessionEventItemContent::ToolInvocation(
                                    tool,
                                ) => {
                                    let expanded = expanded_tool_events.contains(&item.session_event_id);
                                    let chevron = if expanded { "▾" } else { "▸" };
                                    let toggle_view = timeline_view.clone();
                                    let session_event_id = item.session_event_id;
                                    let summary_text =
                                        format!("{} — {}", tool.tool_name, tool.input_preview);

                                    let summary = div()
                                        .id((bubble_id.clone(), "summary"))
                                        .flex()
                                        .flex_row()
                                        .items_center()
                                        .gap(theme.spacing.sm)
                                        .px(theme.spacing.md)
                                        .py(theme.spacing.sm)
                                        .rounded_sm()
                                        .bg(theme.colors.surface_elevated.opacity(0.3))
                                        .hover(|this| this.bg(theme.colors.surface_elevated.opacity(0.4)))
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
                                                .text_xs()
                                                .text_color(theme.colors.foreground_muted)
                                                .child(chevron),
                                        )
                                        .child(
                                            div()
                                                .flex_1()
                                                .min_w_0()
                                                .text_sm()
                                                .text_color(theme.colors.foreground)
                                                .truncate()
                                                .child(summary_text),
                                        );

                                    let mut block = div()
                                        .id(bubble_id.clone())
                                        .w_full()
                                        .min_w_0()
                                        .flex()
                                        .flex_col()
                                        .gap(theme.spacing.xs)
                                        .child(summary);

                                    if expanded {
                                        let mut details = div()
                                            .id((bubble_id.clone(), "details"))
                                            .px(theme.spacing.md)
                                            .py(theme.spacing.sm)
                                            .rounded_sm()
                                            .bg(theme.colors.surface_elevated.opacity(0.18))
                                            .flex()
                                            .flex_col()
                                            .gap(theme.spacing.xs);

                                        if let Some(call_id) = tool.tool_call_id.as_ref() {
                                            details = details.child(
                                                div()
                                                    .text_xs()
                                                    .text_color(theme.colors.foreground_muted)
                                                    .child(format!("call id: {call_id}")),
                                            );
                                        }

                                        details = details.child(
                                            div()
                                                .font(theme.typography.mono.font.clone())
                                                .text_size(theme.typography.mono.size)
                                                .text_color(theme.colors.foreground)
                                                .child(tool.input_preview.clone()),
                                        );

                                        block = block.child(details);
                                    }

                                    list.child(block)
                                }
                                redesmyn_session_view_model::SessionEventItemContent::ToolResult(
                                    tool,
                                ) => {
                                    let expanded = expanded_tool_events.contains(&item.session_event_id);
                                    let chevron = if expanded { "▾" } else { "▸" };
                                    let toggle_view = timeline_view.clone();
                                    let session_event_id = item.session_event_id;

                                    let summary_text =
                                        format!("{} — {}", tool.tool_name, tool.output_preview);
                                    let has_error = tool.error.is_some();

                                    let summary = div()
                                        .id((bubble_id.clone(), "summary"))
                                        .flex()
                                        .flex_row()
                                        .items_center()
                                        .gap(theme.spacing.sm)
                                        .px(theme.spacing.md)
                                        .py(theme.spacing.sm)
                                        .rounded_sm()
                                        .bg(theme.colors.surface_elevated.opacity(0.3))
                                        .hover(|this| this.bg(theme.colors.surface_elevated.opacity(0.4)))
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
                                                .text_xs()
                                                .text_color(theme.colors.foreground_muted)
                                                .child(chevron),
                                        )
                                        .child(
                                            div()
                                                .flex_1()
                                                .min_w_0()
                                                .text_sm()
                                                .text_color(theme.colors.foreground)
                                                .truncate()
                                                .child(summary_text),
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
                                        .child(summary);

                                    if expanded {
                                        let mut details = div()
                                            .id((bubble_id.clone(), "details"))
                                            .px(theme.spacing.md)
                                            .py(theme.spacing.sm)
                                            .rounded_sm()
                                            .bg(theme.colors.surface_elevated.opacity(0.18))
                                            .flex()
                                            .flex_col()
                                            .gap(theme.spacing.xs);

                                        if let Some(call_id) = tool.tool_call_id.as_ref() {
                                            details = details.child(
                                                div()
                                                    .text_xs()
                                                    .text_color(theme.colors.foreground_muted)
                                                    .child(format!("call id: {call_id}")),
                                            );
                                        }

                                        if let Some(error) = tool.error.as_ref() {
                                            details = details.child(
                                                div()
                                                    .text_sm()
                                                    .text_color(theme.colors.danger)
                                                    .child(error.message.clone()),
                                            );
                                        }

                                        details = details.child(
                                            div()
                                                .font(theme.typography.mono.font.clone())
                                                .text_size(theme.typography.mono.size)
                                                .text_color(theme.colors.foreground)
                                                .child(tool.output_preview.clone()),
                                        );

                                        block = block.child(details);
                                    }

                                    list.child(block)
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

        content = content.child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .min_h(px(0.0))
                .border_1()
                .border_color(theme.colors.border.opacity(0.4))
                .rounded_md()
                .bg(theme.colors.surface)
                .child(feed_list),
        );

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

                content = content.child(
                    Callout::new(message)
                        .kind(CalloutKind::Warning)
                        .title("Send message")
                        .action(action),
                );
            } else if let Some(error) = feed.composer.last_error.clone() {
                content = content.child(
                    Callout::new(error)
                        .kind(CalloutKind::Danger)
                        .title("Send message"),
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

        content = content.child(composer);

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .size_full()
            .child(content)
            .track_focus(&self.focus_handle(cx))
    }
}
