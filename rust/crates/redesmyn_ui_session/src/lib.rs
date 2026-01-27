//! Session viewer scaffolding for GPUI (Domain 7).

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;

use gpui::{
    App, AsyncApp, ClipboardItem, ClickEvent, Context, ElementId, Entity, FocusHandle, Focusable,
    Render, ScrollHandle, SharedString, Subscription, Task, WeakEntity, Window, div, px,
};

use gpui::prelude::*;

use redesmyn_markdown::{MarkdownDoc, MarkdownParseOptions, parse_markdown};
use redesmyn_client_api::Client;
use redesmyn_ids::{SessionEventId, SessionId, SubscriptionId};
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
    ButtonKind, Callout, CalloutKind, MarkdownView, ScrollArea, TextArea, TextButton, TextInput,
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

fn update_scroll_state(feed: &mut SessionFeedState, scroll_handle: &ScrollHandle) {
    let offset_y = scroll_handle.offset().y;
    let max_y = scroll_handle.max_offset().height;
    let threshold = px(24.0);
    let effective = if max_y > threshold {
        max_y - threshold
    } else {
        px(0.0)
    };
    let at_bottom = max_y == px(0.0) || offset_y <= -effective;
    feed.set_at_bottom(at_bottom);
}

fn apply_scroll_intents(feed: &mut SessionFeedState, scroll_handle: &ScrollHandle) {
    if feed.scroll.pending_scroll_to_bottom {
        scroll_handle.scroll_to_bottom();
        feed.clear_scroll_intents();
    }
}

#[derive(Clone, Copy)]
struct ScrollRestore {
    top_index: usize,
    top_offset: gpui::Pixels,
    top_event_id: Option<SessionEventId>,
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

        let doc = parse_markdown(text, options);
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
    scroll_handle: ScrollHandle,
    show_debug_controls: bool,
    session_id_input: Entity<TextInput>,
    composer_input: Entity<TextArea>,
    pending_focus_composer: bool,
    feed: Option<SessionFeedState>,
    markdown_cache: HashMap<SessionEventId, Arc<MarkdownDoc>>,
    client: Option<Client>,
    _client_task: Option<Task<()>>,
    subscription_task: Option<Task<()>>,
    subscription_id: Option<SubscriptionId>,
    load_task: Option<Task<()>>,
    load_older_task: Option<Task<()>>,
    send_task: Option<Task<()>>,
    pending_scroll_restore: Option<ScrollRestore>,
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
        let scroll_handle = ScrollHandle::new();
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
            scroll_handle,
            show_debug_controls,
            session_id_input,
            composer_input,
            pending_focus_composer: false,
            feed: None,
            markdown_cache: HashMap::new(),
            client,
            _client_task: client_task,
            subscription_task: None,
            subscription_id: None,
            load_task: None,
            load_older_task: None,
            send_task: None,
            pending_scroll_restore: None,
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
            self.pending_scroll_restore = None;
            self.error = None;
            self.feed = None;
            self.markdown_cache.clear();
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
        self.pending_scroll_restore = None;
        self.markdown_cache.clear();
        self.feed = Some(SessionFeedState::new(session_id));
        self.composer_input
            .update(cx, |input, cx| input.set_text("", cx));
        self.scroll_handle.scroll_to_bottom();
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
                    &mut self.markdown_cache,
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
                apply_scroll_intents(feed, &self.scroll_handle);
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

                let stats = cache_markdown_for_events(&mut self.markdown_cache, &resp.events);
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
                self.scroll_handle.scroll_to_bottom();
                feed.clear_scroll_intents();
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
        let scroll_handle = self.scroll_handle.clone();
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
                            this.on_subscription_event(event, &scroll_handle, cx);
                        })
                    });
                }
            }
        }));
    }

    fn on_subscription_event(
        &mut self,
        event: SubscriptionEvent,
        scroll_handle: &ScrollHandle,
        cx: &mut Context<Self>,
    ) {
        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        update_scroll_state(feed, scroll_handle);

        match event {
            SubscriptionEvent::SessionEvent(ev) => {
                if ev.session_id != feed.session_id {
                    return;
                }
                let stats = cache_markdown_for_events(
                    &mut self.markdown_cache,
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
                apply_scroll_intents(feed, scroll_handle);
            }
            SubscriptionEvent::SessionLiveEvent(ev) => {
                if ev.session_id != feed.session_id {
                    return;
                }
                feed.apply_live_session_event(ev);
                apply_scroll_intents(feed, scroll_handle);
            }
            SubscriptionEvent::Error(err) => {
                feed.apply_live_error(err.message);
            }
            SubscriptionEvent::Subscribed(_) | SubscriptionEvent::EventLog(_) => {}
        }

        cx.notify();
    }

    fn capture_scroll_restore(&mut self) {
        let Some(feed) = self.feed.as_ref() else {
            return;
        };

        let (ix, offset) = self.scroll_handle.logical_scroll_top();
        let items = feed.timeline_items();
        let top_event_id = items.get(ix).and_then(|item| match item {
            SessionTimelineItem::Event(ev) => Some(ev.session_event_id),
            _ => None,
        });

        self.pending_scroll_restore = Some(ScrollRestore {
            top_index: ix,
            top_offset: offset,
            top_event_id,
        });
    }

    fn restore_scroll_after_prepend(&mut self, restore: ScrollRestore, cx: &mut Context<Self>) {
        let Some(feed) = self.feed.as_ref() else {
            return;
        };

        let items = feed.timeline_items();
        let target_index = restore
            .top_event_id
            .and_then(|id| {
                items.iter().position(|item| match item {
                    SessionTimelineItem::Event(ev) => ev.session_event_id == id,
                    _ => false,
                })
            })
            .unwrap_or(restore.top_index)
            .min(items.len().saturating_sub(1));

        self.scroll_handle.scroll_to_top_of_item(target_index);

        let scroll_handle = self.scroll_handle.clone();
        let offset = restore.top_offset;
        cx.spawn(move |_: WeakEntity<Self>, _cx: &mut AsyncApp| async move {
            gpui::Timer::after(Duration::from_millis(0)).await;
            let current = scroll_handle.offset();
            scroll_handle.set_offset(gpui::point(current.x, current.y + offset));
        })
        .detach();
    }

    fn load_older(&mut self, _event: &ClickEvent, _window: &mut Window, cx: &mut Context<Self>) {
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

        self.capture_scroll_restore();
        if let Some(feed) = self.feed.as_mut() {
            feed.start_loading_older();
        }
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
                let stats = cache_markdown_for_events(&mut self.markdown_cache, &resp.events);
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
                if let Some(restore) = self.pending_scroll_restore.take() {
                    self.restore_scroll_after_prepend(restore, cx);
                }
            }
            Err(err) => {
                feed.apply_history_error(err.message);
            }
        }

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
        self.scroll_handle.scroll_to_bottom();
        cx.notify();
    }
}

impl Render for SessionView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let theme = theme_for_window(window, cx);
        let view = cx.entity();

        if self.pending_focus_composer && self.feed.is_some() {
            self.pending_focus_composer = false;
            window.focus(&self.composer_input.focus_handle(cx));
        }

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

        let items = self
            .feed
            .as_ref()
            .map(SessionFeedState::timeline_items)
            .unwrap_or_default();

        let feed_list = items.into_iter().enumerate().fold(
            div().flex().flex_col().gap(theme.spacing.sm),
            |list, (ix, item)| match item {
                SessionTimelineItem::LoadOlder(row) => {
                    let button = if row.in_flight {
                        TextButton::new(("session_load_older", cx.entity_id()), "Loading older…")
                            .disabled(true)
                    } else {
                        TextButton::new(("session_load_older", cx.entity_id()), "Load older")
                            .disabled(!row.enabled)
                    };

                    list.child(div().id(("session_item_load_older", ix)).w_full().child(
                        button.on_click({
                            let view = view.clone();
                            move |event, window, cx| {
                                view.update(cx, |this, cx| this.load_older(event, window, cx))
                            }
                        }),
                    ))
                }
                SessionTimelineItem::NewMessages(row) => list.child(
                    div().id(("session_item_new_messages", ix)).w_full().child(
                        TextButton::new(
                            ("session_jump_bottom", cx.entity_id()),
                            format!("New messages ({}) — Jump to bottom", row.count),
                        )
                        .on_click({
                            let view = view.clone();
                            move |event, window, cx| {
                                view.update(cx, |this, cx| this.jump_to_bottom(event, window, cx))
                            }
                        }),
                    ),
                ),
                SessionTimelineItem::EphemeralText(item) => list.child(
                    div()
                        .id(("session_item_ephemeral", ix))
                        .px(theme.spacing.sm)
                        .py(theme.spacing.sm)
                        .rounded_sm()
                        .bg(theme.colors.surface_elevated.opacity(0.6))
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child(format!(
                            "[{}] {}",
                            match item.role {
                                redesmyn_session_view_model::SessionMessageRole::User => "user",
                                redesmyn_session_view_model::SessionMessageRole::Assistant =>
                                    "assistant",
                                redesmyn_session_view_model::SessionMessageRole::Tool => "tool",
                            },
                            item.text
                        )),
                ),
                SessionTimelineItem::Event(item) => {
                    let event_key = session_event_id_key(item.session_event_id);
                    let bubble_id: ElementId = ("session_event", event_key).into();

                    match item.content {
                        redesmyn_session_view_model::SessionEventItemContent::UserMessage(msg)
                        | redesmyn_session_view_model::SessionEventItemContent::AssistantMessage(
                            msg,
                        ) => {
                            let (role_label, bg) = match msg.role {
                                redesmyn_session_view_model::SessionMessageRole::User => (
                                    "user",
                                    theme.colors.accent.opacity(0.65),
                                ),
                                redesmyn_session_view_model::SessionMessageRole::Assistant => (
                                    "assistant",
                                    theme.colors.surface_elevated.opacity(0.4),
                                ),
                                redesmyn_session_view_model::SessionMessageRole::Tool => (
                                    "tool",
                                    theme.colors.surface_elevated.opacity(0.6),
                                ),
                            };

                            let markdown = self
                                .markdown_cache
                                .entry(item.session_event_id)
                                .or_insert_with(|| {
                                    Arc::new(parse_markdown(
                                        &msg.text,
                                        MarkdownParseOptions::default(),
                                    ))
                                })
                                .clone();

                            let mut bubble = div()
                                .id(bubble_id.clone())
                                .flex()
                                .flex_col()
                                .gap(theme.spacing.sm)
                                .px(theme.spacing.md)
                                .py(theme.spacing.md)
                                .rounded_md()
                                .bg(bg)
                                .child(
                                    div()
                                        .id((bubble_id.clone(), "meta"))
                                        .text_xs()
                                        .text_color(theme.colors.foreground_muted)
                                        .child(role_label),
                                )
                                .child(MarkdownView::new(
                                    (bubble_id.clone(), "markdown"),
                                    markdown,
                                ));

                            if let Some(artifact) = msg.full_text_artifact {
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
                                                    "Output truncated. Full output stored as artifact {artifact_id}."
                                                )),
                                        )
                                        .child(
                                            TextButton::new(
                                                (bubble_id.clone(), "view_full_output"),
                                                "View full output…",
                                            )
                                            .kind(ButtonKind::Ghost)
                                            .tooltip(
                                                "Viewer not implemented yet — copies artifact id.",
                                            )
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

                            list.child(bubble)
                        }
                        other => {
                            let label = match other {
                                redesmyn_session_view_model::SessionEventItemContent::ToolInvocation(
                                    tool,
                                ) => {
                                    format!("tool invocation: {} — {}", tool.tool_name, tool.input_preview)
                                }
                                redesmyn_session_view_model::SessionEventItemContent::ToolResult(
                                    tool,
                                ) => {
                                    format!("tool result: {} — {}", tool.tool_name, tool.output_preview)
                                }
                                _ => format!("{:?}", item.kind),
                            };

                            list.child(
                                div()
                                    .id(bubble_id)
                                    .px(theme.spacing.sm)
                                    .py(theme.spacing.sm)
                                    .rounded_sm()
                                    .bg(theme.colors.surface_elevated.opacity(0.4))
                                    .text_sm()
                                    .text_color(theme.colors.foreground)
                                    .child(label),
                            )
                        }
                    }
                }
            },
        );

        let body = ScrollArea::new(
            ("session_scroll", cx.entity_id()),
            self.scroll_handle.clone(),
        )
        .scrollbar_width(px(10.0))
        .child(
            feed_list.on_scroll_wheel(cx.listener(|this, _event, _window, cx| {
                if let Some(feed) = this.feed.as_mut() {
                    update_scroll_state(feed, &this.scroll_handle);
                    cx.notify();
                }
            })),
        );

        content = content.child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .min_h(px(0.0))
                .border_1()
                .border_color(theme.colors.border.opacity(0.4))
                .rounded_md()
                .child(body),
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
        let composer_can_send = self.client.is_some()
            && self.feed.is_some()
            && !composer_sending
            && !composer_draft.trim().is_empty();

        let send_button = TextButton::new(
            ("session_send_message", cx.entity_id()),
            if composer_sending {
                "Sending…"
            } else {
                "Send"
            },
        )
        .kind(ButtonKind::Primary)
        .disabled(!composer_can_send)
        .on_click({
            let view = view.clone();
            move |_, _, cx| {
                view.update(cx, |this, cx| {
                    this.send_message(AgentMessageConflictAction::Fail, cx)
                });
            }
        });

        content = content.child(
            div()
                .flex()
                .flex_row()
                .gap(theme.spacing.sm)
                .items_end()
                .w_full()
                .child(
                    div()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(self.composer_input.clone()),
                )
                .child(send_button),
        );

        div()
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .size_full()
            .child(content)
            .track_focus(&self.focus_handle(cx))
    }
}
