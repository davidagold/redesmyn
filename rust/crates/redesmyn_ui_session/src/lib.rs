//! Session viewer scaffolding for GPUI (Domain 7).

#![forbid(unsafe_code)]

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpui::{
    AnyElement, App, AsyncApp, ClickEvent, Context, ElementId, Entity, FocusHandle, Focusable,
    KeyBinding, ListOffset, ListState, Render, ScrollHandle, SharedString, Subscription, Task,
    WeakEntity, Window, div, list, px, relative,
};

use gpui::prelude::*;

use redesmyn_client_api::Client;
use redesmyn_ids::{ArtifactId, SessionEventId, SessionId, SubscriptionId, TaskId};
use redesmyn_markdown::{MarkdownDoc, MarkdownParseOptions, parse_markdown};
use redesmyn_protocol::client::{
    AgentKind, AgentMessageConflictAction, GetLatestTaskSessionRequest, GetSessionEventsRequest,
    GetSessionEventsResponse, ListSessionModelsRequest, ListSessionModelsResponse,
    ModelReasoningEffort, RequestPayload, RespondPermissionRequestRequest,
    RespondPermissionRequestResponse, ResponseResult, SendSessionMessageRequest,
    SendSessionMessageResponse, SessionEventCursor, SessionEventKindFilter, SessionModelOption,
    SessionModelSelection, SetSessionCodexApprovalPolicyRequest,
    SetSessionCodexApprovalPolicyResponse, SetSessionCodexSandboxPolicyRequest,
    SetSessionCodexSandboxPolicyResponse, SetSessionModelRequest, SetSessionModelResponse,
    StartAgentRequest, StartAgentResponse, SubscriptionEvent,
};
use redesmyn_protocol::session::{
    CodexApprovalPolicy, CodexSandboxPolicy, PermissionDecision, PermissionDecisionBy,
    PermissionRequest, SessionEventKind, ToolResult,
};
use redesmyn_protocol::ui_driver::UiComposerState;
use redesmyn_protocol::{ArtifactRef, ErrorCategory, ErrorEnvelope, SessionEvent, StorageHint};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;

use redesmyn_session_view_model::{SessionEventItemContent, SessionFeedState, SessionTimelineItem};
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, CascadingMenu, CascadingMenuId, CascadingMenuMetrics,
    CascadingMenuRowStyle, CascadingMenuState, CascadingMenuSurfaceStyle, Expandable, IconButton,
    MarkdownView, ScrollFade, ScrollbarStyle, StyledScrollbar, TextArea, TextButton, TextInput,
    TextInputEvent, cascading_menu_radio_indicator, cascading_menu_row, cascading_menu_row_value,
    cascading_menu_surface, cascading_select_menu_item, set_open_cascading_menu,
};
use redesmyn_ui::styles::ThemeMode;
use redesmyn_ui::utils::{
    ActionAvailabilityProbe, BoundedCache, UiActivityGuard, UserActionState, theme_for_window,
    ui_idle_tracker, ui_test_mode_animation_duration,
};

use serde_json::Value as JsonValue;

gpui::actions!(
    redesmyn_ui_session_shortcuts,
    [OpenSessionModelSelector, OpenSessionReasoningSelector]
);

pub fn bind_session_shortcut_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("alt-m", OpenSessionModelSelector, Some("SessionComposer")),
        KeyBinding::new(
            "alt-m",
            OpenSessionModelSelector,
            Some("SessionComposer > TextArea"),
        ),
        KeyBinding::new(
            "alt-r",
            OpenSessionReasoningSelector,
            Some("SessionComposer"),
        ),
        KeyBinding::new(
            "alt-r",
            OpenSessionReasoningSelector,
            Some("SessionComposer > TextArea"),
        ),
    ]);
}

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

fn resolve_artifact_path_from_storage_hint(
    hint: &StorageHint,
    artifact_store_root: Option<&Path>,
) -> Result<PathBuf, String> {
    match hint {
        StorageHint::LocalPath { local_path } => Ok(PathBuf::from(local_path)),
        StorageHint::BlobKey { blob_key } => {
            let root = artifact_store_root
                .ok_or_else(|| "artifact store root not configured".to_string())?;
            let artifact_id_str = blob_key
                .rsplit('/')
                .next()
                .ok_or_else(|| format!("invalid blob_key: {blob_key}"))?;
            let artifact_id = ArtifactId::from_str(artifact_id_str)
                .map_err(|_| format!("invalid artifact id in blob_key: {blob_key}"))?;
            Ok(root.join("artifacts").join(format!("{artifact_id}.bin")))
        }
        _ => Err("unsupported artifact storage hint".to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionSettingsCategory {
    Permissions,
    Sandbox,
}

impl SessionSettingsCategory {
    const ALL: [Self; 2] = [Self::Permissions, Self::Sandbox];

    fn label(self) -> &'static str {
        match self {
            Self::Permissions => "Permissions",
            Self::Sandbox => "Sandbox",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Permissions => 0,
            Self::Sandbox => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionSettingsMenuFocus {
    Primary,
    Secondary,
}

impl Default for SessionSettingsMenuFocus {
    fn default() -> Self {
        Self::Primary
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionSettingsSandboxOption {
    Default,
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

impl SessionSettingsSandboxOption {
    const ALL: [Self; 4] = [
        Self::Default,
        Self::ReadOnly,
        Self::WorkspaceWrite,
        Self::DangerFullAccess,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::ReadOnly => "Read-only",
            Self::WorkspaceWrite => "Workspace write",
            Self::DangerFullAccess => "Danger: full access",
        }
    }

    fn policy(self) -> Option<CodexSandboxPolicy> {
        match self {
            Self::Default => None,
            Self::ReadOnly => Some(CodexSandboxPolicy::ReadOnly),
            Self::WorkspaceWrite => Some(default_workspace_write_policy()),
            Self::DangerFullAccess => Some(CodexSandboxPolicy::DangerFullAccess),
        }
    }

    const fn is_selected(self, displayed: Option<&CodexSandboxPolicy>) -> bool {
        match (self, displayed) {
            (Self::Default, None) => true,
            (Self::ReadOnly, Some(CodexSandboxPolicy::ReadOnly)) => true,
            (Self::WorkspaceWrite, Some(CodexSandboxPolicy::WorkspaceWrite { .. })) => true,
            (Self::DangerFullAccess, Some(CodexSandboxPolicy::DangerFullAccess)) => true,
            _ => false,
        }
    }
}

fn default_workspace_write_policy() -> CodexSandboxPolicy {
    CodexSandboxPolicy::WorkspaceWrite {
        writable_roots: Vec::new(),
        network_access: false,
        exclude_tmpdir_env_var: false,
        exclude_slash_tmp: false,
    }
}

fn normalize_model_id(value: Option<String>) -> Option<String> {
    value.and_then(|model_id| {
        let trimmed = model_id.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn normalize_session_model_selection(selection: SessionModelSelection) -> SessionModelSelection {
    let reasoning_effort = selection
        .reasoning_effort
        .and_then(|effort| (!matches!(effort, ModelReasoningEffort::Unknown)).then_some(effort));
    SessionModelSelection {
        model_id: normalize_model_id(selection.model_id),
        reasoning_effort,
    }
}

fn model_reasoning_effort_label(effort: Option<ModelReasoningEffort>) -> &'static str {
    match effort {
        Some(ModelReasoningEffort::Minimal) => "Minimal",
        Some(ModelReasoningEffort::Low) => "Low",
        Some(ModelReasoningEffort::Medium) => "Medium",
        Some(ModelReasoningEffort::High) => "High",
        Some(ModelReasoningEffort::Xhigh) => "XHigh",
        Some(ModelReasoningEffort::Unknown) | None => "Default",
    }
}

#[derive(Debug, Clone, Copy)]
struct SessionSettingsApprovalOption {
    policy: Option<CodexApprovalPolicy>,
    label: &'static str,
    _tooltip: &'static str,
}

const SESSION_SETTINGS_APPROVAL_OPTIONS: [SessionSettingsApprovalOption; 5] = [
    SessionSettingsApprovalOption {
        policy: None,
        label: "Default",
        _tooltip: "Use the agent default approvals",
    },
    SessionSettingsApprovalOption {
        policy: Some(CodexApprovalPolicy::UnlessTrusted),
        label: "Unless trusted",
        _tooltip: "Ask unless the environment is trusted",
    },
    SessionSettingsApprovalOption {
        policy: Some(CodexApprovalPolicy::OnRequest),
        label: "On request",
        _tooltip: "Ask when the agent requests permission",
    },
    SessionSettingsApprovalOption {
        policy: Some(CodexApprovalPolicy::OnFailure),
        label: "On failure",
        _tooltip: "Only ask if a command fails (dangerous)",
    },
    SessionSettingsApprovalOption {
        policy: Some(CodexApprovalPolicy::Never),
        label: "Never",
        _tooltip: "Deny all requests",
    },
];

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

fn tool_summary_preview(value: &str, max_chars: usize) -> String {
    let mut first_line = value.lines().next().unwrap_or_default().trim();
    if first_line.is_empty() {
        first_line = value.trim();
    }
    let mut out = String::new();
    let mut count = 0usize;
    let mut last_was_space = false;

    for ch in first_line.chars() {
        if ch.is_whitespace() {
            if !last_was_space && !out.is_empty() {
                out.push(' ');
                count += 1;
                last_was_space = true;
            }
        } else {
            out.push(ch);
            count += 1;
            last_was_space = false;
        }

        if count >= max_chars {
            break;
        }
    }

    let is_multiline = value.lines().nth(1).is_some();
    let first_line_overflow = first_line.chars().count() > max_chars;
    if (is_multiline || first_line_overflow) && !out.is_empty() && !out.ends_with('…') {
        out.push('…');
    }

    out
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
    get_session_events_filtered(client, session_id, before, limit, Vec::new()).await
}

async fn get_session_events_filtered(
    client: &Client,
    session_id: SessionId,
    before: Option<SessionEventCursor>,
    limit: u32,
    kinds: Vec<SessionEventKindFilter>,
) -> Result<GetSessionEventsResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::GetSessionEvents(GetSessionEventsRequest {
            session_id,
            before,
            limit,
            kinds,
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

async fn fetch_latest_policy_events(
    client: &Client,
    session_id: SessionId,
) -> Result<Vec<SessionEvent>, ErrorEnvelope> {
    let approval = get_session_events_filtered(
        client,
        session_id,
        None,
        1,
        vec![SessionEventKindFilter::CodexApprovalPolicyChanged],
    )
    .await?;
    let sandbox = get_session_events_filtered(
        client,
        session_id,
        None,
        1,
        vec![SessionEventKindFilter::CodexSandboxPolicyChanged],
    )
    .await?;
    let permissions_mode = get_session_events_filtered(
        client,
        session_id,
        None,
        1,
        vec![SessionEventKindFilter::PermissionsModeChanged],
    )
    .await?;

    let mut events = Vec::new();
    if let Some(event) = approval.events.into_iter().next() {
        events.push(event);
    }
    if let Some(event) = sandbox.events.into_iter().next() {
        events.push(event);
    }
    if let Some(event) = permissions_mode.events.into_iter().next() {
        events.push(event);
    }

    Ok(events)
}

async fn get_latest_task_session(
    client: &Client,
    task_id: TaskId,
) -> Result<Option<SessionId>, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::GetLatestTaskSession(
            GetLatestTaskSessionRequest { task_id },
        ))
        .await?;

    match response {
        ResponseResult::GetLatestTaskSession(resp) => Ok(resp.session_id),
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

async fn set_session_codex_approval_policy(
    client: &Client,
    session_id: SessionId,
    approval_policy: Option<CodexApprovalPolicy>,
) -> Result<SetSessionCodexApprovalPolicyResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::SetSessionCodexApprovalPolicy(
            SetSessionCodexApprovalPolicyRequest {
                session_id,
                approval_policy,
            },
        ))
        .await?;

    match response {
        ResponseResult::SetSessionCodexApprovalPolicy(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

async fn set_session_codex_sandbox_policy(
    client: &Client,
    session_id: SessionId,
    sandbox_policy: Option<CodexSandboxPolicy>,
) -> Result<SetSessionCodexSandboxPolicyResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::SetSessionCodexSandboxPolicy(
            SetSessionCodexSandboxPolicyRequest {
                session_id,
                sandbox_policy,
            },
        ))
        .await?;

    match response {
        ResponseResult::SetSessionCodexSandboxPolicy(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

async fn list_session_models(
    client: &Client,
    session_id: SessionId,
) -> Result<ListSessionModelsResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::ListSessionModels(
            ListSessionModelsRequest { session_id },
        ))
        .await?;

    match response {
        ResponseResult::ListSessionModels(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

async fn set_session_model(
    client: &Client,
    session_id: SessionId,
    selection: SessionModelSelection,
) -> Result<SetSessionModelResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::SetSessionModel(SetSessionModelRequest {
            session_id,
            selection,
        }))
        .await?;

    match response {
        ResponseResult::SetSessionModel(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

async fn respond_permission_request(
    client: &Client,
    session_id: SessionId,
    request_id: String,
    decision: PermissionDecision,
) -> Result<RespondPermissionRequestResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::RespondPermissionRequest(
            RespondPermissionRequestRequest {
                session_id,
                request_id,
                decision,
            },
        ))
        .await?;

    match response {
        ResponseResult::RespondPermissionRequest(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

async fn start_task_agent(
    client: &Client,
    task_id: TaskId,
    on_conflict: AgentMessageConflictAction,
) -> Result<StartAgentResponse, ErrorEnvelope> {
    let response = client
        .request(RequestPayload::StartAgent(StartAgentRequest {
            task_id,
            agent_kind: AgentKind::Codex,
            initial_prompt: None,
            on_conflict,
        }))
        .await?;

    match response {
        ResponseResult::StartAgent(resp) => Ok(resp),
        ResponseResult::Error(err) => Err(err),
        other => Err(ErrorEnvelope::new(
            ErrorCategory::Internal,
            format!("Unexpected response: {other:?}"),
        )),
    }
}

const REASONING_SCROLL_STATE_CACHE_CAPACITY: usize = 128;
const TOOL_EVENT_SCROLL_HANDLE_CACHE_CAPACITY: usize = 256;

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
struct ExpandCollapseTransition {
    started_at: Instant,
    from: f32,
    to: f32,
    duration: Duration,
}

impl ExpandCollapseTransition {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FullTextMessageLoadState {
    Loading,
    Loaded,
    Failed,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskSessionOperation {
    LoadLatest,
    StartAgent,
}

#[derive(Debug, Clone, Default)]
pub struct TaskSessionBindingState {
    pub task_id: Option<TaskId>,
    pub in_flight: bool,
    pub error: Option<SharedString>,
    pub session_id: Option<SessionId>,
    pub operation: Option<TaskSessionOperation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PermissionDecisionState {
    decision: PermissionDecision,
    decided_by: PermissionDecisionBy,
}

#[derive(Debug, Clone)]
pub enum SessionViewEvent {
    TaskBindingStateChanged(TaskSessionBindingState),
}

pub struct SessionView {
    focus_handle: FocusHandle,
    timeline_list_state: ListState,
    timeline_items: Rc<Vec<SessionTimelineItem>>,
    timeline_follow_bottom: Rc<Cell<bool>>,
    timeline_scroll_to_bottom_pending: bool,
    timeline_scrollbar_hidden: bool,
    timeline_last_scroll_offset: gpui::Pixels,
    timeline_scroll_handler_installed: bool,
    timeline_viewport_width: Option<gpui::Pixels>,
    timeline_list_reset_generation: u64,
    timeline_list_reset_task: Option<Task<()>>,
    timeline_list_reset_pending_scroll_top: Option<ListOffset>,
    timeline_autoload_scheduled: bool,
    reasoning_scroll_states: Rc<RefCell<BoundedCache<String, ReasoningScrollState>>>,
    tool_event_scroll_handles: Rc<RefCell<BoundedCache<SessionEventId, ScrollHandle>>>,
    reasoning_shimmer_phase: u8,
    reasoning_shimmer_task: Option<Task<()>>,
    show_debug_controls: bool,
    session_id_input: Entity<TextInput>,
    composer_input: Entity<TextArea>,
    pending_focus_composer: bool,
    feed: Option<SessionFeedState>,
    collapsed_reasoning: HashSet<String>,
    seen_reasoning_keys: HashSet<String>,
    reasoning_transitions: Rc<HashMap<String, ExpandCollapseTransition>>,
    reasoning_transition_guards: HashMap<String, UiActivityGuard>,
    expanded_tool_events: HashSet<SessionEventId>,
    tool_event_transitions: Rc<HashMap<SessionEventId, ExpandCollapseTransition>>,
    tool_event_transition_guards: HashMap<SessionEventId, UiActivityGuard>,
    expanded_tool_groups: HashSet<SessionEventId>,
    tool_group_transitions: Rc<HashMap<SessionEventId, ExpandCollapseTransition>>,
    tool_group_transition_guards: HashMap<SessionEventId, UiActivityGuard>,
    exec_command_result_by_invocation: Rc<HashMap<SessionEventId, (SessionEventId, ToolResult)>>,
    grouped_exec_command_result_event_ids: Rc<HashSet<SessionEventId>>,
    tool_event_groups: Rc<HashMap<SessionEventId, ToolEventGroup>>,
    tool_event_group_membership: Rc<HashMap<SessionEventId, ToolEventGroupMembership>>,
    markdown_cache: Rc<RefCell<HashMap<SessionEventId, Arc<MarkdownDoc>>>>,
    full_text_message_states: Rc<RefCell<HashMap<SessionEventId, FullTextMessageLoadState>>>,
    artifact_store_root: Option<PathBuf>,
    client: Option<Client>,
    _client_task: Option<Task<()>>,
    subscription_task: Option<Task<()>>,
    subscription_id: Option<SubscriptionId>,
    load_task: Option<Task<()>>,
    load_older_task: Option<Task<()>>,
    send_task: Option<Task<()>>,
    pending_codex_approval_policy: Option<Option<CodexApprovalPolicy>>,
    codex_approval_policy_action: UserActionState,
    set_codex_approval_policy_task: Option<Task<()>>,
    codex_approval_policy_timeout_task: Option<Task<()>>,
    pending_codex_sandbox_policy: Option<Option<CodexSandboxPolicy>>,
    codex_sandbox_policy_action: UserActionState,
    set_codex_sandbox_policy_task: Option<Task<()>>,
    codex_sandbox_policy_timeout_task: Option<Task<()>>,
    session_model_options: Vec<SessionModelOption>,
    session_model_selection: SessionModelSelection,
    pending_session_model_selection: Option<SessionModelSelection>,
    session_model_action: UserActionState,
    set_session_model_task: Option<Task<()>>,
    model_fetch_generation: u64,
    model_fetch_in_flight: bool,
    model_fetch_error: Option<SharedString>,
    model_fetch_task: Option<Task<()>>,
    session_settings_open: bool,
    session_settings_hovered: Option<SessionSettingsCategory>,
    session_settings_focus: SessionSettingsMenuFocus,
    session_settings_submenu_index: usize,
    session_model_menu_index: usize,
    session_reasoning_menu_index: usize,
    session_model_shortcut_availability: ActionAvailabilityProbe,
    session_reasoning_shortcut_availability: ActionAvailabilityProbe,
    policies_fetch_generation: u64,
    policies_fetch_in_flight: bool,
    policies_fetch_error: Option<SharedString>,
    policies_fetch_task: Option<Task<()>>,
    permission_request_ids: Rc<HashSet<String>>,
    permission_decisions_by_request_id: Rc<HashMap<String, PermissionDecisionState>>,
    expanded_permission_requests: HashSet<String>,
    permission_request_actions: HashMap<String, UserActionState>,
    respond_permission_request_tasks: HashMap<String, Task<()>>,
    task_binding_task_id: Option<TaskId>,
    task_binding_action: UserActionState,
    task_binding_generation: u64,
    task_binding_task: Option<Task<()>>,
    task_operation: Option<TaskSessionOperation>,
    start_agent_action: UserActionState,
    start_agent_generation: u64,
    start_agent_task: Option<Task<()>>,
    error: Option<SharedString>,
    _subscriptions: Vec<Subscription>,
}

impl Focusable for SessionView {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl gpui::EventEmitter<SessionViewEvent> for SessionView {}

impl SessionView {
    pub fn new(
        control_plane_client: Option<ClientInProcEndpoint>,
        initial_session_id: Option<SessionId>,
        artifact_store_root: Option<PathBuf>,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();
        let timeline_list_state =
            ListState::new(1, gpui::ListAlignment::Top, px(400.0)).measure_all();
        let show_debug_controls = std::env::var("REDESMYN_SESSION_VIEWER_DEBUG_CONTROLS")
            .ok()
            .is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true"));
        let session_id_input = cx.new(|cx| TextInput::new(cx).placeholder("Session id…"));
        let composer_input = cx.new(|cx| {
            TextArea::new(cx)
                .placeholder("Message…")
                .min_rows(1)
                .reserved_bottom(px(44.0))
        });

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
            timeline_scroll_to_bottom_pending: false,
            timeline_scrollbar_hidden: false,
            timeline_last_scroll_offset: px(0.0),
            timeline_scroll_handler_installed: false,
            timeline_viewport_width: None,
            timeline_list_reset_generation: 0,
            timeline_list_reset_task: None,
            timeline_list_reset_pending_scroll_top: None,
            timeline_autoload_scheduled: false,
            reasoning_scroll_states: Rc::new(RefCell::new(BoundedCache::new(
                REASONING_SCROLL_STATE_CACHE_CAPACITY,
            ))),
            tool_event_scroll_handles: Rc::new(RefCell::new(BoundedCache::new(
                TOOL_EVENT_SCROLL_HANDLE_CACHE_CAPACITY,
            ))),
            reasoning_shimmer_phase: 0,
            reasoning_shimmer_task: None,
            show_debug_controls,
            session_id_input,
            composer_input,
            pending_focus_composer: false,
            feed: None,
            collapsed_reasoning: HashSet::new(),
            seen_reasoning_keys: HashSet::new(),
            reasoning_transitions: Rc::new(HashMap::new()),
            reasoning_transition_guards: HashMap::new(),
            expanded_tool_events: HashSet::new(),
            tool_event_transitions: Rc::new(HashMap::new()),
            tool_event_transition_guards: HashMap::new(),
            expanded_tool_groups: HashSet::new(),
            tool_group_transitions: Rc::new(HashMap::new()),
            tool_group_transition_guards: HashMap::new(),
            exec_command_result_by_invocation: Rc::new(HashMap::new()),
            grouped_exec_command_result_event_ids: Rc::new(HashSet::new()),
            tool_event_groups: Rc::new(HashMap::new()),
            tool_event_group_membership: Rc::new(HashMap::new()),
            markdown_cache: Rc::new(RefCell::new(HashMap::new())),
            full_text_message_states: Rc::new(RefCell::new(HashMap::new())),
            artifact_store_root,
            client,
            _client_task: client_task,
            subscription_task: None,
            subscription_id: None,
            load_task: None,
            load_older_task: None,
            send_task: None,
            pending_codex_approval_policy: None,
            codex_approval_policy_action: UserActionState::default(),
            set_codex_approval_policy_task: None,
            codex_approval_policy_timeout_task: None,
            pending_codex_sandbox_policy: None,
            codex_sandbox_policy_action: UserActionState::default(),
            set_codex_sandbox_policy_task: None,
            codex_sandbox_policy_timeout_task: None,
            session_model_options: Vec::new(),
            session_model_selection: SessionModelSelection {
                model_id: None,
                reasoning_effort: None,
            },
            pending_session_model_selection: None,
            session_model_action: UserActionState::default(),
            set_session_model_task: None,
            model_fetch_generation: 0,
            model_fetch_in_flight: false,
            model_fetch_error: None,
            model_fetch_task: None,
            session_settings_open: false,
            session_settings_hovered: None,
            session_settings_focus: SessionSettingsMenuFocus::default(),
            session_settings_submenu_index: 0,
            session_model_menu_index: 0,
            session_reasoning_menu_index: 0,
            session_model_shortcut_availability: ActionAvailabilityProbe::new(),
            session_reasoning_shortcut_availability: ActionAvailabilityProbe::new(),
            policies_fetch_generation: 0,
            policies_fetch_in_flight: false,
            policies_fetch_error: None,
            policies_fetch_task: None,
            permission_request_ids: Rc::new(HashSet::new()),
            permission_decisions_by_request_id: Rc::new(HashMap::new()),
            expanded_permission_requests: HashSet::new(),
            permission_request_actions: HashMap::new(),
            respond_permission_request_tasks: HashMap::new(),
            task_binding_task_id: None,
            task_binding_action: UserActionState::default(),
            task_binding_generation: 0,
            task_binding_task: None,
            task_operation: None,
            start_agent_action: UserActionState::default(),
            start_agent_generation: 0,
            start_agent_task: None,
            error: None,
            _subscriptions: subscriptions,
        };

        let initial = this.session_id_input.read(cx).text().clone();
        if !initial.as_ref().trim().is_empty() {
            this.load_session_id(initial.as_ref(), cx);
        }

        this
    }

    #[must_use]
    pub fn task_binding_state(&self) -> TaskSessionBindingState {
        let session_id = self.feed.as_ref().map(|feed| feed.session_id);
        let in_flight = self.task_binding_action.in_flight || self.start_agent_action.in_flight;
        let error = self
            .start_agent_action
            .error
            .clone()
            .or_else(|| self.task_binding_action.error.clone());
        let operation = if self.start_agent_action.in_flight
            || self.start_agent_action.error.is_some()
        {
            Some(TaskSessionOperation::StartAgent)
        } else if self.task_binding_action.in_flight || self.task_binding_action.error.is_some() {
            Some(TaskSessionOperation::LoadLatest)
        } else {
            self.task_operation
        };

        TaskSessionBindingState {
            task_id: self.task_binding_task_id,
            in_flight,
            error,
            session_id,
            operation,
        }
    }

    #[must_use]
    pub fn client(&self) -> Option<Client> {
        self.client.clone()
    }

    #[must_use]
    pub fn session_model_options_snapshot(&self) -> Vec<SessionModelOption> {
        self.session_model_options.clone()
    }

    pub fn bind_latest_task_session(&mut self, task_id: Option<TaskId>, cx: &mut Context<Self>) {
        if self.task_binding_task_id == task_id {
            return;
        }

        self.task_binding_generation = self.task_binding_generation.wrapping_add(1);
        let generation = self.task_binding_generation;
        self.task_binding_task_id = task_id;
        self.task_binding_action = UserActionState::default();
        self.task_binding_task = None;
        self.start_agent_action = UserActionState::default();
        self.start_agent_task = None;
        self.task_operation = Some(TaskSessionOperation::LoadLatest);

        self.set_session_id(None, cx);

        let Some(task_id) = task_id else {
            cx.notify();
            return;
        };

        let Some(client) = self.client.clone() else {
            self.task_binding_action
                .fail("Control plane client is unavailable.");
            cx.emit(SessionViewEvent::TaskBindingStateChanged(
                self.task_binding_state(),
            ));
            cx.notify();
            return;
        };

        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.session_view.bind_latest_task_session",
            task_id = %task_id
        );
        let _guard = span.enter();

        self.task_binding_action.start();
        cx.emit(SessionViewEvent::TaskBindingStateChanged(
            self.task_binding_state(),
        ));
        cx.notify();

        let view = cx.entity();
        self.task_binding_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let client = client.clone();
            let cx = cx.clone();
            async move {
                let result = get_latest_task_session(&client, task_id).await;
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.on_task_binding_loaded(generation, task_id, result, cx)
                    })
                });
            }
        }));
    }

    pub fn refresh_latest_task_session(&mut self, cx: &mut Context<Self>) {
        let Some(task_id) = self.task_binding_task_id else {
            return;
        };

        if self.task_binding_action.in_flight {
            return;
        }

        self.task_binding_generation = self.task_binding_generation.wrapping_add(1);
        let generation = self.task_binding_generation;
        self.task_binding_action = UserActionState::default();
        self.task_binding_task = None;
        self.start_agent_action = UserActionState::default();
        self.start_agent_task = None;
        self.task_operation = Some(TaskSessionOperation::LoadLatest);

        self.set_session_id(None, cx);

        let Some(client) = self.client.clone() else {
            self.task_binding_action
                .fail("Control plane client is unavailable.");
            cx.emit(SessionViewEvent::TaskBindingStateChanged(
                self.task_binding_state(),
            ));
            cx.notify();
            return;
        };

        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.session_view.refresh_latest_task_session",
            task_id = %task_id
        );
        let _guard = span.enter();

        self.task_binding_action.start();
        cx.emit(SessionViewEvent::TaskBindingStateChanged(
            self.task_binding_state(),
        ));
        cx.notify();

        let view = cx.entity();
        self.task_binding_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let client = client.clone();
            let cx = cx.clone();
            async move {
                let result = get_latest_task_session(&client, task_id).await;
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.on_task_binding_loaded(generation, task_id, result, cx)
                    })
                });
            }
        }));
    }

    fn on_task_binding_loaded(
        &mut self,
        generation: u64,
        task_id: TaskId,
        result: Result<Option<SessionId>, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        if generation != self.task_binding_generation || self.task_binding_task_id != Some(task_id)
        {
            return;
        }

        self.task_binding_task = None;

        match result {
            Ok(session_id) => {
                self.task_binding_action.succeed();
                self.task_binding_action.clear_error();
                self.set_session_id(session_id, cx);
            }
            Err(err) => {
                redesmyn_logging::tracing::warn!(
                    task_id = %task_id,
                    error = %err.message,
                    "failed to load latest task session"
                );
                self.task_binding_action.fail(err.message);
                self.set_session_id(None, cx);
            }
        }

        cx.emit(SessionViewEvent::TaskBindingStateChanged(
            self.task_binding_state(),
        ));
        cx.notify();
    }

    pub fn start_agent_for_task(&mut self, task_id: TaskId, cx: &mut Context<Self>) {
        if self.start_agent_action.in_flight {
            return;
        }

        if self.task_binding_task_id != Some(task_id) {
            self.task_binding_task_id = Some(task_id);
        }

        let Some(client) = self.client.clone() else {
            self.start_agent_action
                .fail("Control plane client is unavailable.");
            cx.emit(SessionViewEvent::TaskBindingStateChanged(
                self.task_binding_state(),
            ));
            cx.notify();
            return;
        };

        self.start_agent_generation = self.start_agent_generation.wrapping_add(1);
        let generation = self.start_agent_generation;
        self.task_operation = Some(TaskSessionOperation::StartAgent);
        self.start_agent_action.start();
        cx.emit(SessionViewEvent::TaskBindingStateChanged(
            self.task_binding_state(),
        ));
        cx.notify();

        let view = cx.entity();
        self.start_agent_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let client = client.clone();
            let cx = cx.clone();
            async move {
                let result =
                    start_task_agent(&client, task_id, AgentMessageConflictAction::Fail).await;
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.on_start_agent_completed(generation, task_id, result, cx)
                    })
                });
            }
        }));
    }

    fn on_start_agent_completed(
        &mut self,
        generation: u64,
        task_id: TaskId,
        result: Result<StartAgentResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        if generation != self.start_agent_generation || self.task_binding_task_id != Some(task_id) {
            return;
        }

        self.start_agent_task = None;
        match result {
            Ok(resp) => {
                self.start_agent_action.succeed();
                self.start_agent_action.clear_error();
                self.set_session_id(Some(resp.session_id), cx);
            }
            Err(err) => {
                redesmyn_logging::tracing::warn!(
                    task_id = %task_id,
                    error = %err.message,
                    "failed to start task agent"
                );
                self.start_agent_action.fail(err.message);
            }
        }

        cx.emit(SessionViewEvent::TaskBindingStateChanged(
            self.task_binding_state(),
        ));
        cx.notify();
    }

    pub fn should_defer_pan_to_scroll_view(
        &self,
        window_point: gpui::Point<gpui::Pixels>,
        delta: gpui::Point<gpui::Pixels>,
    ) -> bool {
        let bounds = self.timeline_list_state.viewport_bounds();
        if !bounds.contains(&window_point) {
            return false;
        }

        let max_y = self.timeline_list_state.max_offset_for_scrollbar().height;
        if max_y == px(0.0) {
            return false;
        }

        let delta_y = if delta.y == px(0.0) { delta.x } else { delta.y };
        if delta_y == px(0.0) {
            return false;
        }

        let offset_y = self.timeline_list_state.scroll_px_offset_for_scrollbar().y;
        let epsilon = px(1.0);

        if delta_y > px(0.0) {
            // Scrolling "up": allow pan only when already at the top edge.
            offset_y < -epsilon
        } else {
            // Scrolling "down": allow pan only when already at the bottom edge.
            offset_y > (-max_y + epsilon)
        }
    }

    pub fn set_session_id(&mut self, session_id: Option<SessionId>, cx: &mut Context<Self>) {
        let Some(session_id) = session_id else {
            if let Some(client) = self.client.clone()
                && let Some(subscription_id) = self.subscription_id.take()
            {
                cx.spawn(move |_: WeakEntity<Self>, _cx: &mut AsyncApp| async move {
                    let _ = client.unsubscribe(subscription_id).await;
                })
                .detach();
            }

            self.subscription_id = None;
            self.subscription_task = None;
            self.load_task = None;
            self.load_older_task = None;
            self.error = None;
            self.feed = None;
            self.pending_codex_approval_policy = None;
            self.codex_approval_policy_action = UserActionState::default();
            self.set_codex_approval_policy_task = None;
            self.codex_approval_policy_timeout_task = None;
            self.pending_codex_sandbox_policy = None;
            self.codex_sandbox_policy_action = UserActionState::default();
            self.set_codex_sandbox_policy_task = None;
            self.codex_sandbox_policy_timeout_task = None;
            self.session_settings_open = false;
            self.session_settings_hovered = None;
            self.session_settings_focus = SessionSettingsMenuFocus::Primary;
            self.session_settings_submenu_index = 0;
            self.permission_request_ids = Rc::new(HashSet::new());
            self.permission_decisions_by_request_id = Rc::new(HashMap::new());
            self.expanded_permission_requests.clear();
            self.permission_request_actions = HashMap::new();
            self.respond_permission_request_tasks = HashMap::new();
            self.timeline_follow_bottom.set(false);
            self.timeline_last_scroll_offset = px(0.0);
            self.collapsed_reasoning.clear();
            self.seen_reasoning_keys.clear();
            self.reasoning_transitions = Rc::new(HashMap::new());
            self.reasoning_transition_guards.clear();
            self.expanded_tool_events.clear();
            self.tool_event_transitions = Rc::new(HashMap::new());
            self.tool_event_transition_guards.clear();
            self.expanded_tool_groups.clear();
            self.tool_group_transitions = Rc::new(HashMap::new());
            self.tool_group_transition_guards.clear();
            self.reasoning_scroll_states.borrow_mut().clear();
            self.tool_event_scroll_handles.borrow_mut().clear();
            self.exec_command_result_by_invocation = Rc::new(HashMap::new());
            self.grouped_exec_command_result_event_ids = Rc::new(HashSet::new());
            self.tool_event_groups = Rc::new(HashMap::new());
            self.tool_event_group_membership = Rc::new(HashMap::new());
            self.markdown_cache.borrow_mut().clear();
            self.full_text_message_states.borrow_mut().clear();
            self.set_timeline_items(Vec::new());
            self.timeline_list_reset_generation = 0;
            self.timeline_list_reset_task = None;
            self.timeline_list_reset_pending_scroll_top = None;
            self.timeline_autoload_scheduled = false;
            self.timeline_scrollbar_hidden = false;
            self.reasoning_shimmer_phase = 0;
            self.reasoning_shimmer_task = None;
            self.pending_focus_composer = false;
            self.composer_input
                .update(cx, |input, cx| input.set_text("", cx));
            self.session_id_input
                .update(cx, |input, cx| input.set_text("", cx));
            cx.emit(SessionViewEvent::TaskBindingStateChanged(
                self.task_binding_state(),
            ));
            cx.notify();
            return;
        };

        self.session_id_input
            .update(cx, |input, cx| input.set_text(session_id.to_string(), cx));
        self.load_session_id(&session_id.to_string(), cx);
        cx.emit(SessionViewEvent::TaskBindingStateChanged(
            self.task_binding_state(),
        ));
    }

    pub fn request_focus_composer(&mut self, cx: &mut Context<Self>) {
        self.pending_focus_composer = true;
        cx.notify();
    }

    pub fn set_settings_menu_open(&mut self, open: bool, cx: &mut Context<Self>) {
        if open {
            self.open_session_settings_menu(cx);
        } else {
            self.close_session_settings_menu(cx);
        }
    }

    fn open_session_settings_menu(&mut self, cx: &mut Context<Self>) {
        if self.session_settings_open {
            return;
        }

        let already_open = cx
            .try_global::<CascadingMenuState>()
            .map(|state| state.open_menu() == Some(CascadingMenuId::SessionSettings))
            .unwrap_or(false);
        if !already_open {
            set_open_cascading_menu(Some(CascadingMenuId::SessionSettings), cx);
        }

        self.session_settings_open = true;
        self.session_settings_hovered = None;
        self.session_settings_focus = SessionSettingsMenuFocus::Primary;
        self.session_settings_submenu_index = 0;
        cx.notify();
    }

    fn toggle_session_settings_menu(&mut self, cx: &mut Context<Self>) {
        if self.session_settings_open {
            self.close_session_settings_menu(cx);
        } else {
            self.open_session_settings_menu(cx);
        }
    }

    pub fn send_settings_menu_key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        self.handle_session_settings_key(key, cx)
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
        self.pending_codex_approval_policy = None;
        self.codex_approval_policy_action = UserActionState::default();
        self.set_codex_approval_policy_task = None;
        self.codex_approval_policy_timeout_task = None;
        self.pending_codex_sandbox_policy = None;
        self.codex_sandbox_policy_action = UserActionState::default();
        self.set_codex_sandbox_policy_task = None;
        self.codex_sandbox_policy_timeout_task = None;
        self.session_model_options.clear();
        self.session_model_selection = SessionModelSelection {
            model_id: None,
            reasoning_effort: None,
        };
        self.pending_session_model_selection = None;
        self.session_model_action = UserActionState::default();
        self.set_session_model_task = None;
        self.model_fetch_generation = self.model_fetch_generation.wrapping_add(1);
        self.model_fetch_in_flight = true;
        self.model_fetch_error = None;
        self.model_fetch_task = None;
        self.session_settings_open = false;
        self.session_settings_hovered = None;
        self.session_settings_focus = SessionSettingsMenuFocus::Primary;
        self.session_settings_submenu_index = 0;
        self.session_model_menu_index = 0;
        self.session_reasoning_menu_index = 0;
        self.policies_fetch_generation = self.policies_fetch_generation.wrapping_add(1);
        self.policies_fetch_in_flight = true;
        self.policies_fetch_error = None;
        self.policies_fetch_task = None;
        self.permission_request_ids = Rc::new(HashSet::new());
        self.permission_decisions_by_request_id = Rc::new(HashMap::new());
        self.expanded_permission_requests.clear();
        self.permission_request_actions = HashMap::new();
        self.respond_permission_request_tasks = HashMap::new();
        self.collapsed_reasoning.clear();
        self.seen_reasoning_keys.clear();
        self.reasoning_transitions = Rc::new(HashMap::new());
        self.reasoning_transition_guards.clear();
        self.expanded_tool_events.clear();
        self.tool_event_transitions = Rc::new(HashMap::new());
        self.tool_event_transition_guards.clear();
        self.expanded_tool_groups.clear();
        self.tool_group_transitions = Rc::new(HashMap::new());
        self.tool_group_transition_guards.clear();
        self.reasoning_scroll_states.borrow_mut().clear();
        self.tool_event_scroll_handles.borrow_mut().clear();
        self.exec_command_result_by_invocation = Rc::new(HashMap::new());
        self.grouped_exec_command_result_event_ids = Rc::new(HashSet::new());
        self.tool_event_groups = Rc::new(HashMap::new());
        self.tool_event_group_membership = Rc::new(HashMap::new());
        self.markdown_cache.borrow_mut().clear();
        self.full_text_message_states.borrow_mut().clear();
        self.feed = Some(SessionFeedState::new(session_id));
        self.composer_input
            .update(cx, |input, cx| input.set_text("", cx));
        self.set_timeline_items(Vec::new());
        cx.notify();

        let view = cx.entity();
        let policies_view = view.clone();
        let policies_generation = self.policies_fetch_generation;
        let policy_client = client.clone();
        self.policies_fetch_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let result = fetch_latest_policy_events(&policy_client, session_id).await;
                let _ = cx.update(|cx| {
                    policies_view.update(cx, |this, cx| {
                        this.on_policies_fetched(policies_generation, session_id, result, cx);
                    })
                });
            }
        }));

        let model_view = view.clone();
        let model_generation = self.model_fetch_generation;
        let model_client = client.clone();
        self.model_fetch_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let result = list_session_models(&model_client, session_id).await;
                let _ = cx.update(|cx| {
                    model_view.update(cx, |this, cx| {
                        this.on_models_fetched(model_generation, session_id, result, cx);
                    })
                });
            }
        }));

        let history_client = client.clone();
        self.load_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let response = get_session_events(&history_client, session_id, None, 50).await;
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
        self.timeline_scroll_to_bottom_pending = true;
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

    fn set_codex_approval_policy(
        &mut self,
        approval_policy: Option<CodexApprovalPolicy>,
        cx: &mut Context<Self>,
    ) {
        let Some(client) = self.client.clone() else {
            self.codex_approval_policy_action
                .fail("Control plane client is unavailable.");
            cx.notify();
            return;
        };

        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        if self.codex_approval_policy_action.in_flight {
            return;
        }

        let closed_menu = self.session_settings_open;
        self.session_settings_open = false;
        self.session_settings_hovered = None;
        self.session_settings_focus = SessionSettingsMenuFocus::Primary;
        self.session_settings_submenu_index = 0;
        self.session_settings_focus = SessionSettingsMenuFocus::Primary;
        self.session_settings_submenu_index = 0;

        let displayed = self
            .pending_codex_approval_policy
            .unwrap_or(feed.codex_approval_policy);
        if displayed == approval_policy {
            if closed_menu {
                cx.notify();
            }
            return;
        }

        self.codex_approval_policy_action.start();
        self.pending_codex_approval_policy = Some(approval_policy);
        self.codex_approval_policy_timeout_task = None;
        cx.notify();

        let session_id = feed.session_id;
        let view = cx.entity();
        let timeout_view = view.clone();
        self.set_codex_approval_policy_task =
            Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    let result =
                        set_session_codex_approval_policy(&client, session_id, approval_policy)
                            .await;
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            this.on_set_codex_approval_policy_completed(approval_policy, result, cx)
                        })
                    });
                }
            }));

        let expected_policy = approval_policy;
        self.codex_approval_policy_timeout_task =
            Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    cx.background_executor().timer(Duration::from_secs(8)).await;
                    let _ = cx.update(|cx| {
                        timeout_view.update(cx, |this, cx| {
                            this.on_codex_approval_policy_timeout(expected_policy, cx);
                        })
                    });
                }
            }));
    }

    fn on_set_codex_approval_policy_completed(
        &mut self,
        approval_policy: Option<CodexApprovalPolicy>,
        result: Result<SetSessionCodexApprovalPolicyResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        self.set_codex_approval_policy_task = None;

        match result {
            Ok(_resp) => {
                // The control-plane request only acknowledges command issuance; wait until we see
                // the durable `CodexApprovalPolicyChanged` session event to mark it applied.
                let _ = approval_policy;
            }
            Err(err) => {
                self.codex_approval_policy_action.fail(err.message);
                self.pending_codex_approval_policy = None;
                self.codex_approval_policy_timeout_task = None;
            }
        }

        cx.notify();
    }

    fn on_codex_approval_policy_timeout(
        &mut self,
        expected_policy: Option<CodexApprovalPolicy>,
        cx: &mut Context<Self>,
    ) {
        if !self.codex_approval_policy_action.in_flight {
            return;
        }

        if self.pending_codex_approval_policy != Some(expected_policy) {
            return;
        }

        self.codex_approval_policy_action
            .fail("Timed out waiting for approvals to apply.");
        self.pending_codex_approval_policy = None;
        self.codex_approval_policy_timeout_task = None;
        cx.notify();
    }

    fn set_codex_sandbox_policy(
        &mut self,
        sandbox_policy: Option<CodexSandboxPolicy>,
        cx: &mut Context<Self>,
    ) {
        let Some(client) = self.client.clone() else {
            self.codex_sandbox_policy_action
                .fail("Control plane client is unavailable.");
            cx.notify();
            return;
        };

        let Some(feed) = self.feed.as_mut() else {
            return;
        };

        if self.codex_sandbox_policy_action.in_flight {
            return;
        }

        let closed_menu = self.session_settings_open;
        self.session_settings_open = false;
        self.session_settings_hovered = None;

        let displayed = self
            .pending_codex_sandbox_policy
            .clone()
            .unwrap_or(feed.codex_sandbox_policy.clone());
        if displayed == sandbox_policy {
            if closed_menu {
                cx.notify();
            }
            return;
        }

        self.codex_sandbox_policy_action.start();
        self.pending_codex_sandbox_policy = Some(sandbox_policy.clone());
        self.codex_sandbox_policy_timeout_task = None;
        cx.notify();

        let session_id = feed.session_id;
        let expected_policy = sandbox_policy.clone();
        let sandbox_policy_for_command = sandbox_policy.clone();
        let view = cx.entity();
        let timeout_view = view.clone();
        self.set_codex_sandbox_policy_task =
            Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    let result = set_session_codex_sandbox_policy(
                        &client,
                        session_id,
                        sandbox_policy_for_command.clone(),
                    )
                    .await;
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            this.on_set_codex_sandbox_policy_completed(
                                sandbox_policy_for_command.clone(),
                                result,
                                cx,
                            )
                        })
                    });
                }
            }));

        self.codex_sandbox_policy_timeout_task =
            Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    cx.background_executor().timer(Duration::from_secs(8)).await;
                    let _ = cx.update(|cx| {
                        timeout_view.update(cx, |this, cx| {
                            this.on_codex_sandbox_policy_timeout(expected_policy.clone(), cx);
                        })
                    });
                }
            }));
    }

    fn on_set_codex_sandbox_policy_completed(
        &mut self,
        sandbox_policy: Option<CodexSandboxPolicy>,
        result: Result<SetSessionCodexSandboxPolicyResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        self.set_codex_sandbox_policy_task = None;

        match result {
            Ok(_resp) => {
                // The control-plane request only acknowledges command issuance; wait until we see
                // the durable `CodexSandboxPolicyChanged` session event to mark it applied.
                let _ = sandbox_policy;
            }
            Err(err) => {
                self.codex_sandbox_policy_action.fail(err.message);
                self.pending_codex_sandbox_policy = None;
                self.codex_sandbox_policy_timeout_task = None;
            }
        }

        cx.notify();
    }

    fn on_codex_sandbox_policy_timeout(
        &mut self,
        expected_policy: Option<CodexSandboxPolicy>,
        cx: &mut Context<Self>,
    ) {
        if !self.codex_sandbox_policy_action.in_flight {
            return;
        }

        if self.pending_codex_sandbox_policy != Some(expected_policy.clone()) {
            return;
        }

        self.codex_sandbox_policy_action
            .fail("Timed out waiting for sandbox to apply.");
        self.pending_codex_sandbox_policy = None;
        self.codex_sandbox_policy_timeout_task = None;
        cx.notify();
    }

    fn on_models_fetched(
        &mut self,
        generation: u64,
        session_id: SessionId,
        result: Result<ListSessionModelsResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        if generation != self.model_fetch_generation {
            return;
        }

        self.model_fetch_task = None;
        self.model_fetch_in_flight = false;

        match result {
            Ok(resp) => {
                if self
                    .feed
                    .as_ref()
                    .is_some_and(|feed| feed.session_id == session_id)
                {
                    self.session_model_options = resp.options;
                    self.session_model_selection =
                        normalize_session_model_selection(resp.selection);
                    self.model_fetch_error = None;
                    self.session_model_action.clear_error();
                }
            }
            Err(err) => {
                self.model_fetch_error = Some(err.message.clone().into());
                if self.session_model_options.is_empty() {
                    self.session_model_action.fail(err.message);
                }
            }
        }

        cx.notify();
    }

    fn refresh_session_models(&mut self, session_id: SessionId, cx: &mut Context<Self>) {
        let Some(client) = self.client.clone() else {
            return;
        };

        self.model_fetch_generation = self.model_fetch_generation.wrapping_add(1);
        self.model_fetch_in_flight = true;
        self.model_fetch_error = None;
        self.model_fetch_task = None;
        let generation = self.model_fetch_generation;
        let view = cx.entity();

        self.model_fetch_task = Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let result = list_session_models(&client, session_id).await;
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.on_models_fetched(generation, session_id, result, cx);
                    })
                });
            }
        }));
    }

    fn set_session_model_selection(
        &mut self,
        selection: SessionModelSelection,
        close_menu: bool,
        cx: &mut Context<Self>,
    ) {
        let Some(client) = self.client.clone() else {
            self.session_model_action
                .fail("Control plane client is unavailable.");
            cx.notify();
            return;
        };

        let Some(feed) = self.feed.as_ref() else {
            return;
        };

        if self.session_model_action.in_flight {
            return;
        }

        if close_menu {
            let open_menu = cx
                .try_global::<CascadingMenuState>()
                .map(|state| state.open_menu())
                .unwrap_or(None);
            if matches!(
                open_menu,
                Some(CascadingMenuId::SessionModel | CascadingMenuId::SessionReasoning)
            ) {
                set_open_cascading_menu(None, cx);
            }
        }

        let desired = normalize_session_model_selection(selection);
        let displayed = normalize_session_model_selection(
            self.pending_session_model_selection
                .clone()
                .unwrap_or_else(|| self.session_model_selection.clone()),
        );
        if desired == displayed {
            return;
        }

        self.session_model_action.start();
        self.pending_session_model_selection = Some(desired.clone());
        cx.notify();

        let session_id = feed.session_id;
        let view = cx.entity();
        self.set_session_model_task =
            Some(cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    let result = set_session_model(&client, session_id, desired.clone()).await;
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            this.on_set_session_model_completed(desired.clone(), result, cx);
                        })
                    });
                }
            }));
    }

    fn on_set_session_model_completed(
        &mut self,
        requested: SessionModelSelection,
        result: Result<SetSessionModelResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        self.set_session_model_task = None;

        match result {
            Ok(_resp) => {
                self.session_model_action.succeed();
                self.pending_session_model_selection = None;
                self.session_model_selection = normalize_session_model_selection(requested.clone());
                if let Some(feed) = self.feed.as_ref() {
                    self.refresh_session_models(feed.session_id, cx);
                }
            }
            Err(err) => {
                self.session_model_action.fail(err.message);
                self.pending_session_model_selection = None;
            }
        }

        cx.notify();
    }

    fn displayed_session_model_selection(&self) -> SessionModelSelection {
        normalize_session_model_selection(
            self.pending_session_model_selection
                .clone()
                .unwrap_or_else(|| self.session_model_selection.clone()),
        )
    }

    fn selected_model_option_for_selection<'a>(
        &'a self,
        selection: &SessionModelSelection,
    ) -> Option<&'a SessionModelOption> {
        selection
            .model_id
            .as_ref()
            .and_then(|model_id| {
                self.session_model_options
                    .iter()
                    .find(|option| option.model_id == *model_id)
            })
            .or_else(|| {
                self.session_model_options
                    .iter()
                    .find(|option| option.is_default)
            })
    }

    fn reasoning_values_for_selection(
        &self,
        selection: &SessionModelSelection,
    ) -> Vec<Option<ModelReasoningEffort>> {
        let selected_model_option = self.selected_model_option_for_selection(selection);
        let mut supported_reasoning_efforts = Vec::new();
        if let Some(option) = selected_model_option {
            supported_reasoning_efforts = option.supported_reasoning_efforts.clone();
        } else {
            for option in &self.session_model_options {
                for effort in &option.supported_reasoning_efforts {
                    if !supported_reasoning_efforts.contains(effort) {
                        supported_reasoning_efforts.push(*effort);
                    }
                }
            }
        }

        let preferred_order = [
            ModelReasoningEffort::Minimal,
            ModelReasoningEffort::Low,
            ModelReasoningEffort::Medium,
            ModelReasoningEffort::High,
            ModelReasoningEffort::Xhigh,
        ];
        let mut values: Vec<Option<ModelReasoningEffort>> = vec![None];
        if supported_reasoning_efforts.is_empty() {
            values.extend(preferred_order.iter().copied().map(Some));
        } else {
            for effort in preferred_order {
                if supported_reasoning_efforts.contains(&effort) {
                    values.push(Some(effort));
                }
            }
        }
        values
    }

    fn set_session_model_menu_index_from_selection(&mut self) {
        let selection = self.displayed_session_model_selection();
        self.session_model_menu_index = selection
            .model_id
            .as_ref()
            .and_then(|model_id| {
                self.session_model_options
                    .iter()
                    .position(|option| option.model_id == *model_id)
                    .map(|idx| idx + 1)
            })
            .unwrap_or(0);
    }

    fn set_session_reasoning_menu_index_from_selection(&mut self) {
        let selection = self.displayed_session_model_selection();
        let values = self.reasoning_values_for_selection(&selection);
        self.session_reasoning_menu_index = values
            .iter()
            .position(|value| *value == selection.reasoning_effort)
            .unwrap_or(0);
    }

    fn apply_session_model_index(&mut self, index: usize, cx: &mut Context<Self>) {
        let selection = self.displayed_session_model_selection();
        if index == 0 {
            self.set_session_model_selection(
                SessionModelSelection {
                    model_id: None,
                    reasoning_effort: selection.reasoning_effort,
                },
                true,
                cx,
            );
            return;
        }

        let option_index = index.saturating_sub(1);
        let Some(option) = self.session_model_options.get(option_index) else {
            return;
        };
        let supported = &option.supported_reasoning_efforts;
        let reasoning_effort = selection
            .reasoning_effort
            .and_then(|value| supported.contains(&value).then_some(value));
        self.set_session_model_selection(
            SessionModelSelection {
                model_id: Some(option.model_id.clone()),
                reasoning_effort,
            },
            true,
            cx,
        );
    }

    fn apply_session_reasoning_index(&mut self, index: usize, cx: &mut Context<Self>) {
        let selection = self.displayed_session_model_selection();
        let values = self.reasoning_values_for_selection(&selection);
        let Some(reasoning_effort) = values.get(index).copied() else {
            return;
        };
        self.set_session_model_selection(
            SessionModelSelection {
                model_id: selection.model_id,
                reasoning_effort,
            },
            true,
            cx,
        );
    }

    fn handle_session_model_menu_key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        let open_menu = cx
            .try_global::<CascadingMenuState>()
            .map(|state| state.open_menu())
            .unwrap_or(None);
        if open_menu != Some(CascadingMenuId::SessionModel) {
            return false;
        }

        let len = self.session_model_options.len() + 1;
        if len == 0 {
            return false;
        }

        match key {
            "escape" => {
                set_open_cascading_menu(None, cx);
                cx.notify();
                true
            }
            "up" => {
                if self.session_model_menu_index > 0 {
                    self.session_model_menu_index -= 1;
                    cx.notify();
                }
                true
            }
            "down" => {
                let next = (self.session_model_menu_index + 1).min(len.saturating_sub(1));
                if next != self.session_model_menu_index {
                    self.session_model_menu_index = next;
                    cx.notify();
                }
                true
            }
            "enter" => {
                self.apply_session_model_index(self.session_model_menu_index, cx);
                true
            }
            _ => false,
        }
    }

    fn handle_session_reasoning_menu_key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        let open_menu = cx
            .try_global::<CascadingMenuState>()
            .map(|state| state.open_menu())
            .unwrap_or(None);
        if open_menu != Some(CascadingMenuId::SessionReasoning) {
            return false;
        }

        let selection = self.displayed_session_model_selection();
        let values = self.reasoning_values_for_selection(&selection);
        let len = values.len();
        if len == 0 {
            return false;
        }

        match key {
            "escape" => {
                set_open_cascading_menu(None, cx);
                cx.notify();
                true
            }
            "up" => {
                if self.session_reasoning_menu_index > 0 {
                    self.session_reasoning_menu_index -= 1;
                    cx.notify();
                }
                true
            }
            "down" => {
                let next = (self.session_reasoning_menu_index + 1).min(len.saturating_sub(1));
                if next != self.session_reasoning_menu_index {
                    self.session_reasoning_menu_index = next;
                    cx.notify();
                }
                true
            }
            "enter" => {
                self.apply_session_reasoning_index(self.session_reasoning_menu_index, cx);
                true
            }
            _ => false,
        }
    }

    fn handle_open_session_model_selector(
        &mut self,
        _: &OpenSessionModelSelector,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let available = self
            .session_model_shortcut_availability
            .is_action_available_or(window, cx, &OpenSessionModelSelector, false);
        if !available {
            return;
        }
        self.session_settings_open = false;
        self.set_session_model_menu_index_from_selection();
        set_open_cascading_menu(Some(CascadingMenuId::SessionModel), cx);
        cx.notify();
    }

    fn handle_open_session_reasoning_selector(
        &mut self,
        _: &OpenSessionReasoningSelector,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let available = self
            .session_reasoning_shortcut_availability
            .is_action_available_or(window, cx, &OpenSessionReasoningSelector, false);
        if !available {
            return;
        }
        self.session_settings_open = false;
        self.set_session_reasoning_menu_index_from_selection();
        set_open_cascading_menu(Some(CascadingMenuId::SessionReasoning), cx);
        cx.notify();
    }

    fn close_session_settings_menu(&mut self, cx: &mut Context<Self>) {
        if !self.session_settings_open {
            return;
        }

        let should_clear = cx
            .try_global::<CascadingMenuState>()
            .map(|state| state.open_menu() == Some(CascadingMenuId::SessionSettings))
            .unwrap_or(false);
        if should_clear {
            set_open_cascading_menu(None, cx);
        }

        self.session_settings_open = false;
        self.session_settings_hovered = None;
        self.session_settings_focus = SessionSettingsMenuFocus::Primary;
        self.session_settings_submenu_index = 0;
        cx.notify();
    }

    pub fn close_settings_menu(&mut self, cx: &mut Context<Self>) {
        self.close_session_settings_menu(cx);
    }

    fn session_settings_active_category(&self) -> SessionSettingsCategory {
        self.session_settings_hovered
            .unwrap_or(SessionSettingsCategory::Permissions)
    }

    fn session_settings_submenu_len(&self, category: SessionSettingsCategory) -> usize {
        match category {
            SessionSettingsCategory::Permissions => SESSION_SETTINGS_APPROVAL_OPTIONS.len(),
            SessionSettingsCategory::Sandbox => SessionSettingsSandboxOption::ALL.len(),
        }
    }

    fn session_settings_move_primary(&mut self, delta: i32, cx: &mut Context<Self>) {
        let current = self.session_settings_active_category().index() as i32;
        let len = SessionSettingsCategory::ALL.len() as i32;
        if len == 0 {
            return;
        }

        let next = (current + delta).clamp(0, len - 1) as usize;
        let next_category = SessionSettingsCategory::ALL[next];
        let mut changed = false;

        if self.session_settings_hovered != Some(next_category) {
            self.session_settings_hovered = Some(next_category);
            changed = true;
        }
        if self.session_settings_focus != SessionSettingsMenuFocus::Primary {
            self.session_settings_focus = SessionSettingsMenuFocus::Primary;
            changed = true;
        }
        if self.session_settings_submenu_index != 0 {
            self.session_settings_submenu_index = 0;
            changed = true;
        }

        if changed {
            cx.notify();
        }
    }

    fn session_settings_move_secondary(&mut self, delta: i32, cx: &mut Context<Self>) {
        let category = self.session_settings_active_category();
        let len = self.session_settings_submenu_len(category) as i32;
        if len == 0 {
            return;
        }

        let current = self.session_settings_submenu_index as i32;
        let next = (current + delta).clamp(0, len - 1) as usize;
        let mut changed = false;

        if self.session_settings_hovered.is_none() {
            self.session_settings_hovered = Some(category);
            changed = true;
        }
        if self.session_settings_focus != SessionSettingsMenuFocus::Secondary {
            self.session_settings_focus = SessionSettingsMenuFocus::Secondary;
            changed = true;
        }
        if self.session_settings_submenu_index != next {
            self.session_settings_submenu_index = next;
            changed = true;
        }

        if changed {
            cx.notify();
        }
    }

    fn session_settings_open_submenu(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        if self.session_settings_hovered.is_none() {
            self.session_settings_hovered = Some(SessionSettingsCategory::Permissions);
            changed = true;
        }
        if self.session_settings_focus != SessionSettingsMenuFocus::Secondary {
            self.session_settings_focus = SessionSettingsMenuFocus::Secondary;
            changed = true;
        }
        if self.session_settings_submenu_index != 0 {
            self.session_settings_submenu_index = 0;
            changed = true;
        }
        if changed {
            cx.notify();
        }
    }

    fn session_settings_apply_selection(&mut self, cx: &mut Context<Self>) {
        let category = self.session_settings_active_category();

        match category {
            SessionSettingsCategory::Permissions => {
                if self.codex_approval_policy_action.in_flight {
                    return;
                }

                let idx = self
                    .session_settings_submenu_index
                    .min(SESSION_SETTINGS_APPROVAL_OPTIONS.len().saturating_sub(1));
                let policy = SESSION_SETTINGS_APPROVAL_OPTIONS[idx].policy;
                self.set_codex_approval_policy(policy, cx);
            }
            SessionSettingsCategory::Sandbox => {
                if self.codex_sandbox_policy_action.in_flight {
                    return;
                }

                let idx = self
                    .session_settings_submenu_index
                    .min(SessionSettingsSandboxOption::ALL.len().saturating_sub(1));
                let option = SessionSettingsSandboxOption::ALL[idx];
                self.set_codex_sandbox_policy(option.policy(), cx);
            }
        }
    }

    fn handle_session_settings_key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        if !self.session_settings_open {
            return false;
        }

        match key {
            "escape" => {
                self.close_session_settings_menu(cx);
                true
            }
            "up" => {
                match self.session_settings_focus {
                    SessionSettingsMenuFocus::Primary => self.session_settings_move_primary(-1, cx),
                    SessionSettingsMenuFocus::Secondary => {
                        self.session_settings_move_secondary(-1, cx);
                    }
                }
                true
            }
            "down" => {
                match self.session_settings_focus {
                    SessionSettingsMenuFocus::Primary => self.session_settings_move_primary(1, cx),
                    SessionSettingsMenuFocus::Secondary => {
                        self.session_settings_move_secondary(1, cx)
                    }
                }
                true
            }
            "left" => {
                if self.session_settings_focus == SessionSettingsMenuFocus::Secondary {
                    self.session_settings_focus = SessionSettingsMenuFocus::Primary;
                    cx.notify();
                    true
                } else {
                    false
                }
            }
            "right" => {
                if self.session_settings_focus == SessionSettingsMenuFocus::Primary {
                    self.session_settings_open_submenu(cx);
                    true
                } else {
                    false
                }
            }
            "enter" => match self.session_settings_focus {
                SessionSettingsMenuFocus::Primary => {
                    self.session_settings_open_submenu(cx);
                    true
                }
                SessionSettingsMenuFocus::Secondary => {
                    self.session_settings_apply_selection(cx);
                    true
                }
            },
            _ => false,
        }
    }

    fn respond_permission_request(
        &mut self,
        request_id: &str,
        decision: PermissionDecision,
        cx: &mut Context<Self>,
    ) {
        let Some(client) = self.client.clone() else {
            self.permission_request_actions
                .entry(request_id.to_owned())
                .or_default()
                .fail("Control plane client is unavailable.");
            cx.notify();
            return;
        };

        let Some(feed) = self.feed.as_ref() else {
            return;
        };

        let action = self
            .permission_request_actions
            .entry(request_id.to_owned())
            .or_default();
        if action.in_flight {
            return;
        }

        action.start();
        cx.notify();

        let session_id = feed.session_id;
        let request_id = request_id.to_owned();
        let request_id_for_map = request_id.clone();
        let view = cx.entity();
        let task = cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            let request_id_for_request = request_id.clone();
            async move {
                let result = respond_permission_request(
                    &client,
                    session_id,
                    request_id_for_request,
                    decision,
                )
                .await;
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.on_respond_permission_request_completed(request_id, result, cx)
                    })
                });
            }
        });
        self.respond_permission_request_tasks
            .insert(request_id_for_map, task);
    }

    fn on_respond_permission_request_completed(
        &mut self,
        request_id: String,
        result: Result<RespondPermissionRequestResponse, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        self.respond_permission_request_tasks.remove(&request_id);

        let Some(action) = self.permission_request_actions.get_mut(&request_id) else {
            return;
        };

        match result {
            Ok(_resp) => {
                action.succeed();
                action.clear_error();
            }
            Err(err) => {
                action.fail(err.message);
            }
        }

        cx.notify();
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
                self.collapse_unseen_reasoning_in_timeline();
                // Ensure the scrollbar stays stable while scrolling through history by eagerly
                // measuring all loaded items once per page-load.
                self.timeline_list_state.clone().measure_all();
                self.timeline_follow_bottom.set(true);
                self.start_subscription(after, cx);
            }
            Err(err) => {
                self.error = Some(err.message.into());
            }
        }

        cx.notify();
    }

    fn on_policies_fetched(
        &mut self,
        generation: u64,
        session_id: SessionId,
        result: Result<Vec<SessionEvent>, ErrorEnvelope>,
        cx: &mut Context<Self>,
    ) {
        if generation != self.policies_fetch_generation {
            return;
        }

        self.policies_fetch_task = None;
        self.policies_fetch_in_flight = false;

        match result {
            Ok(events) => {
                self.policies_fetch_error = None;
                if let Some(feed) = self.feed.as_mut()
                    && feed.session_id == session_id
                {
                    for event in events {
                        feed.apply_live_event(event);
                    }
                    self.refresh_timeline_items();
                }
            }
            Err(err) => {
                self.policies_fetch_error = Some(err.message.into());
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
                if let SessionEventKind::CodexApprovalPolicyChanged(changed) = &ev.kind {
                    if self.codex_approval_policy_action.in_flight
                        && self.pending_codex_approval_policy == Some(changed.approval_policy)
                    {
                        self.codex_approval_policy_action.succeed();
                        self.pending_codex_approval_policy = None;
                        self.codex_approval_policy_timeout_task = None;
                    }
                    self.codex_approval_policy_action.clear_error();
                }
                if let SessionEventKind::CodexSandboxPolicyChanged(changed) = &ev.kind {
                    if self.codex_sandbox_policy_action.in_flight
                        && self
                            .pending_codex_sandbox_policy
                            .as_ref()
                            .is_some_and(|pending| pending == &changed.sandbox_policy)
                    {
                        self.codex_sandbox_policy_action.succeed();
                        self.pending_codex_sandbox_policy = None;
                        self.codex_sandbox_policy_timeout_task = None;
                    }
                    self.codex_sandbox_policy_action.clear_error();
                }
                let reasoning_key = match &ev.kind {
                    SessionEventKind::AssistantReasoning(reasoning) => Some(
                        reasoning
                            .item_id
                            .clone()
                            .unwrap_or_else(|| ev.session_event_id.to_string()),
                    ),
                    _ => None,
                };
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
                if let Some(key) = reasoning_key {
                    if self.seen_reasoning_keys.insert(key.clone()) {
                        // Default thinking blocks to collapsed when first observed.
                        self.collapsed_reasoning.insert(key);
                    } else if !self.collapsed_reasoning.contains(&key) {
                        // If the user expanded the in-progress reasoning, auto-collapse it once the
                        // completed (durable) reasoning arrives.
                        self.toggle_reasoning(&key, cx);
                    }
                }
            }
            SubscriptionEvent::SessionLiveEvent(ev) => {
                if ev.session_id != feed.session_id {
                    return;
                }
                let reasoning_key = match &ev.kind {
                    redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryPartAdded(
                        _,
                    )
                    | redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningSummaryDelta(
                        _,
                    )
                    | redesmyn_protocol::session_live::SessionLiveEventKind::AssistantReasoningRawDelta(
                        _,
                    ) => Some(
                        ev.item_id
                            .clone()
                            .unwrap_or_else(|| "assistant_reasoning".to_owned()),
                    ),
                    _ => None,
                };
                feed.apply_live_session_event(ev);
                self.refresh_timeline_items();
                if let Some(key) = reasoning_key
                    && self.seen_reasoning_keys.insert(key.clone())
                {
                    self.collapsed_reasoning.insert(key);
                }
            }
            SubscriptionEvent::Error(err) => {
                feed.apply_live_error(err.message);
            }
            SubscriptionEvent::Subscribed(_) | SubscriptionEvent::EventLog(_) => {}
        }

        cx.notify();
    }

    fn collapse_unseen_reasoning_in_timeline(&mut self) {
        for item in self.timeline_items.iter() {
            let key = match item {
                SessionTimelineItem::EphemeralReasoning(ephemeral) => Some(ephemeral.key.clone()),
                SessionTimelineItem::Event(event) => match &event.content {
                    SessionEventItemContent::AssistantReasoning(reasoning) => Some(
                        reasoning
                            .item_id
                            .clone()
                            .unwrap_or_else(|| event.session_event_id.to_string()),
                    ),
                    _ => None,
                },
                _ => None,
            };

            if let Some(key) = key
                && self.seen_reasoning_keys.insert(key.clone())
            {
                self.collapsed_reasoning.insert(key);
            }
        }
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
                self.collapse_unseen_reasoning_in_timeline();
                // GPUI list scrollbars are based on measured row heights. When we append a history
                // page, eagerly measure all loaded items so the thumb doesn't "flicker" while the
                // user scrolls through newly loaded history.
                self.timeline_list_state.clone().measure_all();
            }
            Err(err) => {
                feed.apply_history_error(err.message);
            }
        }

        cx.notify();
    }

    fn schedule_list_reset(&mut self, scroll_top: ListOffset, cx: &mut Context<Self>) {
        if !self.timeline_scrollbar_hidden {
            self.timeline_scrollbar_hidden = true;
        }
        self.timeline_list_reset_pending_scroll_top = Some(scroll_top);
        self.timeline_list_reset_generation = self.timeline_list_reset_generation.wrapping_add(1);
        let generation = self.timeline_list_reset_generation;
        let debounce = ui_test_mode_animation_duration(Duration::from_millis(50));
        self.timeline_list_reset_task =
            Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    gpui::Timer::after(debounce).await;
                    let Some(view) = weak.upgrade() else { return };
                    let _ = cx.update(|cx| {
                        view.update(cx, |this, cx| {
                            if this.timeline_list_reset_generation != generation {
                                return;
                            }
                            this.timeline_list_reset_task = None;
                            let Some(scroll_top) =
                                this.timeline_list_reset_pending_scroll_top.take()
                            else {
                                return;
                            };
                            this.timeline_scrollbar_hidden = false;
                            this.timeline_list_state
                                .reset(this.timeline_items.len() + 1);
                            this.timeline_list_state.scroll_to(scroll_top);
                            cx.notify();
                        })
                    });
                }
            }));
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
        let was_expanded = self.expanded_tool_events.contains(&session_event_id);
        if was_expanded {
            self.expanded_tool_events.remove(&session_event_id);
        } else {
            self.expanded_tool_events.insert(session_event_id);
        }

        let target = if was_expanded { 0.0 } else { 1.0 };
        let current = self
            .tool_event_transitions
            .get(&session_event_id)
            .map(|transition| transition.value())
            .unwrap_or_else(|| if was_expanded { 1.0 } else { 0.0 });

        let duration = ui_test_mode_animation_duration(Duration::from_millis(180));
        let mut transitions: HashMap<SessionEventId, ExpandCollapseTransition> =
            self.tool_event_transitions.as_ref().clone();

        if duration == Duration::from_millis(0) || (current - target).abs() < 1e-3 {
            transitions.remove(&session_event_id);
            self.tool_event_transition_guards.remove(&session_event_id);
        } else {
            transitions.insert(
                session_event_id,
                ExpandCollapseTransition {
                    started_at: Instant::now(),
                    from: current,
                    to: target,
                    duration,
                },
            );

            if !self
                .tool_event_transition_guards
                .contains_key(&session_event_id)
                && let Some(tracker) = ui_idle_tracker(cx)
            {
                self.tool_event_transition_guards
                    .insert(session_event_id, tracker.begin_transition());
            }
        }

        self.tool_event_transitions = Rc::new(transitions);
        self.invalidate_timeline_item(session_event_id);
        cx.notify();
    }

    fn toggle_reasoning(&mut self, key: &str, cx: &mut Context<Self>) {
        let key = key.to_owned();
        self.seen_reasoning_keys.insert(key.clone());
        let was_expanded = !self.collapsed_reasoning.contains(&key);
        if was_expanded {
            self.collapsed_reasoning.insert(key.clone());
        } else {
            self.collapsed_reasoning.remove(&key);
        }

        let target = if was_expanded { 0.0 } else { 1.0 };
        let current = self
            .reasoning_transitions
            .get(&key)
            .map(|transition| transition.value())
            .unwrap_or_else(|| if was_expanded { 1.0 } else { 0.0 });

        let duration = ui_test_mode_animation_duration(Duration::from_millis(180));
        let mut transitions: HashMap<String, ExpandCollapseTransition> =
            self.reasoning_transitions.as_ref().clone();

        if duration == Duration::from_millis(0) || (current - target).abs() < 1e-3 {
            transitions.remove(&key);
            self.reasoning_transition_guards.remove(&key);
        } else {
            transitions.insert(
                key.clone(),
                ExpandCollapseTransition {
                    started_at: Instant::now(),
                    from: current,
                    to: target,
                    duration,
                },
            );

            if !self.reasoning_transition_guards.contains_key(&key)
                && let Some(tracker) = ui_idle_tracker(cx)
            {
                self.reasoning_transition_guards
                    .insert(key.clone(), tracker.begin_transition());
            }
        }

        self.reasoning_transitions = Rc::new(transitions);
        self.invalidate_reasoning_item(&key);
        cx.notify();
    }

    fn toggle_permission_request(&mut self, request_id: &str, cx: &mut Context<Self>) {
        if self.expanded_permission_requests.contains(request_id) {
            self.expanded_permission_requests.remove(request_id);
        } else {
            self.expanded_permission_requests
                .insert(request_id.to_owned());
        }

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
        let mut transitions: HashMap<SessionEventId, ExpandCollapseTransition> =
            self.tool_group_transitions.as_ref().clone();

        if duration == Duration::from_millis(0) || (current - target).abs() < 1e-3 {
            transitions.remove(&group_id);
            self.tool_group_transition_guards.remove(&group_id);
        } else {
            transitions.insert(
                group_id,
                ExpandCollapseTransition {
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
        self.rebuild_permission_approval_state();
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

    fn rebuild_permission_approval_state(&mut self) {
        let mut request_ids = HashSet::new();
        let mut decisions: HashMap<String, PermissionDecisionState> = HashMap::new();

        for item in self.timeline_items.iter() {
            let SessionTimelineItem::Event(event) = item else {
                continue;
            };

            match &event.content {
                SessionEventItemContent::PermissionRequested(requested) => {
                    request_ids.insert(requested.request_id.clone());
                }
                SessionEventItemContent::PermissionDecided(decided) => {
                    decisions.insert(
                        decided.request_id.clone(),
                        PermissionDecisionState {
                            decision: decided.decision,
                            decided_by: decided.decided_by,
                        },
                    );
                }
                _ => {}
            }
        }

        self.permission_request_ids = Rc::new(request_ids);
        self.permission_decisions_by_request_id = Rc::new(decisions);
        self.expanded_permission_requests
            .retain(|id| self.permission_request_ids.contains(id));
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
                    let summary_main = tool_summary_preview(&summary_main, 160);

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
                    last_summary = Some(tool_summary_preview(&tool.output_preview, 160));

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

    fn full_text_message_artifact_path(
        &self,
        full_text_artifact: &ArtifactRef,
    ) -> Result<PathBuf, String> {
        let hint = full_text_artifact
            .storage_hint
            .as_ref()
            .ok_or_else(|| "artifact storage hint missing".to_string())?;
        resolve_artifact_path_from_storage_hint(hint, self.artifact_store_root.as_deref())
    }

    fn ensure_full_text_message_loaded(
        &mut self,
        session_event_id: SessionEventId,
        full_text_artifact: ArtifactRef,
        cx: &mut Context<Self>,
    ) {
        let path = match self.full_text_message_artifact_path(&full_text_artifact) {
            Ok(path) => path,
            Err(err) => {
                redesmyn_logging::tracing::warn!(
                    session_event_id = %session_event_id,
                    artifact_id = %full_text_artifact.artifact_id,
                    storage_hint = ?full_text_artifact.storage_hint,
                    error = %err,
                    "unsupported full-text message artifact storage hint"
                );
                self.full_text_message_states
                    .borrow_mut()
                    .insert(session_event_id, FullTextMessageLoadState::Failed);
                self.schedule_timeline_item_invalidation(session_event_id, cx);
                return;
            }
        };

        if path.as_os_str().is_empty() {
            self.full_text_message_states
                .borrow_mut()
                .insert(session_event_id, FullTextMessageLoadState::Failed);
            self.schedule_timeline_item_invalidation(session_event_id, cx);
            return;
        }

        let should_start = {
            let mut states = self.full_text_message_states.borrow_mut();
            match states.get(&session_event_id).copied() {
                Some(FullTextMessageLoadState::Loaded)
                | Some(FullTextMessageLoadState::Loading)
                | Some(FullTextMessageLoadState::Failed) => false,
                None => {
                    states.insert(session_event_id, FullTextMessageLoadState::Loading);
                    true
                }
            }
        };

        if should_start {
            self.schedule_timeline_item_invalidation(session_event_id, cx);
        }

        if !should_start {
            return;
        }

        let view = cx.entity();
        let path_for_log = path.clone();
        cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let task = gpui::background_executor().spawn(async move {
                    let text = std::fs::read_to_string(&path)?;
                    let doc = Arc::new(parse_markdown(&text, MarkdownParseOptions::default()));
                    Ok::<Arc<MarkdownDoc>, std::io::Error>(doc)
                });

                let result = task.await;
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| match result {
                        Ok(doc) => {
                            this.markdown_cache
                                .borrow_mut()
                                .insert(session_event_id, doc);
                            this.full_text_message_states
                                .borrow_mut()
                                .insert(session_event_id, FullTextMessageLoadState::Loaded);
                            this.schedule_timeline_item_invalidation(session_event_id, cx);
                        }
                        Err(err) => {
                            if err.kind() == std::io::ErrorKind::NotFound {
                                redesmyn_logging::tracing::debug!(
                                    session_event_id = %session_event_id,
                                    path = %path_for_log.display(),
                                    "full-text message artifact not found"
                                );
                            } else {
                                redesmyn_logging::tracing::warn!(
                                    session_event_id = %session_event_id,
                                    path = %path_for_log.display(),
                                    error = %err,
                                    "failed to load full-text message artifact"
                                );
                            }
                            this.full_text_message_states
                                .borrow_mut()
                                .insert(session_event_id, FullTextMessageLoadState::Failed);
                            this.schedule_timeline_item_invalidation(session_event_id, cx);
                        }
                    })
                });
            }
        })
        .detach();
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

    fn schedule_timeline_item_invalidation(
        &self,
        session_event_id: SessionEventId,
        cx: &mut Context<Self>,
    ) {
        let view = cx.entity();
        cx.spawn(move |_: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let _ = cx.update(|cx| {
                    view.update(cx, |this, cx| {
                        this.invalidate_timeline_item(session_event_id);
                        cx.notify();
                    })
                });
            }
        })
        .detach();
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
        let mut next: HashMap<SessionEventId, ExpandCollapseTransition> = HashMap::new();

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

    fn tick_tool_event_transitions_for_render(&mut self, window: &mut Window) {
        if self.tool_event_transitions.is_empty() {
            return;
        }

        let transitions = Rc::clone(&self.tool_event_transitions);
        let active_event_ids: Vec<SessionEventId> = transitions.keys().copied().collect();
        let mut next: HashMap<SessionEventId, ExpandCollapseTransition> = HashMap::new();

        for (session_event_id, transition) in transitions.iter() {
            if transition.is_done() {
                self.tool_event_transition_guards.remove(session_event_id);
            } else {
                next.insert(*session_event_id, *transition);
            }
        }

        if next.len() != self.tool_event_transitions.len() {
            self.tool_event_transitions = Rc::new(next);
        } else {
            // Keep the same Rc when nothing changes so list closures don't churn.
        }

        for session_event_id in active_event_ids {
            self.invalidate_timeline_item(session_event_id);
        }

        if !self.tool_event_transitions.is_empty() {
            window.request_animation_frame();
        }
    }

    fn tick_reasoning_transitions_for_render(&mut self, window: &mut Window) {
        if self.reasoning_transitions.is_empty() {
            return;
        }

        let transitions = Rc::clone(&self.reasoning_transitions);
        let active_keys: Vec<String> = transitions.keys().cloned().collect();
        let mut next: HashMap<String, ExpandCollapseTransition> = HashMap::new();

        for (key, transition) in transitions.iter() {
            if transition.is_done() {
                self.reasoning_transition_guards.remove(key);
            } else {
                next.insert(key.clone(), *transition);
            }
        }

        if next.len() != self.reasoning_transitions.len() {
            self.reasoning_transitions = Rc::new(next);
        } else {
            // Keep the same Rc when nothing changes so list closures don't churn.
        }

        for key in active_keys {
            self.invalidate_reasoning_item(&key);
        }

        if !self.reasoning_transitions.is_empty() {
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

        if self.timeline_scroll_to_bottom_pending {
            self.timeline_scroll_to_bottom_pending = false;
            self.timeline_list_state
                .scroll_to_reveal_item(self.timeline_items.len());
        } else if assistant_generating
            && self.timeline_follow_bottom.get()
            && distance_to_bottom > threshold
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
        self.tick_tool_event_transitions_for_render(window);
        self.tick_reasoning_transitions_for_render(window);

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
        let full_text_message_states = Rc::clone(&self.full_text_message_states);
        let expanded_tool_events = self.expanded_tool_events.clone();
        let expanded_tool_groups = self.expanded_tool_groups.clone();
        let collapsed_reasoning = self.collapsed_reasoning.clone();
        let reasoning_transitions = Rc::clone(&self.reasoning_transitions);
        let reasoning_shimmer_alpha = self.reasoning_shimmer_alpha();
        let exec_command_results = Rc::clone(&self.exec_command_result_by_invocation);
        let grouped_exec_command_result_ids =
            Rc::clone(&self.grouped_exec_command_result_event_ids);
        let tool_event_groups = Rc::clone(&self.tool_event_groups);
        let tool_event_group_membership = Rc::clone(&self.tool_event_group_membership);
        let tool_event_transitions = Rc::clone(&self.tool_event_transitions);
        let tool_group_transitions = Rc::clone(&self.tool_group_transitions);
        let permission_request_ids = Rc::clone(&self.permission_request_ids);
        let permission_decisions_by_request_id =
            Rc::clone(&self.permission_decisions_by_request_id);
        let reasoning_scroll_states = Rc::clone(&self.reasoning_scroll_states);
        let tool_event_scroll_handles = Rc::clone(&self.tool_event_scroll_handles);
        let follow_bottom = Rc::clone(&self.timeline_follow_bottom);
        let entity_id = cx.entity_id();
        let timeline_view = view.clone();

        let timeline_list_state = self.timeline_list_state.clone();
        let feed_list = list(timeline_list_state.clone(), move |ix, window, cx| {
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
	                            if !states.contains_key(&key) {
	                                states.insert(key.clone(), ReasoningScrollState::new());
	                            }
	                            let state =
	                                states.get_mut(&key).expect("cached reasoning state");

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

                        let summary_scrollable = ScrollFade::new(
                            summary_scroll_handle.clone(),
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
                                    let bar_color = theme.colors.foreground_muted.opacity(
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
                        )
                        .fade_height(theme.spacing.lg)
                        .bg(theme.colors.surface);

                        container = container.child(
                            StyledScrollbar::for_scroll_handle(
                                ("session_item_ephemeral_reasoning_summary_scrollbar", ix),
                                summary_scroll_handle.clone(),
                                summary_scrollable,
                            )
                            .style(ScrollbarStyle {
                                inset: -theme.spacing.sm,
                                ..ScrollbarStyle::default()
                            }),
                        );

	                        if let Some(raw_text) = raw_text {
	                            let (raw_scroll_handle, follow_raw_bottom) = {
	                                let mut states = reasoning_scroll_states.borrow_mut();
	                                if !states.contains_key(&key) {
	                                    states.insert(key.clone(), ReasoningScrollState::new());
	                                }
	                                let state =
	                                    states.get_mut(&key).expect("cached reasoning state");

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
                            let raw_id_key = stable_str_key(key.as_str());
                            let raw_scrollable = ScrollFade::new(
                                raw_scroll_handle.clone(),
                                div()
                                    .id(("session_item_ephemeral_reasoning_raw", raw_id_key))
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
                            )
                            .fade_height(theme.spacing.lg)
                            .bg(theme.colors.surface);

                            container = container.child(
                                StyledScrollbar::for_scroll_handle(
                                    ("session_item_ephemeral_reasoning_raw_scrollbar", raw_id_key),
                                    raw_scroll_handle.clone(),
                                    raw_scrollable,
                                )
                                .style(ScrollbarStyle {
                                    inset: -theme.spacing.sm,
                                    ..ScrollbarStyle::default()
                                }),
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
                            let progress = reasoning_transitions
                                .get(&key)
                                .map(|transition| transition.value())
                                .unwrap_or_else(|| if is_expanded { 1.0 } else { 0.0 });
                            let is_animating = reasoning_transitions.contains_key(&key);
                            let chevron = if is_expanded { "▾" } else { "▸" };

                            let toggle_view = timeline_view.clone();
                            let toggle_key = key.clone();
                            let gap = if is_animating {
                                theme.spacing.sm * progress
                            } else if is_expanded {
                                theme.spacing.sm
                            } else {
                                px(0.0)
                            };
                            let mut container = div()
                                .id(bubble_id.clone())
                                .w_full()
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(gap)
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

                            if is_expanded || is_animating {
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

                                let mut body = div()
                                    .id((bubble_id.clone(), "reasoning_body"))
                                    .w_full()
                                    .min_w_0()
                                    .flex()
                                    .flex_col()
                                    .gap(theme.spacing.sm);

	                                let (summary_scroll_handle, follow_summary_bottom) = {
	                                    let mut states = reasoning_scroll_states.borrow_mut();
	                                    if !states.contains_key(&key) {
	                                        states.insert(key.clone(), ReasoningScrollState::new());
	                                    }
	                                    let state =
	                                        states.get_mut(&key).expect("cached reasoning state");

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

                                let summary_scrollable = ScrollFade::new(
                                    summary_scroll_handle.clone(),
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
                                )
                                .fade_height(theme.spacing.lg)
                                .bg(theme.colors.surface);

                                body = body.child(
                                    StyledScrollbar::for_scroll_handle(
                                        (bubble_id.clone(), "reasoning_summary_scrollbar"),
                                        summary_scroll_handle.clone(),
                                        summary_scrollable,
                                    )
                                    .style(ScrollbarStyle {
                                        inset: -theme.spacing.sm,
                                        ..ScrollbarStyle::default()
                                    }),
                                );

	                                if let Some(raw) = reasoning.raw {
	                                    let (raw_scroll_handle, follow_raw_bottom) = {
	                                        let mut states = reasoning_scroll_states.borrow_mut();
	                                        if !states.contains_key(&key) {
	                                            states.insert(
	                                                key.clone(),
	                                                ReasoningScrollState::new(),
	                                            );
	                                        }
	                                        let state = states
	                                            .get_mut(&key)
	                                            .expect("cached reasoning state");

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

                                    body = body.child(
                                        div()
                                            .h(px(1.0))
                                            .bg(theme.colors.border.opacity(0.2)),
                                    );

                                    let raw_scrollable = ScrollFade::new(
                                        raw_scroll_handle.clone(),
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
                                    )
                                    .fade_height(theme.spacing.lg)
                                    .bg(theme.colors.surface);

                                    body = body.child(
                                        StyledScrollbar::for_scroll_handle(
                                            (bubble_id.clone(), "reasoning_raw_scrollbar"),
                                            raw_scroll_handle.clone(),
                                            raw_scrollable,
                                        )
                                        .style(ScrollbarStyle {
                                            inset: -theme.spacing.sm,
                                            ..ScrollbarStyle::default()
                                        }),
                                    );
                                }

                                let max_h = px(520.0) * progress;
                                let mut body = Expandable::new(body).opacity(progress);
                                if is_animating {
                                    body = body.max_height(max_h);
                                }
                                container = container.child(body);
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

                            let needs_full_text = full_text_artifact.is_some();
                            let full_text_state = {
                                full_text_message_states
                                    .borrow()
                                    .get(&session_event_id)
                                    .copied()
                            };
                            let show_truncation_notice = !needs_full_text;

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

                            let bubble_body: AnyElement = if needs_full_text
                                && full_text_state != Some(FullTextMessageLoadState::Loaded)
                            {
                                if full_text_state.is_none()
                                    && let Some(artifact) = full_text_artifact.clone()
                                {
                                    let load_view = timeline_view.clone();
                                    load_view.update(cx, |this, cx| {
                                        this.ensure_full_text_message_loaded(
                                            session_event_id,
                                            artifact,
                                            cx,
                                        );
                                    });
                                }

                                if full_text_state == Some(FullTextMessageLoadState::Failed) {
                                    let placeholder_color =
                                        text_color.unwrap_or(theme.colors.foreground_muted);
                                    let mut placeholder = div()
                                        .id((bubble_id.clone(), "full_text_failed"))
                                        .flex()
                                        .flex_col()
                                        .gap(theme.spacing.xs)
                                        .text_size(theme.typography.caption.size)
                                        .text_color(placeholder_color)
                                        .child("Failed to load full message.");

                                    if let Some(artifact) = full_text_artifact.clone() {
                                        let retry_view = timeline_view.clone();
                                        placeholder = placeholder.child(
                                            TextButton::new(
                                                (bubble_id.clone(), "retry_full_text"),
                                                "Retry",
                                            )
                                            .kind(ButtonKind::Secondary)
                                            .on_click(move |_, _, cx| {
                                                let artifact = artifact.clone();
                                                retry_view.update(cx, |this, cx| {
                                                    this.full_text_message_states
                                                        .borrow_mut()
                                                        .remove(&session_event_id);
                                                    this.ensure_full_text_message_loaded(
                                                        session_event_id,
                                                        artifact,
                                                        cx,
                                                    );
                                                });
                                            }),
                                        );
                                    }

                                    placeholder.into_any_element()
                                } else {
                                    let placeholder_color =
                                        text_color.unwrap_or(theme.colors.foreground_muted);
                                    div()
                                        .id((bubble_id.clone(), "loading_full_text"))
                                        .text_size(theme.typography.caption.size)
                                        .text_color(placeholder_color)
                                        .child("Loading full message…")
                                        .into_any_element()
                                }
                            } else {
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

                                let mut view =
                                    MarkdownView::new((bubble_id.clone(), "markdown"), doc)
                                        .show_truncation_notice(show_truncation_notice)
                                        .text_size(theme.typography.caption.size);
                                if let Some(text_color) = text_color {
                                    view = view.text_color(text_color);
                                }
                                view.into_any_element()
                            };

                            let bubble = div()
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
                                .child(bubble_body);

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

                            row = row.py(timeline_item_gap_y);

                            list.child(row.child(bubble))
                        }
                        SessionEventItemContent::PermissionsModeChanged(_)
                        | SessionEventItemContent::CodexApprovalPolicyChanged(_)
                        | SessionEventItemContent::CodexSandboxPolicyChanged(_) => div(),
                        SessionEventItemContent::PermissionRequested(requested) => {
                            let request_id = requested.request_id.clone();
                            let request_id_key = stable_str_key(request_id.as_str());
                            let decision_state =
                                permission_decisions_by_request_id.get(&request_id).copied();

                            let action_state = timeline_view
                                .read(cx)
                                .permission_request_actions
                                .get(&request_id)
                                .cloned()
                                .unwrap_or_default();

                            let details_expanded = timeline_view
                                .read(cx)
                                .expanded_permission_requests
                                .contains(&request_id);

                            const PERMISSION_DETAILS_MAX_LINES: usize = 12;
                            const PERMISSION_DETAILS_MAX_CHARS: usize = 800;

                            let value_needs_truncate = |value: &str| -> bool {
                                let mut chars = 0usize;
                                let mut lines = 0usize;

                                for ch in value.chars() {
                                    if chars >= PERMISSION_DETAILS_MAX_CHARS {
                                        return true;
                                    }

                                    if ch == '\n' {
                                        lines = lines.saturating_add(1);
                                        if lines >= PERMISSION_DETAILS_MAX_LINES {
                                            return true;
                                        }
                                    }

                                    chars = chars.saturating_add(1);
                                }

                                false
                            };

                            let truncate_value = |value: &str| -> String {
                                let mut out = String::new();
                                let mut chars = 0usize;
                                let mut lines = 0usize;

                                for ch in value.chars() {
                                    if chars >= PERMISSION_DETAILS_MAX_CHARS {
                                        break;
                                    }

                                    if ch == '\n' {
                                        lines = lines.saturating_add(1);
                                        if lines >= PERMISSION_DETAILS_MAX_LINES {
                                            break;
                                        }
                                    }

                                    out.push(ch);
                                    chars = chars.saturating_add(1);
                                }

                                if out.len() < value.len() {
                                    out = out.trim_end().to_owned();
                                    out.push_str("\n…");
                                }

                                out
                            };

                            let detail_row = |label: &'static str, value: String| {
                                div()
                                    .flex()
                                    .flex_row()
                                    .gap(theme.spacing.sm)
                                    .child(
                                        div()
                                            .flex_shrink_0()
                                            .text_color(theme.colors.foreground_muted)
                                            .child(format!("{label}:")),
                                    )
                                    .child(div().min_w_0().text_color(theme.colors.foreground).child(value))
                            };

                            let (details, details_collapsible) = match &requested.request {
                                PermissionRequest::CommandExecution(req) => {
                                    let command_collapsible = req
                                        .command
                                        .as_deref()
                                        .is_some_and(value_needs_truncate);
                                    let command = req
                                        .command
                                        .as_deref()
                                        .map(|command| {
                                            if details_expanded {
                                                command.to_owned()
                                            } else if command_collapsible {
                                                truncate_value(command)
                                            } else {
                                                command.to_owned()
                                            }
                                        })
                                        .unwrap_or_else(|| "<unknown>".to_owned());

                                    let reason_collapsible =
                                        req.reason.as_deref().is_some_and(value_needs_truncate);
                                    let reason = req
                                        .reason
                                        .as_deref()
                                        .map(|reason| {
                                            if details_expanded {
                                                reason.to_owned()
                                            } else if reason_collapsible {
                                                truncate_value(reason)
                                            } else {
                                                reason.to_owned()
                                            }
                                        })
                                        .unwrap_or_default();

                                    let mut body = div()
                                        .flex()
                                        .flex_col()
                                        .gap(theme.spacing.xs)
                                        .font(theme.typography.mono.font.clone())
                                        .text_size(theme.typography.caption.size)
                                        .child(detail_row("command", command));

                                    if let Some(cwd) = req.cwd.as_deref() {
                                        body = body.child(detail_row("cwd", cwd.to_owned()));
                                    }
                                    if !reason.is_empty() {
                                        body = body.child(detail_row("reason", reason));
                                    }

                                    (
                                        div()
                                            .px(theme.spacing.sm)
                                            .py(theme.spacing.sm)
                                            .rounded_sm()
                                            .bg(theme.colors.surface.opacity(0.35))
                                            .border_1()
                                            .border_color(theme.colors.border.opacity(0.25))
                                            .child(body)
                                            .into_any_element(),
                                        command_collapsible || reason_collapsible,
                                    )
                                }
                                PermissionRequest::FileChange(req) => {
                                    let root = req
                                        .grant_root
                                        .as_deref()
                                        .unwrap_or("<unknown>")
                                        .to_owned();
                                    let reason_collapsible =
                                        req.reason.as_deref().is_some_and(value_needs_truncate);
                                    let reason = req
                                        .reason
                                        .as_deref()
                                        .map(|reason| {
                                            if details_expanded {
                                                reason.to_owned()
                                            } else if reason_collapsible {
                                                truncate_value(reason)
                                            } else {
                                                reason.to_owned()
                                            }
                                        })
                                        .unwrap_or_default();

                                    let mut body = div()
                                        .flex()
                                        .flex_col()
                                        .gap(theme.spacing.xs)
                                        .font(theme.typography.mono.font.clone())
                                        .text_size(theme.typography.caption.size)
                                        .child(detail_row("root", root));

                                    if !reason.is_empty() {
                                        body = body.child(detail_row("reason", reason));
                                    }

                                    (
                                        div()
                                            .px(theme.spacing.sm)
                                            .py(theme.spacing.sm)
                                            .rounded_sm()
                                            .bg(theme.colors.surface.opacity(0.35))
                                            .border_1()
                                            .border_color(theme.colors.border.opacity(0.25))
                                            .child(body)
                                            .into_any_element(),
                                        reason_collapsible,
                                    )
                                }
                                PermissionRequest::Unknown { unknown_kind, .. } => (
                                    div()
                                        .px(theme.spacing.sm)
                                        .py(theme.spacing.sm)
                                        .rounded_sm()
                                        .bg(theme.colors.surface.opacity(0.35))
                                        .border_1()
                                        .border_color(theme.colors.border.opacity(0.25))
                                        .font(theme.typography.mono.font.clone())
                                        .text_size(theme.typography.caption.size)
                                        .text_color(theme.colors.foreground_muted)
                                        .child(format!("provider request: {unknown_kind}"))
                                        .into_any_element(),
                                    false,
                                ),
                            };

                            let (callout_kind, title) = match decision_state {
                                Some(PermissionDecisionState {
                                    decision: PermissionDecision::Deny,
                                    ..
                                }) => (CalloutKind::Danger, "Permission denied"),
                                Some(_) => (CalloutKind::Info, "Permission decided"),
                                None => (CalloutKind::Warning, "Permission required"),
                            };

                            let mut action = div()
                                .flex()
                                .flex_col()
                                .gap(theme.spacing.sm)
                                .child(details);

                            if details_collapsible {
                                let toggle_label = if details_expanded {
                                    "Show less"
                                } else {
                                    "Show more"
                                };

                                let toggle_view = timeline_view.clone();
                                let request_id_for_toggle = request_id.clone();
                                action = action.child(
                                    div().flex().justify_end().child(
                                        TextButton::new(
                                            ("permission_request_toggle", request_id_key),
                                            toggle_label,
                                        )
                                        .kind(ButtonKind::Secondary)
                                        .on_click(move |event, _window, cx| {
                                            if event.standard_click() {
                                                toggle_view.update(cx, |this, cx| {
                                                    this.toggle_permission_request(
                                                        request_id_for_toggle.as_str(),
                                                        cx,
                                                    );
                                                });
                                            }
                                        }),
                                    ),
                                );
                            }

                            if let Some(err) = action_state.error.clone() {
                                action = action.child(
                                    div()
                                        .text_color(theme.colors.danger)
                                        .text_size(theme.typography.caption.size)
                                        .child(err),
                                );
                            }

                            if let Some(state) = decision_state {
                                let decision_label = match state.decision {
                                    PermissionDecision::Approve => "Approved",
                                    PermissionDecision::Deny => "Denied",
                                    PermissionDecision::Unknown => "Decided",
                                };
                                let decided_by_label = match state.decided_by {
                                    PermissionDecisionBy::User => "user",
                                    PermissionDecisionBy::ModeAutoApprove => "auto_approve",
                                    PermissionDecisionBy::ModeAutoDeny => "auto_deny",
                                    PermissionDecisionBy::Timeout => "timeout",
                                    PermissionDecisionBy::Unknown => "unknown",
                                };
                                let color = match state.decision {
                                    PermissionDecision::Approve => theme.colors.foreground_muted,
                                    PermissionDecision::Deny => theme.colors.danger,
                                    PermissionDecision::Unknown => theme.colors.foreground_muted,
                                };
                                action = action.child(
                                    div()
                                        .text_color(color)
                                        .text_size(theme.typography.caption.size)
                                        .child(format!("{decision_label} ({decided_by_label})")),
                                );
                            } else {
                                let disabled = action_state.in_flight;
                                let disabled_reason = if action_state.in_flight {
                                    "Responding…"
                                } else {
                                    ""
                                };

                                let approve_view = timeline_view.clone();
                                let deny_view = timeline_view.clone();
                                let request_id_for_buttons = request_id.clone();

                                let buttons = div()
                                    .flex()
                                    .flex_row()
                                    .gap(theme.spacing.sm)
                                    .child(
                                        TextButton::new(
                                            ("permission_request_approve", request_id_key),
                                            "Approve",
                                        )
                                        .kind(ButtonKind::Primary)
                                        .disabled(disabled)
                                        .disabled_reason(disabled_reason)
                                        .on_click(move |event, _window, cx| {
                                            if event.standard_click() {
                                                let request_id = request_id_for_buttons.clone();
                                                approve_view.update(cx, |this, cx| {
                                                    this.respond_permission_request(
                                                        request_id.as_str(),
                                                        PermissionDecision::Approve,
                                                        cx,
                                                    );
                                                });
                                            }
                                        }),
                                    )
                                    .child(
                                        TextButton::new(
                                            ("permission_request_deny", request_id_key),
                                            "Deny",
                                        )
                                        .kind(ButtonKind::Danger)
                                        .disabled(disabled)
                                        .disabled_reason(disabled_reason)
                                        .on_click(move |event, _window, cx| {
                                            if event.standard_click() {
                                                deny_view.update(cx, |this, cx| {
                                                    this.respond_permission_request(
                                                        request_id.as_str(),
                                                        PermissionDecision::Deny,
                                                        cx,
                                                    );
                                                });
                                            }
                                        }),
                                    );

                                action = action.child(buttons);

                                if action_state.in_flight {
                                    action = action.child(
                                        div()
                                            .text_color(theme.colors.foreground_muted)
                                            .text_size(theme.typography.caption.size)
                                            .child("Responding…"),
                                    );
                                }
                            }

                            let callout = Callout::new(requested.summary)
                                .kind(callout_kind)
                                .title(title)
                                .action(action);

                            list.child(
                                div()
                                    .px(theme.spacing.sm)
                                    .py(timeline_item_gap_y)
                                    .child(callout),
                            )
                        }
                        SessionEventItemContent::PermissionDecided(decided) => {
                            if permission_request_ids.contains(&decided.request_id) {
                                div()
                            } else {
                                let (kind, message) = match decided.decision {
                                    PermissionDecision::Approve => (CalloutKind::Info, "Approved"),
                                    PermissionDecision::Deny => (CalloutKind::Danger, "Denied"),
                                    PermissionDecision::Unknown => (CalloutKind::Info, "Decided"),
                                };

                                let by = match decided.decided_by {
                                    PermissionDecisionBy::User => "user",
                                    PermissionDecisionBy::ModeAutoApprove => "auto_approve",
                                    PermissionDecisionBy::ModeAutoDeny => "auto_deny",
                                    PermissionDecisionBy::Timeout => "timeout",
                                    PermissionDecisionBy::Unknown => "unknown",
                                };

                                list.child(
                                    div()
                                        .px(theme.spacing.sm)
                                        .py(timeline_item_gap_y)
                                        .child(
                                            Callout::new(format!("{message} ({by})"))
                                                .kind(kind)
                                                .title("Permission decision"),
                                        ),
                                )
                            }
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
                                        let progress = tool_event_transitions
                                            .get(&item.session_event_id)
                                            .map(|transition| transition.value())
                                            .unwrap_or_else(|| if expanded { 1.0 } else { 0.0 });
                                        let is_animating = tool_event_transitions
                                            .contains_key(&item.session_event_id);
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

                                        let exec_command_full = is_exec_command.then(|| {
                                            exec_command
                                                .as_deref()
                                                .map(tidy_shell_command)
                                                .unwrap_or_else(|| tool.input_preview.clone())
                                        });

                                        let summary_main = if is_exec_command {
                                            tool_summary_preview(
                                                exec_command_full
                                                    .as_deref()
                                                    .unwrap_or(&tool.input_preview),
                                                240,
                                            )
                                        } else {
                                            tool_summary_preview(&tool.input_preview, 240)
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

                                        let gap = if is_animating {
                                            theme.spacing.xs * progress
                                        } else if expanded {
                                            theme.spacing.xs
                                        } else {
                                            px(0.0)
                                        };
                                        let mut block = div()
                                            .id(bubble_id.clone())
                                            .w_full()
                                            .min_w_0()
                                            .flex()
                                            .flex_col()
                                            .gap(gap)
                                            .px(theme.spacing.sm)
                                            .py(timeline_item_gap_y)
                                            .child(summary);

                                        if expanded || is_animating {
                                            let mut details_content = div()
                                                .pl(theme.spacing.lg)
                                                .flex()
                                                .flex_col()
                                                .gap(theme.spacing.xs)
                                                .font(theme.typography.mono.font.clone())
                                                .text_size(theme.typography.caption.size);

                                            if is_exec_command {
                                                if let Some(cwd) = exec_cwd.as_deref() {
                                                    details_content = details_content.child(
                                                        div()
                                                            .text_color(theme.colors.foreground_muted)
                                                            .child(format!("cwd: {cwd}")),
                                                    );
                                                }

                                                let command_text = exec_command_full
                                                    .clone()
                                                    .unwrap_or_else(|| tool.input_preview.clone());
                                                let is_multiline_command = command_text.contains('\n');
                                                details_content = details_content.child(if is_multiline_command {
                                                    div()
                                                        .id((bubble_id.clone(), "exec_command_block"))
                                                        .w_full()
                                                        .min_w_0()
                                                        .overflow_x_scroll()
                                                        .scrollbar_width(px(10.0))
                                                        .px(theme.spacing.md)
                                                        .py(theme.spacing.sm)
                                                        .rounded_md()
                                                        .bg(theme.colors.surface_elevated.opacity(0.35))
                                                        .border_1()
                                                        .border_color(theme.colors.border.opacity(0.6))
                                                        .text_color(theme.colors.foreground)
                                                        .whitespace_nowrap()
                                                        .child(command_text)
                                                } else {
                                                    div()
                                                        .id((bubble_id.clone(), "exec_command_block"))
                                                        .text_color(theme.colors.foreground)
                                                        .child(command_text)
                                                });

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
                                                        details_content = details_content.child(
                                                            div()
                                                                .text_color(color)
                                                                .child(format!("exit {exit_code}")),
                                                        );
                                                    }

                                                    if let Some(error) = result.error.as_ref() {
                                                        details_content = details_content.child(
                                                            div()
                                                                .text_color(theme.colors.danger)
                                                                .child(error.message.clone()),
                                                        );
                                                    }

                                                    if !remainder.is_empty() {
                                                        details_content = details_content.child(
                                                            div()
                                                                .text_color(theme.colors.foreground)
                                                                .child(remainder.to_string()),
                                                        );
                                                    }
                                                }
                                            } else {
                                                details_content = details_content.child(
                                                    div()
                                                        .text_color(theme.colors.foreground)
                                                        .child(tool.input_preview.clone()),
                                                );
                                            }

	                                            let details_scroll_handle = {
	                                                let mut handles =
	                                                    tool_event_scroll_handles.borrow_mut();
	                                                handles
	                                                    .get(&session_event_id)
	                                                    .cloned()
	                                                    .unwrap_or_else(|| {
	                                                        let handle = ScrollHandle::new();
	                                                        handles.insert(
	                                                            session_event_id,
	                                                            handle.clone(),
	                                                        );
	                                                        handle
	                                                    })
	                                            };

                                            let details_fade = ScrollFade::new(
                                                details_scroll_handle.clone(),
                                                div()
                                                    .id((bubble_id.clone(), "details"))
                                                    .w_full()
                                                    .min_w_0()
                                                    .max_h(px(640.0))
                                                    .overflow_y_scroll()
                                                    .track_scroll(&details_scroll_handle)
                                                    .occlude()
                                                    .on_scroll_wheel({
                                                        let handle = details_scroll_handle.clone();
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
                                                    .child(details_content),
                                            )
                                            .fade_height(theme.spacing.lg)
                                            .bg(theme.colors.surface);
                                            let details = StyledScrollbar::for_scroll_handle(
                                                (bubble_id.clone(), "details_scrollbar"),
                                                details_scroll_handle.clone(),
                                                details_fade,
                                            )
                                            .style(ScrollbarStyle {
                                                inset: -theme.spacing.sm,
                                                ..ScrollbarStyle::default()
                                            });

                                            let mut details =
                                                Expandable::new(details).opacity(progress);
                                            if is_animating {
                                                details = details
                                                    .max_height(px(640.0) * progress);
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

                                                let member_body_child = div()
                                                    .w_full()
                                                    .min_w_0()
                                                    .pl(theme.spacing.lg)
                                                    .child(block);

                                                let mut member_body =
                                                    Expandable::new(member_body_child)
                                                        .opacity(opacity);
                                                if is_animating {
                                                    member_body = member_body.max_height(max_h);
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
                                                        .child(member_body),
                                                )
                                            } else {
                                                let member_row_child = div()
                                                    .id((bubble_id.clone(), "group_member"))
                                                    .w_full()
                                                    .min_w_0()
                                                    .pl(theme.spacing.lg)
                                                    .child(block);

                                                let mut member_row =
                                                    Expandable::new(member_row_child)
                                                        .opacity(opacity);
                                                if is_animating {
                                                    member_row = member_row.max_height(max_h);
                                                }
                                                list.child(member_row)
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
                                            let progress = tool_event_transitions
                                                .get(&item.session_event_id)
                                                .map(|transition| transition.value())
                                                .unwrap_or_else(|| if expanded { 1.0 } else { 0.0 });
                                            let is_animating = tool_event_transitions
                                                .contains_key(&item.session_event_id);
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
                                                        .child(tool_summary_preview(&tool.output_preview, 240)),
                                                )
                                                .when(has_error, |this| {
                                                    this.child(
                                                        div()
                                                            .text_xs()
                                                            .text_color(theme.colors.danger)
                                                            .child("error"),
                                                    )
                                                });

                                            let gap = if is_animating {
                                                theme.spacing.xs * progress
                                            } else if expanded {
                                                theme.spacing.xs
                                            } else {
                                                px(0.0)
                                            };
                                            let mut block = div()
                                                .id(bubble_id.clone())
                                                .w_full()
                                                .min_w_0()
                                                .flex()
                                                .flex_col()
                                                .gap(gap)
                                                .px(theme.spacing.sm)
                                                .py(timeline_item_gap_y)
                                                .child(summary);

                                            if expanded || is_animating {
                                                let mut details_content = div()
                                                    .pl(theme.spacing.lg)
                                                    .flex()
                                                    .flex_col()
                                                    .gap(theme.spacing.xs)
                                                    .font(theme.typography.mono.font.clone())
                                                    .text_size(theme.typography.caption.size);

                                                if let Some(error) = tool.error.as_ref() {
                                                    details_content = details_content.child(
                                                        div()
                                                            .text_color(theme.colors.danger)
                                                            .child(error.message.clone()),
                                                    );
                                                }

                                                details_content = details_content.child(
                                                    div()
                                                        .text_color(theme.colors.foreground)
                                                        .child(tool.output_preview.clone()),
                                                );

	                                                let details_scroll_handle = {
	                                                    let mut handles =
	                                                        tool_event_scroll_handles.borrow_mut();
	                                                    handles
	                                                        .get(&session_event_id)
	                                                        .cloned()
	                                                        .unwrap_or_else(|| {
	                                                            let handle = ScrollHandle::new();
	                                                            handles.insert(
	                                                                session_event_id,
	                                                                handle.clone(),
	                                                            );
	                                                            handle
	                                                        })
	                                                };

                                                let details_fade = ScrollFade::new(
                                                    details_scroll_handle.clone(),
                                                    div()
                                                        .id((bubble_id.clone(), "details"))
                                                        .w_full()
                                                        .min_w_0()
                                                        .max_h(px(640.0))
                                                        .overflow_y_scroll()
                                                        .track_scroll(&details_scroll_handle)
                                                        .occlude()
                                                        .on_scroll_wheel({
                                                            let handle = details_scroll_handle.clone();
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
                                                        .child(details_content),
                                                )
                                                .fade_height(theme.spacing.lg)
                                                .bg(theme.colors.surface);
                                                let details = StyledScrollbar::for_scroll_handle(
                                                    (bubble_id.clone(), "details_scrollbar"),
                                                    details_scroll_handle.clone(),
                                                    details_fade,
                                                )
                                                .style(ScrollbarStyle {
                                                    inset: -theme.spacing.sm,
                                                    ..ScrollbarStyle::default()
                                                });

                                                let mut details =
                                                    Expandable::new(details).opacity(progress);
                                                if is_animating {
                                                    details = details
                                                        .max_height(px(640.0) * progress);
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

                                                    let member_body_child = div()
                                                        .w_full()
                                                        .min_w_0()
                                                        .pl(theme.spacing.lg)
                                                        .child(block);

                                                    let mut member_body =
                                                        Expandable::new(member_body_child)
                                                            .opacity(opacity);
                                                    if is_animating {
                                                        member_body = member_body.max_height(max_h);
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
                                                            .child(member_body),
                                                    )
                                                } else {
                                                    let member_row_child = div()
                                                        .id((bubble_id.clone(), "group_member"))
                                                        .w_full()
                                                        .min_w_0()
                                                        .pl(theme.spacing.lg)
                                                        .child(block);

                                                    let mut member_row =
                                                        Expandable::new(member_row_child)
                                                            .opacity(opacity);
                                                    if is_animating {
                                                        member_row =
                                                            member_row.max_height(max_h);
                                                    }
                                                    list.child(member_row)
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
        let feed_list = if self.timeline_scrollbar_hidden {
            feed_list.into_any_element()
        } else {
            StyledScrollbar::for_list_state(
                ("session_timeline_scrollbar", entity_id),
                timeline_list_state,
                feed_list,
            )
            .style(ScrollbarStyle {
                // Prefer placing the thumb in the "gutter" when the session timeline is embedded in a
                // padded container (e.g. details panels).
                inset: -theme.spacing.sm,
                ..ScrollbarStyle::default()
            })
            .into_any_element()
        };

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
        } else if self.codex_approval_policy_action.in_flight
            || self.codex_sandbox_policy_action.in_flight
        {
            (false, "Updating…")
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

        let displayed_codex_approval_policy = self.pending_codex_approval_policy.unwrap_or(
            self.feed
                .as_ref()
                .and_then(|feed| feed.codex_approval_policy),
        );
        let displayed_codex_sandbox_policy = self.pending_codex_sandbox_policy.clone().unwrap_or(
            self.feed
                .as_ref()
                .and_then(|feed| feed.codex_sandbox_policy.clone()),
        );

        let approvals_label = match displayed_codex_approval_policy {
            None if self.pending_codex_approval_policy.is_none()
                && self.policies_fetch_in_flight =>
            {
                "Loading…"
            }
            None if self.pending_codex_approval_policy.is_none()
                && self.policies_fetch_error.is_some() =>
            {
                "Unknown"
            }
            None => "Default",
            Some(CodexApprovalPolicy::UnlessTrusted) => "Unless trusted",
            Some(CodexApprovalPolicy::OnFailure) => "On failure",
            Some(CodexApprovalPolicy::OnRequest) => "On request",
            Some(CodexApprovalPolicy::Never) => "Never",
            Some(CodexApprovalPolicy::Unknown) => "Unknown",
        };

        let sandbox_label = match displayed_codex_sandbox_policy.as_ref() {
            None if self.pending_codex_sandbox_policy.is_none()
                && self.policies_fetch_in_flight =>
            {
                "Loading…"
            }
            None if self.pending_codex_sandbox_policy.is_none()
                && self.policies_fetch_error.is_some() =>
            {
                "Unknown"
            }
            None => "Default",
            Some(CodexSandboxPolicy::ReadOnly) => "Read-only",
            Some(CodexSandboxPolicy::WorkspaceWrite { .. }) => "Workspace write",
            Some(CodexSandboxPolicy::ExternalSandbox { .. }) => "External sandbox",
            Some(CodexSandboxPolicy::DangerFullAccess) => "Danger: full access",
            Some(CodexSandboxPolicy::Unknown) => "Unknown",
        };

        let approvals_disabled = self.codex_approval_policy_action.in_flight
            || self.client.is_none()
            || self.feed.is_none();

        let sandbox_disabled = self.codex_sandbox_policy_action.in_flight
            || self.client.is_none()
            || self.feed.is_none();

        let policies_status = if let Some(error) = self.model_fetch_error.clone() {
            Some(
                div()
                    .min_w_0()
                    .text_color(theme.colors.danger)
                    .text_size(theme.typography.caption.size)
                    .truncate()
                    .child(format!("Model: {error}")),
            )
        } else if let Some(error) = self.session_model_action.error.clone() {
            Some(
                div()
                    .min_w_0()
                    .text_color(theme.colors.danger)
                    .text_size(theme.typography.caption.size)
                    .truncate()
                    .child(format!("Model: {error}")),
            )
        } else if let Some(error) = self.policies_fetch_error.clone() {
            Some(
                div()
                    .min_w_0()
                    .text_color(theme.colors.danger)
                    .text_size(theme.typography.caption.size)
                    .truncate()
                    .child(format!("Policies: {error}")),
            )
        } else if let Some(error) = self.codex_approval_policy_action.error.clone() {
            Some(
                div()
                    .min_w_0()
                    .text_color(theme.colors.danger)
                    .text_size(theme.typography.caption.size)
                    .truncate()
                    .child(format!("Approvals: {error}")),
            )
        } else if let Some(error) = self.codex_sandbox_policy_action.error.clone() {
            Some(
                div()
                    .min_w_0()
                    .text_color(theme.colors.danger)
                    .text_size(theme.typography.caption.size)
                    .truncate()
                    .child(format!("Sandbox: {error}")),
            )
        } else if self.model_fetch_in_flight
            || self.session_model_action.in_flight
            || self.codex_approval_policy_action.in_flight
            || self.codex_sandbox_policy_action.in_flight
        {
            Some(
                div()
                    .text_color(theme.colors.foreground_muted)
                    .text_size(theme.typography.caption.size)
                    .child("Updating…"),
            )
        } else {
            None
        };

        let settings_disabled = self.client.is_none() || self.feed.is_none();
        let open_menu = cx
            .try_global::<CascadingMenuState>()
            .map(|state| state.open_menu())
            .unwrap_or(None);
        let settings_open =
            self.session_settings_open && open_menu == Some(CascadingMenuId::SessionSettings);
        let settings_hover_bg = theme.colors.accent;
        let settings_open_hover_bg = theme.colors.accent.opacity(0.65);
        let settings_open_border = theme.colors.ring.opacity(0.5);
        let settings_foreground = theme.colors.foreground;
        let settings_border_transparent = theme.colors.border.opacity(0.0);

        let mut settings_button = div()
            .id(("session_settings_button", entity_id))
            .flex()
            .flex_row()
            .items_center()
            .gap(px(6.0))
            .px(px(8.0))
            .py(px(3.0))
            .rounded(theme.radius.sm)
            .border_1()
            .border_color(theme.colors.border.opacity(0.0))
            .text_xs()
            .text_color(theme.colors.foreground_muted)
            .focusable()
            .focus(|mut style| {
                style.border_color = Some(theme.colors.ring);
                style
            })
            .when(settings_open, |this| {
                this.bg(theme.colors.accent.opacity(0.55))
                    .border_color(settings_open_border)
                    .text_color(settings_foreground)
            })
            .when(!settings_disabled, move |this| {
                this.cursor_pointer().hover(move |this| {
                    if settings_open {
                        this.bg(settings_open_hover_bg)
                            .border_color(settings_open_border)
                            .text_color(settings_foreground)
                    } else {
                        this.bg(settings_hover_bg)
                            .border_color(settings_border_transparent)
                            .text_color(settings_foreground)
                    }
                })
            })
            .when(settings_disabled, |this| {
                this.opacity(0.55).cursor_not_allowed()
            })
            .child("Settings");

        if !settings_disabled {
            settings_button = settings_button.on_mouse_down(gpui::MouseButton::Left, {
                let view = view.clone();
                move |_, _, app| {
                    view.update(app, |this, cx| {
                        this.toggle_session_settings_menu(cx);
                    });
                }
            });
        }

        let settings_menu = if settings_open && !settings_disabled {
            let row_style = CascadingMenuRowStyle::compact(&theme);
            let surface_style = CascadingMenuSurfaceStyle::compact(&theme);
            let primary_row_style = CascadingMenuRowStyle {
                gap: theme.spacing.sm,
                ..row_style
            };

            let primary_width = px(220.0);
            let secondary_width = px(240.0);
            let overlap = px(6.0);
            let padding_y = surface_style.padding_y;
            let row_height = row_style.height;
            let row_gap = theme.spacing.xs;

            let mut primary_list = div().flex().flex_col().gap(row_gap);

            for category in SessionSettingsCategory::ALL {
                let hovered = self.session_settings_hovered == Some(category);
                let current_label = match category {
                    SessionSettingsCategory::Permissions => approvals_label,
                    SessionSettingsCategory::Sandbox => sandbox_label,
                };
                let keyboard_selected =
                    hovered && self.session_settings_focus == SessionSettingsMenuFocus::Primary;
                let hover_opacity = if hovered && !keyboard_selected {
                    1.0
                } else {
                    0.0
                };

                let row =
                    cascading_menu_row(&theme, primary_row_style, keyboard_selected, hover_opacity)
                        .cursor_pointer()
                        .on_mouse_move(cx.listener(move |this, _, _, cx| {
                            if this.session_settings_hovered != Some(category)
                                || this.session_settings_focus != SessionSettingsMenuFocus::Primary
                            {
                                this.session_settings_hovered = Some(category);
                                this.session_settings_focus = SessionSettingsMenuFocus::Primary;
                                this.session_settings_submenu_index = 0;
                                cx.notify();
                            }
                        }))
                        .child(
                            div()
                                .flex_shrink_0()
                                .text_color(theme.colors.foreground)
                                .child(category.label()),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_row()
                                .items_center()
                                .justify_end()
                                .flex_1()
                                .gap(theme.spacing.xs)
                                .min_w_0()
                                .child(cascading_menu_row_value(&theme, current_label))
                                .child(div().flex_shrink_0().child("›")),
                        );

                primary_list = primary_list.child(row);
            }

            let primary_menu = cascading_menu_surface(&theme, surface_style)
                .w(primary_width)
                .child(primary_list);

            let approval_options = &SESSION_SETTINGS_APPROVAL_OPTIONS;

            let primary_row_count = SessionSettingsCategory::ALL.len();
            let primary_height = padding_y * 2.0
                + row_height * primary_row_count as f32
                + row_gap * primary_row_count.saturating_sub(1) as f32;

            let submenu_top = self.session_settings_hovered.map(|category| {
                let base_top = padding_y + (row_height + row_gap) * category.index() as f32;
                let rows = match category {
                    SessionSettingsCategory::Permissions => approval_options.len(),
                    SessionSettingsCategory::Sandbox => SessionSettingsSandboxOption::ALL.len(),
                };

                let height = padding_y * 2.0
                    + row_height * rows as f32
                    + row_gap * rows.saturating_sub(1) as f32;

                let mut top = base_top;
                if top + height > primary_height {
                    top = primary_height - height;
                }
                top
            });

            let submenu: Option<AnyElement> = match self.session_settings_hovered {
                Some(SessionSettingsCategory::Permissions) => {
                    let mut list = div().flex().flex_col().gap(row_gap);

                    for (idx, option) in approval_options.iter().enumerate() {
                        let value = option.policy;
                        let selected = displayed_codex_approval_policy == value;
                        let active = self.session_settings_focus
                            == SessionSettingsMenuFocus::Secondary
                            && self.session_settings_submenu_index == idx;

                        let indicator = cascading_menu_radio_indicator(&theme, selected, active);

                        let mut row = cascading_menu_row(&theme, primary_row_style, active, 0.0)
                            .child(indicator)
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_color(theme.colors.foreground)
                                    .truncate()
                                    .child(option.label),
                            );

                        if approvals_disabled {
                            row = row.opacity(0.55);
                        } else {
                            let view = view.clone();
                            row = row
                                .cursor_pointer()
                                .on_mouse_move(cx.listener(move |this, _, _, cx| {
                                    if this.session_settings_focus
                                        != SessionSettingsMenuFocus::Secondary
                                        || this.session_settings_submenu_index != idx
                                    {
                                        this.session_settings_focus =
                                            SessionSettingsMenuFocus::Secondary;
                                        this.session_settings_submenu_index = idx;
                                        cx.notify();
                                    }
                                }))
                                .on_mouse_down(gpui::MouseButton::Left, move |_, _, cx| {
                                    view.update(cx, |this, cx| {
                                        this.set_codex_approval_policy(value, cx);
                                    });
                                });
                        }

                        list = list.child(row);
                    }

                    Some(
                        cascading_menu_surface(&theme, surface_style)
                            .w(secondary_width)
                            .child(list)
                            .into_any_element(),
                    )
                }
                Some(SessionSettingsCategory::Sandbox) => {
                    let mut list = div().flex().flex_col().gap(row_gap);
                    let displayed = displayed_codex_sandbox_policy.as_ref();

                    for (idx, option) in SessionSettingsSandboxOption::ALL
                        .iter()
                        .copied()
                        .enumerate()
                    {
                        let selected = option.is_selected(displayed);
                        let active = self.session_settings_focus
                            == SessionSettingsMenuFocus::Secondary
                            && self.session_settings_submenu_index == idx;

                        let indicator = cascading_menu_radio_indicator(&theme, selected, active);

                        let label_color = match option {
                            SessionSettingsSandboxOption::DangerFullAccess => theme.colors.danger,
                            _ => theme.colors.foreground,
                        };

                        let mut row = cascading_menu_row(&theme, primary_row_style, active, 0.0)
                            .child(indicator)
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .text_color(label_color)
                                    .truncate()
                                    .child(option.label()),
                            );

                        if sandbox_disabled {
                            row = row.opacity(0.55);
                        } else {
                            let view = view.clone();
                            row = row
                                .cursor_pointer()
                                .on_mouse_move(cx.listener(move |this, _, _, cx| {
                                    if this.session_settings_focus
                                        != SessionSettingsMenuFocus::Secondary
                                        || this.session_settings_submenu_index != idx
                                    {
                                        this.session_settings_focus =
                                            SessionSettingsMenuFocus::Secondary;
                                        this.session_settings_submenu_index = idx;
                                        cx.notify();
                                    }
                                }))
                                .on_mouse_down(gpui::MouseButton::Left, move |_, _, cx| {
                                    view.update(cx, |this, cx| {
                                        this.set_codex_sandbox_policy(option.policy(), cx);
                                    });
                                });
                        }

                        list = list.child(row);
                    }

                    Some(
                        cascading_menu_surface(&theme, surface_style)
                            .w(secondary_width)
                            .child(list)
                            .into_any_element(),
                    )
                }
                None => None,
            };

            let menu_metrics = CascadingMenuMetrics {
                primary_width,
                secondary_width,
                overlap,
            };

            Some(
                div()
                    .absolute()
                    .bottom(px(36.0))
                    .left(px(0.0))
                    .key_context("SessionSettingsMenu")
                    .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .child(
                        CascadingMenu::new(primary_menu)
                            .metrics(menu_metrics)
                            .maybe_secondary(submenu_top, submenu),
                    )
                    .into_any_element(),
            )
        } else {
            None
        };

        let settings_anchor = div()
            .relative()
            .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(settings_button)
            .when_some(settings_menu, |this, menu| this.child(menu));

        let displayed_model_selection = self.displayed_session_model_selection();
        let selected_model_option =
            self.selected_model_option_for_selection(&displayed_model_selection);

        let resolved_reasoning_effort = displayed_model_selection
            .reasoning_effort
            .or_else(|| selected_model_option.map(|option| option.default_reasoning_effort));

        let model_value: SharedString =
            if self.model_fetch_in_flight && self.session_model_options.is_empty() {
                "Loading…".into()
            } else if let Some(model_id) = displayed_model_selection.model_id.as_ref() {
                selected_model_option
                    .map(|option| option.display_name.clone().into())
                    .unwrap_or_else(|| model_id.clone().into())
            } else if let Some(option) = selected_model_option {
                option.display_name.clone().into()
            } else {
                "Auto".into()
            };
        let reasoning_value: SharedString =
            model_reasoning_effort_label(resolved_reasoning_effort).into();

        let reasoning_values = self.reasoning_values_for_selection(&displayed_model_selection);

        let selector_keycap = |label: &'static str, enabled: bool| {
            div()
                .px(px(4.0))
                .py(px(1.0))
                .rounded(theme.radius.sm)
                .bg(theme.colors.surface.opacity(0.92))
                .border_1()
                .border_color(theme.colors.border.opacity(0.35))
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .opacity(if enabled { 0.55 } else { 1.0 })
                .child(label)
        };
        let selector_shortcut_fallback = false;
        let model_shortcut_enabled = self
            .session_model_shortcut_availability
            .is_action_available_or(
                window,
                cx,
                &OpenSessionModelSelector,
                selector_shortcut_fallback,
            );
        let reasoning_shortcut_enabled = self
            .session_reasoning_shortcut_availability
            .is_action_available_or(
                window,
                cx,
                &OpenSessionReasoningSelector,
                selector_shortcut_fallback,
            );

        let model_menu_open = open_menu == Some(CascadingMenuId::SessionModel);
        let reasoning_menu_open = open_menu == Some(CascadingMenuId::SessionReasoning);
        let model_selector_disabled = self.client.is_none()
            || self.feed.is_none()
            || self.session_model_action.in_flight
            || (self.model_fetch_in_flight && self.session_model_options.is_empty());
        let reasoning_selector_disabled = model_selector_disabled;

        let selector_open_border = theme.colors.ring.opacity(0.5);
        let selector_hover_bg = theme.colors.accent;
        let selector_open_hover_bg = theme.colors.accent.opacity(0.65);
        let selector_border_transparent = theme.colors.border.opacity(0.0);

        let mut model_selector_button = div()
            .id(("session_model_selector_button", entity_id))
            .flex()
            .flex_row()
            .items_center()
            .gap(px(6.0))
            .px(px(8.0))
            .py(px(3.0))
            .rounded(theme.radius.sm)
            .border_1()
            .border_color(theme.colors.border.opacity(0.0))
            .text_xs()
            .text_color(theme.colors.foreground_muted)
            .focusable()
            .focus(|mut style| {
                style.border_color = Some(theme.colors.ring);
                style
            })
            .when(model_menu_open, |this| {
                this.bg(theme.colors.accent.opacity(0.55))
                    .border_color(selector_open_border)
                    .text_color(theme.colors.foreground)
            })
            .when(!model_selector_disabled, move |this| {
                this.cursor_pointer().hover(move |this| {
                    if model_menu_open {
                        this.bg(selector_open_hover_bg)
                            .border_color(selector_open_border)
                            .text_color(theme.colors.foreground)
                    } else {
                        this.bg(selector_hover_bg)
                            .border_color(selector_border_transparent)
                            .text_color(theme.colors.foreground)
                    }
                })
            })
            .when(model_selector_disabled, |this| {
                this.opacity(0.55).cursor_not_allowed()
            })
            .child("◈")
            .child(div().max_w(px(120.0)).truncate().child(model_value.clone()))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.xs)
                    .child(selector_keycap("⌥M", model_shortcut_enabled)),
            );
        if !model_selector_disabled {
            model_selector_button = model_selector_button.on_mouse_down(gpui::MouseButton::Left, {
                let view = view.clone();
                move |_, _, app| {
                    view.update(app, |this, cx| {
                        this.session_settings_open = false;
                        let current = cx
                            .try_global::<CascadingMenuState>()
                            .map(|state| state.open_menu())
                            .unwrap_or(None);
                        let next = if current == Some(CascadingMenuId::SessionModel) {
                            None
                        } else {
                            this.set_session_model_menu_index_from_selection();
                            Some(CascadingMenuId::SessionModel)
                        };
                        set_open_cascading_menu(next, cx);
                        cx.notify();
                    });
                }
            });
        }

        let model_menu = if model_menu_open && !model_selector_disabled {
            let row_style = CascadingMenuRowStyle {
                gap: theme.spacing.sm,
                ..CascadingMenuRowStyle::compact(&theme)
            };
            let surface_style = CascadingMenuSurfaceStyle::compact(&theme);
            let mut list = div().flex().flex_col().gap(theme.spacing.xs);

            let model_selected = displayed_model_selection.model_id.is_none();
            let default_model_label = self
                .session_model_options
                .iter()
                .find(|option| option.is_default)
                .map(|option| option.display_name.clone())
                .unwrap_or_else(|| "Provider default".to_string());
            let default_active = self.session_model_menu_index == 0;
            let default_row = cascading_select_menu_item(
                &theme,
                row_style,
                default_active,
                0.0,
                cascading_menu_radio_indicator(&theme, model_selected, default_active),
                default_model_label,
            )
            .cursor_pointer()
            .on_mouse_move(cx.listener(|this, _, _, cx| {
                if this.session_model_menu_index != 0 {
                    this.session_model_menu_index = 0;
                    cx.notify();
                }
            }))
            .on_mouse_down(gpui::MouseButton::Left, {
                let view = view.clone();
                let current_reasoning = displayed_model_selection.reasoning_effort;
                move |_, _, cx| {
                    view.update(cx, |this, cx| {
                        this.set_session_model_selection(
                            SessionModelSelection {
                                model_id: None,
                                reasoning_effort: current_reasoning,
                            },
                            true,
                            cx,
                        );
                    });
                }
            });
            list = list.child(default_row);

            for (idx, option) in self.session_model_options.iter().enumerate() {
                let option_model_id = option.model_id.clone();
                let option_display = option.display_name.clone();
                let selected = displayed_model_selection
                    .model_id
                    .as_ref()
                    .is_some_and(|id| id == &option_model_id);
                let supported = option.supported_reasoning_efforts.clone();
                let current_reasoning = displayed_model_selection.reasoning_effort;
                let row_index = idx + 1;
                let active = self.session_model_menu_index == row_index;
                let row = cascading_select_menu_item(
                    &theme,
                    row_style,
                    active,
                    0.0,
                    cascading_menu_radio_indicator(&theme, selected, active),
                    option_display,
                )
                .cursor_pointer()
                .on_mouse_move(cx.listener(move |this, _, _, cx| {
                    if this.session_model_menu_index != row_index {
                        this.session_model_menu_index = row_index;
                        cx.notify();
                    }
                }))
                .on_mouse_down(gpui::MouseButton::Left, {
                    let view = view.clone();
                    move |_, _, cx| {
                        let reasoning_effort = current_reasoning
                            .and_then(|value| supported.contains(&value).then_some(value));
                        view.update(cx, |this, cx| {
                            this.set_session_model_selection(
                                SessionModelSelection {
                                    model_id: Some(option_model_id.clone()),
                                    reasoning_effort,
                                },
                                true,
                                cx,
                            );
                        });
                    }
                });
                list = list.child(row);
            }

            Some(
                div()
                    .absolute()
                    .bottom(px(36.0))
                    .left(px(0.0))
                    .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .child(
                        cascading_menu_surface(&theme, surface_style)
                            .w(px(280.0))
                            .child(list),
                    )
                    .into_any_element(),
            )
        } else {
            None
        };

        let model_selector_anchor = div()
            .relative()
            .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(model_selector_button)
            .when_some(model_menu, |this, menu| this.child(menu));

        let mut reasoning_selector_button = div()
            .id(("session_reasoning_selector_button", entity_id))
            .flex()
            .flex_row()
            .items_center()
            .gap(px(6.0))
            .px(px(8.0))
            .py(px(3.0))
            .rounded(theme.radius.sm)
            .border_1()
            .border_color(theme.colors.border.opacity(0.0))
            .text_xs()
            .text_color(theme.colors.foreground_muted)
            .focusable()
            .focus(|mut style| {
                style.border_color = Some(theme.colors.ring);
                style
            })
            .when(reasoning_menu_open, |this| {
                this.bg(theme.colors.accent.opacity(0.55))
                    .border_color(selector_open_border)
                    .text_color(theme.colors.foreground)
            })
            .when(!reasoning_selector_disabled, move |this| {
                this.cursor_pointer().hover(move |this| {
                    if reasoning_menu_open {
                        this.bg(selector_open_hover_bg)
                            .border_color(selector_open_border)
                            .text_color(theme.colors.foreground)
                    } else {
                        this.bg(selector_hover_bg)
                            .border_color(selector_border_transparent)
                            .text_color(theme.colors.foreground)
                    }
                })
            })
            .when(reasoning_selector_disabled, |this| {
                this.opacity(0.55).cursor_not_allowed()
            })
            .child("◎")
            .child(reasoning_value.clone())
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.xs)
                    .child(selector_keycap("⌥R", reasoning_shortcut_enabled)),
            );
        if !reasoning_selector_disabled {
            reasoning_selector_button =
                reasoning_selector_button.on_mouse_down(gpui::MouseButton::Left, {
                    let view = view.clone();
                    move |_, _, app| {
                        view.update(app, |this, cx| {
                            this.session_settings_open = false;
                            let current = cx
                                .try_global::<CascadingMenuState>()
                                .map(|state| state.open_menu())
                                .unwrap_or(None);
                            let next = if current == Some(CascadingMenuId::SessionReasoning) {
                                None
                            } else {
                                this.set_session_reasoning_menu_index_from_selection();
                                Some(CascadingMenuId::SessionReasoning)
                            };
                            set_open_cascading_menu(next, cx);
                            cx.notify();
                        });
                    }
                });
        }

        let reasoning_menu = if reasoning_menu_open && !reasoning_selector_disabled {
            let row_style = CascadingMenuRowStyle {
                gap: theme.spacing.sm,
                ..CascadingMenuRowStyle::compact(&theme)
            };
            let surface_style = CascadingMenuSurfaceStyle::compact(&theme);
            let mut list = div().flex().flex_col().gap(theme.spacing.xs);

            for (idx, effort) in reasoning_values.iter().copied().enumerate() {
                let selected = displayed_model_selection.reasoning_effort == effort;
                let label = model_reasoning_effort_label(effort);
                let model_id = displayed_model_selection.model_id.clone();
                let active = self.session_reasoning_menu_index == idx;
                let row = cascading_select_menu_item(
                    &theme,
                    row_style,
                    active,
                    0.0,
                    cascading_menu_radio_indicator(&theme, selected, active),
                    label,
                )
                .cursor_pointer()
                .on_mouse_move(cx.listener(move |this, _, _, cx| {
                    if this.session_reasoning_menu_index != idx {
                        this.session_reasoning_menu_index = idx;
                        cx.notify();
                    }
                }))
                .on_mouse_down(gpui::MouseButton::Left, {
                    let view = view.clone();
                    move |_, _, cx| {
                        view.update(cx, |this, cx| {
                            this.set_session_model_selection(
                                SessionModelSelection {
                                    model_id: model_id.clone(),
                                    reasoning_effort: effort,
                                },
                                true,
                                cx,
                            );
                        });
                    }
                });
                list = list.child(row);
            }

            Some(
                div()
                    .absolute()
                    .bottom(px(36.0))
                    .left(px(0.0))
                    .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .child(
                        cascading_menu_surface(&theme, surface_style)
                            .w(px(220.0))
                            .child(list),
                    )
                    .into_any_element(),
            )
        } else {
            None
        };

        let reasoning_selector_anchor = div()
            .relative()
            .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(reasoning_selector_button)
            .when_some(reasoning_menu, |this, menu| this.child(menu));

        let composer_action_bar = div()
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .gap(theme.spacing.sm)
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .min_w_0()
                    .child(model_selector_anchor)
                    .child(reasoning_selector_anchor)
                    .child(settings_anchor)
                    .when_some(policies_status, |this, status| this.child(status)),
            )
            .child(send_button);

        let composer = div()
            .relative()
            .w_full()
            .key_context("SessionComposer")
            .child(self.composer_input.clone())
            .child(
                div()
                    .absolute()
                    .left(theme.spacing.sm)
                    .right(theme.spacing.sm)
                    .bottom(theme.spacing.sm)
                    .child(composer_action_bar),
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

        let mut root = div()
            .flex()
            .flex_col()
            .gap(theme.spacing.sm)
            .size_full()
            .child(content)
            .on_action(cx.listener(Self::handle_open_session_model_selector))
            .on_action(cx.listener(Self::handle_open_session_reasoning_selector))
            .track_focus(&self.focus_handle(cx));

        let any_session_menu_open = matches!(
            open_menu,
            Some(
                CascadingMenuId::SessionSettings
                    | CascadingMenuId::SessionModel
                    | CascadingMenuId::SessionReasoning
            )
        );

        if any_session_menu_open {
            let view = cx.entity();
            root = root.on_mouse_down(gpui::MouseButton::Left, move |_, _, cx| {
                view.update(cx, |this, cx| {
                    this.close_session_settings_menu(cx);
                    let open_menu = cx
                        .try_global::<CascadingMenuState>()
                        .map(|state| state.open_menu())
                        .unwrap_or(None);
                    if matches!(
                        open_menu,
                        Some(CascadingMenuId::SessionModel | CascadingMenuId::SessionReasoning)
                    ) {
                        set_open_cascading_menu(None, cx);
                    }
                });
            });

            let view = cx.entity();
            root = root.capture_key_down(move |event, _window, cx| {
                let key = event.keystroke.key.as_str();
                let handled = view.update(cx, |this, cx| {
                    this.handle_session_settings_key(key, cx)
                        || this.handle_session_model_menu_key(key, cx)
                        || this.handle_session_reasoning_menu_key(key, cx)
                });

                if handled {
                    cx.stop_propagation();
                }
            });
        }

        root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_storage_hint_local_path() {
        let hint = StorageHint::LocalPath {
            local_path: "/tmp/redesmyn_message.md".to_string(),
        };
        let path = resolve_artifact_path_from_storage_hint(&hint, None).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/redesmyn_message.md"));
    }

    #[test]
    fn resolve_storage_hint_blob_key() {
        let root = PathBuf::from("/tmp/redesmyn_state");
        let artifact_id = ArtifactId::new();
        let hint = StorageHint::BlobKey {
            blob_key: format!("artifact/{artifact_id}"),
        };
        let path = resolve_artifact_path_from_storage_hint(&hint, Some(root.as_path())).unwrap();
        assert_eq!(
            path,
            root.join("artifacts").join(format!("{artifact_id}.bin"))
        );
    }

    #[test]
    fn resolve_storage_hint_blob_key_with_prefix() {
        let root = PathBuf::from("/tmp/redesmyn_state");
        let artifact_id = ArtifactId::new();
        let hint = StorageHint::BlobKey {
            blob_key: format!("local://artifact/{artifact_id}"),
        };
        let path = resolve_artifact_path_from_storage_hint(&hint, Some(root.as_path())).unwrap();
        assert_eq!(
            path,
            root.join("artifacts").join(format!("{artifact_id}.bin"))
        );
    }

    #[test]
    fn resolve_storage_hint_blob_key_requires_root() {
        let artifact_id = ArtifactId::new();
        let hint = StorageHint::BlobKey {
            blob_key: format!("artifact/{artifact_id}"),
        };
        let err = resolve_artifact_path_from_storage_hint(&hint, None).unwrap_err();
        assert!(err.contains("root"), "err={err}");
    }
}
