//! Desktop UI automation driver + semantic UI snapshot contract (T-15).
//!
//! This module defines the canonical typed message model used by local-only
//! deterministic UI automation surfaces.

use redesmyn_ids::{CommandId, EpicId, RequestId, SessionId, TaskId, TaskRelationId};

use crate::{ErrorEnvelope, ProtocolEnvelope, Timestamp};

/// A single UI driver protocol frame.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiDriverFrame {
    pub envelope: ProtocolEnvelope,
    pub message: UiDriverMessage,
}

impl UiDriverFrame {
    #[must_use]
    pub fn new(envelope: ProtocolEnvelope, message: UiDriverMessage) -> Self {
        Self { envelope, message }
    }
}

/// UI driver method identifier (initial minimal set; additive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiDriverMethod {
    GetSnapshot,
    OpenEpic,
    SelectTask,
    TriggerMerge,
    OpenSessionView,
    OpenDiffView,
    CaptureScreenshot,
    SetLeftPaneCollapsed,
    SetSettingsDialogOpen,
    CreateChatSession,
    CloseChatSession,
    PinChatSession,
    UnpinChatSession,
    TriggerRefresh,
    WaitForSnapshot,
    WaitForIdle,
    GraphSelectNode,
    GraphClearSelection,
    GraphToggleExpandedTaskCard,
    GraphMultiSelectAddNode,
    GraphMultiSelectRemoveNode,
    SessionSettingsMenuSetOpen,
    SessionSettingsMenuSendKey,
}

/// A request issued to the UI driver.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiDriverRequest {
    pub request_id: RequestId,
    pub payload: UiDriverRequestPayload,
}

impl UiDriverRequest {
    #[must_use]
    pub const fn method(&self) -> UiDriverMethod {
        self.payload.method()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum UiDriverRequestPayload {
    GetSnapshot(GetUiSnapshotRequest),
    OpenEpic(OpenEpicRequest),
    SelectTask(SelectTaskRequest),
    TriggerMerge(TriggerMergeRequest),
    OpenSessionView(OpenSessionViewRequest),
    OpenDiffView(OpenDiffViewRequest),
    CaptureScreenshot(CaptureScreenshotRequest),
    SetLeftPaneCollapsed(SetLeftPaneCollapsedRequest),
    SetSettingsDialogOpen(SetSettingsDialogOpenRequest),
    CreateChatSession(CreateChatSessionRequest),
    CloseChatSession(CloseChatSessionRequest),
    PinChatSession(PinChatSessionRequest),
    UnpinChatSession(UnpinChatSessionRequest),
    TriggerRefresh(TriggerRefreshRequest),
    WaitForSnapshot(WaitForUiSnapshotRequest),
    WaitForIdle(WaitForUiIdleRequest),
    GraphSelectNode(SelectGraphNodeRequest),
    GraphClearSelection(ClearGraphSelectionRequest),
    GraphToggleExpandedTaskCard(ToggleExpandedTaskCardRequest),
    GraphMultiSelectAddNode(MultiSelectAddNodeRequest),
    GraphMultiSelectRemoveNode(MultiSelectRemoveNodeRequest),
    SessionSettingsMenuSetOpen(SessionSettingsMenuSetOpenRequest),
    SessionSettingsMenuSendKey(SessionSettingsMenuSendKeyRequest),
}

impl UiDriverRequestPayload {
    #[must_use]
    pub const fn method(&self) -> UiDriverMethod {
        match self {
            Self::GetSnapshot(_) => UiDriverMethod::GetSnapshot,
            Self::OpenEpic(_) => UiDriverMethod::OpenEpic,
            Self::SelectTask(_) => UiDriverMethod::SelectTask,
            Self::TriggerMerge(_) => UiDriverMethod::TriggerMerge,
            Self::OpenSessionView(_) => UiDriverMethod::OpenSessionView,
            Self::OpenDiffView(_) => UiDriverMethod::OpenDiffView,
            Self::CaptureScreenshot(_) => UiDriverMethod::CaptureScreenshot,
            Self::SetLeftPaneCollapsed(_) => UiDriverMethod::SetLeftPaneCollapsed,
            Self::SetSettingsDialogOpen(_) => UiDriverMethod::SetSettingsDialogOpen,
            Self::CreateChatSession(_) => UiDriverMethod::CreateChatSession,
            Self::CloseChatSession(_) => UiDriverMethod::CloseChatSession,
            Self::PinChatSession(_) => UiDriverMethod::PinChatSession,
            Self::UnpinChatSession(_) => UiDriverMethod::UnpinChatSession,
            Self::TriggerRefresh(_) => UiDriverMethod::TriggerRefresh,
            Self::WaitForSnapshot(_) => UiDriverMethod::WaitForSnapshot,
            Self::WaitForIdle(_) => UiDriverMethod::WaitForIdle,
            Self::GraphSelectNode(_) => UiDriverMethod::GraphSelectNode,
            Self::GraphClearSelection(_) => UiDriverMethod::GraphClearSelection,
            Self::GraphToggleExpandedTaskCard(_) => UiDriverMethod::GraphToggleExpandedTaskCard,
            Self::GraphMultiSelectAddNode(_) => UiDriverMethod::GraphMultiSelectAddNode,
            Self::GraphMultiSelectRemoveNode(_) => UiDriverMethod::GraphMultiSelectRemoveNode,
            Self::SessionSettingsMenuSetOpen(_) => UiDriverMethod::SessionSettingsMenuSetOpen,
            Self::SessionSettingsMenuSendKey(_) => UiDriverMethod::SessionSettingsMenuSendKey,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiDriverResponseStatus {
    Ok,
    Error,
}

/// A response issued by the UI driver.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiDriverResponse {
    pub request_id: RequestId,
    pub result: UiDriverResponseResult,
}

impl UiDriverResponse {
    #[must_use]
    pub const fn status(&self) -> UiDriverResponseStatus {
        match &self.result {
            UiDriverResponseResult::Error(_) => UiDriverResponseStatus::Error,
            _ => UiDriverResponseStatus::Ok,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum UiDriverResponseResult {
    GetSnapshot(GetUiSnapshotResponse),
    OpenEpic(OpenEpicResponse),
    SelectTask(SelectTaskResponse),
    TriggerMerge(TriggerMergeResponse),
    OpenSessionView(OpenSessionViewResponse),
    OpenDiffView(OpenDiffViewResponse),
    CaptureScreenshot(CaptureScreenshotResponse),
    SetLeftPaneCollapsed(SetLeftPaneCollapsedResponse),
    SetSettingsDialogOpen(SetSettingsDialogOpenResponse),
    CreateChatSession(CreateChatSessionResponse),
    CloseChatSession(CloseChatSessionResponse),
    PinChatSession(PinChatSessionResponse),
    UnpinChatSession(UnpinChatSessionResponse),
    TriggerRefresh(TriggerRefreshResponse),
    WaitForSnapshot(WaitForUiSnapshotResponse),
    WaitForIdle(WaitForUiIdleResponse),
    GraphSelectNode(SelectGraphNodeResponse),
    GraphClearSelection(ClearGraphSelectionResponse),
    GraphToggleExpandedTaskCard(ToggleExpandedTaskCardResponse),
    GraphMultiSelectAddNode(MultiSelectAddNodeResponse),
    GraphMultiSelectRemoveNode(MultiSelectRemoveNodeResponse),
    SessionSettingsMenuSetOpen(SessionSettingsMenuSetOpenResponse),
    SessionSettingsMenuSendKey(SessionSettingsMenuSendKeyResponse),
    Error(ErrorEnvelope),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetUiSnapshotRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GetUiSnapshotResponse {
    pub snapshot: UiSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpenEpicRequest {
    pub epic_slug: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpenEpicResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SelectTaskRequest {
    pub task_slug: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SelectTaskResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TriggerMergeRequest {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub task_slug: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TriggerMergeResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<CommandId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpenSessionViewRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpenSessionViewResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpenDiffViewRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OpenDiffViewResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSettingsMenuSetOpenRequest {
    pub open: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSettingsMenuSetOpenResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSettingsMenuSendKeyRequest {
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionSettingsMenuSendKeyResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetLeftPaneCollapsedRequest {
    pub collapsed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetLeftPaneCollapsedResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSettingsDialogOpenRequest {
    pub open: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SetSettingsDialogOpenResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CreateChatSessionRequest {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CreateChatSessionResponse {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CloseChatSessionRequest {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CloseChatSessionResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PinChatSessionRequest {
    pub session_id: SessionId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PinChatSessionResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnpinChatSessionRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnpinChatSessionResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TriggerRefreshRequest {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub epic_slug: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TriggerRefreshResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<CommandId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForUiSnapshotRequest {
    pub timeout_ms: u64,
    pub predicate: UiSnapshotPredicate,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiSnapshotPredicate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primary_view: Option<UiPrimaryView>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub epic_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub in_flight_empty: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_layout_settled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_selection_settled: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForUiSnapshotResponse {
    pub snapshot: UiSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForUiIdleRequest {
    pub timeout_ms: u64,
    pub quiescence_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WaitForUiIdleResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SelectGraphNodeRequest {
    pub task_id: TaskId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SelectGraphNodeResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClearGraphSelectionRequest {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ClearGraphSelectionResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToggleExpandedTaskCardRequest {
    pub task_id: TaskId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToggleExpandedTaskCardResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MultiSelectAddNodeRequest {
    pub task_id: TaskId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MultiSelectAddNodeResponse {}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MultiSelectRemoveNodeRequest {
    pub task_id: TaskId,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MultiSelectRemoveNodeResponse {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiScreenshotWindow {
    Primary,
    All,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CaptureScreenshotRequest {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name_hint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<UiScreenshotWindow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_decorations: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CaptureScreenshotResponse {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub png_path: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub png_data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiPrimaryView {
    EpicSelector,
    EpicWorkspace,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiLeftPaneState {
    pub visible: bool,
    pub collapsed: bool,
    pub width: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiSelectionState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epic_id: Option<EpicId>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub epic_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub task_slug: String,
    /// Selected domain graph edge (TaskRelationId), when the UI supports relation selection.
    ///
    /// Note: UI graph edge selection is surfaced via `UiSnapshot.graph.selected_edge`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_id: Option<TaskRelationId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UiGraphLoadState {
    Unselected,
    Loading,
    Loaded,
    Empty,
    Error,
    /// A state not understood by this binary (forward compatible).
    #[serde(other)]
    Unknown,
}

impl Default for UiGraphLoadState {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum UiGraphNodeId {
    Task(TaskId),
    Trunk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct UiGraphEdgeId {
    pub from: UiGraphNodeId,
    pub to: UiGraphNodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiGraphCameraState {
    pub origin_world_x: i32,
    pub origin_world_y: i32,
    pub zoom_percent: u32,
}

impl Default for UiGraphCameraState {
    fn default() -> Self {
        Self {
            origin_world_x: 0,
            origin_world_y: 0,
            zoom_percent: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct UiGraphState {
    #[serde(default)]
    pub load_state: UiGraphLoadState,
    pub node_count: u32,
    pub edge_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_node: Option<UiGraphNodeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_edge: Option<UiGraphEdgeId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multi_selected_nodes: Vec<UiGraphNodeId>,
    pub expanded_task_card_open: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expanded_task_id: Option<TaskId>,
    pub selection_bar_visible: bool,
    /// True when the graph's layout animation has finished (node positions stable).
    ///
    /// This intentionally does **not** include camera or selection-bar transitions.
    pub layout_settled: bool,
    /// True when selection-driven UI transitions have finished (layout + camera + selection bar).
    ///
    /// Prefer this for deterministic tests that need "selection applied".
    pub selection_settled: bool,
    #[serde(default)]
    pub camera: UiGraphCameraState,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiInFlightAction {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command_id: Option<CommandId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiErrorCallout {
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct UiComposerState {
    pub sending: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Stable, machine-readable representation of what the UI is showing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiSnapshot {
    pub captured_at: Timestamp,
    pub primary_view: UiPrimaryView,
    pub left_pane: UiLeftPaneState,
    pub selection: UiSelectionState,
    #[serde(default)]
    pub graph: UiGraphState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub in_flight: Vec<UiInFlightAction>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<UiErrorCallout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_chat_session_id: Option<SessionId>,
    #[serde(default)]
    pub pinned_chat_composer: UiComposerState,
}

/// UI driver protocol messages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
pub enum UiDriverMessage {
    Request(UiDriverRequest),
    Response(UiDriverResponse),
}
