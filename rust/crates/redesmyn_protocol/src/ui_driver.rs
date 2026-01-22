//! Desktop UI automation driver + semantic UI snapshot contract (T-15).
//!
//! This module defines the canonical typed message model used by local-only
//! deterministic UI automation surfaces.

use redesmyn_ids::{CommandId, EpicId, RequestId, TaskId, TaskRelationId};

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
pub struct CaptureScreenshotRequest {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CaptureScreenshotResponse {
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub edge_id: Option<TaskRelationId>,
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

/// Stable, machine-readable representation of what the UI is showing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UiSnapshot {
    pub captured_at: Timestamp,
    pub primary_view: UiPrimaryView,
    pub left_pane: UiLeftPaneState,
    pub selection: UiSelectionState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub in_flight: Vec<UiInFlightAction>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<UiErrorCallout>,
}

/// UI driver protocol messages.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum UiDriverMessage {
    Request(UiDriverRequest),
    Response(UiDriverResponse),
}
