mod command_palette_overlay;

use std::sync::Arc;
use std::time::Duration;

use gpui::{
    App, AsyncApp, ClickEvent, Context, Entity, FocusHandle, Focusable, Render, ScrollHandle,
    SharedString, Subscription, Task, WeakEntity, Window, div, prelude::*, px,
};
use tokio::sync::{mpsc, watch};

use redesmyn_protocol::ui_driver::{
    CaptureScreenshotResponse, ClearGraphSelectionResponse, CreateChatSessionResponse,
    MultiSelectAddNodeResponse, MultiSelectRemoveNodeResponse, SelectGraphNodeResponse,
    ToggleExpandedTaskCardResponse, TriggerRefreshResponse, UiComposerState, UiDriverRequestPayload,
    UiDriverResponse, UiDriverResponseResult, UiErrorCallout, UiInFlightAction, UiLeftPaneState,
    UiPrimaryView, UiSelectionState, UiSnapshot, UiSnapshotPredicate, WaitForUiIdleRequest,
    WaitForUiIdleResponse, WaitForUiSnapshotRequest, WaitForUiSnapshotResponse,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, RepoScope, Timestamp};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;
use redesmyn_ui_session::SessionView;

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, OverlaySurfaceKind, ProgressPill, ScrollArea,
    SplitPane, SplitPaneAxis, SplitPaneEvent, SplitPaneState, TextButton, TextInput,
    TextInputEvent, overlay_surface,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::task_filters::{
    MERGE_READINESS_OPTIONS, TASK_STATE_OPTIONS, TaskAgentStatus, TaskFilterCategory, TaskFilters,
    merge_readiness_title, task_state_title,
};
use redesmyn_ui::utils::{
    TransitionMap, UserActionState, theme_for_window, ui_test_mode_animation_duration,
};
use redesmyn_ui_graph::GraphView;

use crate::app::SessionViewerFixtureEmitter;
use crate::command_palette::{
    CloseCommandPalette, SelectNextCommand, SelectPreviousCommand, ToggleCommandPalette,
};
use crate::control_plane_client::{ControlPlaneClient, ControlPlaneClientError};
use crate::task_filters::{
    CloseTaskFilters, OpenTaskFilters, TaskFiltersActivate, TaskFiltersClearFocusedChip,
    TaskFiltersMoveDown, TaskFiltersMoveLeft, TaskFiltersMoveRight, TaskFiltersMoveUp,
    TaskFiltersToggleChipFocus,
};

use self::command_palette_overlay::CommandPaletteOverlay;

#[derive(Debug)]
pub struct DesktopModel {
    config: Arc<redesmyn_config::RustConfig>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
    session_control_plane_client: Option<ClientInProcEndpoint>,
    task_control_plane_client: Option<ClientInProcEndpoint>,
    chrome_control_plane_client: Option<ControlPlaneClient>,
    session_viewer_fixture: Option<SessionViewerFixtureEmitter>,
    ui_driver_rx: Option<mpsc::UnboundedReceiver<crate::ui_driver::UiDriverCommand>>,
}

impl DesktopModel {
    #[must_use]
    pub fn new(
        config: Arc<redesmyn_config::RustConfig>,
        daemon_host_id: Option<redesmyn_ids::HostId>,
        tokio_handle: tokio::runtime::Handle,
        session_control_plane_client: Option<ClientInProcEndpoint>,
        task_control_plane_client: Option<ClientInProcEndpoint>,
        chrome_control_plane_client: Option<ClientInProcEndpoint>,
        session_viewer_fixture: Option<SessionViewerFixtureEmitter>,
        ui_driver_rx: Option<mpsc::UnboundedReceiver<crate::ui_driver::UiDriverCommand>>,
    ) -> Self {
        Self {
            config,
            daemon_host_id,
            session_control_plane_client,
            task_control_plane_client,
            chrome_control_plane_client: chrome_control_plane_client
                .map(|conn| ControlPlaneClient::new(tokio_handle, conn)),
            session_viewer_fixture,
            ui_driver_rx,
        }
    }

    pub fn take_control_plane_client(&mut self) -> Option<ClientInProcEndpoint> {
        self.session_control_plane_client.take()
    }

    pub fn take_task_control_plane_client(&mut self) -> Option<ClientInProcEndpoint> {
        self.task_control_plane_client.take()
    }

    pub fn take_ui_driver_rx(
        &mut self,
    ) -> Option<mpsc::UnboundedReceiver<crate::ui_driver::UiDriverCommand>> {
        self.ui_driver_rx.take()
    }

    pub fn take_session_viewer_fixture(&mut self) -> Option<SessionViewerFixtureEmitter> {
        self.session_viewer_fixture.take()
    }
}

#[derive(Debug, Clone)]
struct UiUpdateCounter {
    tx: watch::Sender<u64>,
}

impl UiUpdateCounter {
    fn new() -> Self {
        let (tx, _rx) = watch::channel(0_u64);
        Self { tx }
    }

    fn bump(&self) {
        let next = *self.tx.borrow() + 1;
        let _ = self.tx.send(next);
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.tx.subscribe()
    }
}

pub struct RootView {
    model: Entity<DesktopModel>,
    split_pane: Entity<SplitPane>,
    session_pane: Entity<EpicSessionPaneHost>,
    workspace_pane: Entity<WorkspacePaneHost>,
    focus_handle: FocusHandle,
    command_palette: CommandPaletteOverlay,
    chrome: ChromeState,
    ui_updates: UiUpdateCounter,
    ui_driver_action: UserActionState,
    ui_driver_action_label: Option<SharedString>,
    _subscriptions: Vec<Subscription>,
}

impl RootView {
    #[must_use]
    pub fn new(model: Entity<DesktopModel>, cx: &mut Context<Self>) -> Self {
        let ui_driver_rx = model.update(cx, |model, _cx| model.take_ui_driver_rx());

        let initial_split_state = cx
            .try_global::<UiContext>()
            .map(|ui| ui.main_split_pane_state())
            .unwrap_or_else(SplitPaneState::default);

        let ui_updates = UiUpdateCounter::new();
        let session_pane = cx.new(|cx| EpicSessionPaneHost::new(model.clone(), cx));
        let workspace_pane = cx.new(|cx| {
            WorkspacePaneHost::new(
                model.clone(),
                initial_split_state.collapsed,
                ui_updates.clone(),
                cx,
            )
        });

        let split_pane_state = initial_split_state;
        let split_primary = session_pane.clone();
        let split_secondary = workspace_pane.clone();

        let split_pane = cx.new(|_| {
            SplitPane::new(
                SplitPaneAxis::Horizontal,
                split_pane_state,
                split_primary.into(),
                split_secondary.into(),
            )
            .min_primary_px(280.0)
        });

        let focus_handle = cx.focus_handle();
        let command_palette = CommandPaletteOverlay::new(
            focus_handle.clone(),
            split_pane.clone(),
            workspace_pane.clone(),
            cx,
        );

        let palette_input = command_palette.input_entity();

        let mut subscriptions = Vec::new();
        subscriptions.push(cx.observe_global::<UiContext>(|this, cx| this.notify_ui_updated(cx)));

        subscriptions.push(cx.subscribe(&split_pane, |this, _, event, cx| match event {
            SplitPaneEvent::StateChanged(state) => {
                this.workspace_pane.update(cx, |pane, cx| {
                    pane.set_sessions_collapsed(state.collapsed, cx)
                });
                this.persist_split_pane_state(*state, cx);
                this.ui_updates.bump();
            }
        }));

        subscriptions.push(cx.subscribe(&palette_input, |this, _, event, cx| {
            this.command_palette
                .handle_text_input_event(event.clone(), cx);
            this.ui_updates.bump();
        }));

        let mut this = Self {
            model,
            split_pane,
            session_pane,
            workspace_pane,
            focus_handle,
            command_palette,
            chrome: ChromeState::new(),
            ui_updates,
            ui_driver_action: UserActionState::default(),
            ui_driver_action_label: None,
            _subscriptions: subscriptions,
        };

        if let Some(rx) = ui_driver_rx {
            this.start_ui_driver(rx, cx);
        }

        this
    }

    fn notify_ui_updated(&mut self, cx: &mut Context<Self>) {
        self.ui_updates.bump();
        cx.notify();
    }

    fn persist_split_pane_state(&mut self, state: SplitPaneState, cx: &mut Context<Self>) {
        if cx.try_global::<UiContext>().is_none() {
            return;
        }

        let result = cx
            .global_mut::<UiContext>()
            .set_main_split_pane_state(state);
        match result {
            Ok(()) => self
                .workspace_pane
                .update(cx, |pane, cx| pane.set_ui_settings_error(None, cx)),
            Err(error) => {
                redesmyn_logging::tracing::error!(
                    error = %error,
                    "failed to persist main split pane state"
                );
                self.workspace_pane.update(cx, |pane, cx| {
                    pane.set_ui_settings_error(Some(error.to_string().into()), cx);
                });
            }
        }
    }

    fn toggle_command_palette(
        &mut self,
        _: &ToggleCommandPalette,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.toggle(window, cx);
        self.ui_updates.bump();
    }

    fn close_command_palette(
        &mut self,
        _: &CloseCommandPalette,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.handle_close_action(window, cx);
        self.ui_updates.bump();
    }

    fn select_previous_command(
        &mut self,
        _: &SelectPreviousCommand,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.select_previous(cx);
        self.ui_updates.bump();
    }

    fn select_next_command(
        &mut self,
        _: &SelectNextCommand,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.command_palette.select_next(cx);
        self.ui_updates.bump();
    }

    fn subscribe_ui_updates(&self) -> watch::Receiver<u64> {
        self.ui_updates.subscribe()
    }

    fn start_ui_driver(
        &mut self,
        mut rx: mpsc::UnboundedReceiver<crate::ui_driver::UiDriverCommand>,
        cx: &mut Context<Self>,
    ) {
        cx.spawn(move |root: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                while let Some(cmd) = rx.recv().await {
                    let request_id = cmd.request.request_id;
                    let result = handle_ui_driver_request(&root, cmd.request.payload, &cx).await;
                    let response = UiDriverResponse { request_id, result };
                    let _ = cmd.respond_to.send(response);
                }
            }
        })
        .detach();
    }

    fn set_left_pane_collapsed(&mut self, collapsed: bool, cx: &mut Context<Self>) {
        let state = self.split_pane.read(cx).state();
        if state.collapsed == collapsed {
            return;
        }

        self.split_pane
            .update(cx, |pane, cx| pane.toggle_collapsed(cx));
        self.ui_updates.bump();
    }

    fn toggle_sessions_pane(&mut self, cx: &mut Context<Self>) {
        self.split_pane
            .update(cx, |pane, cx| pane.toggle_collapsed(cx));
    }

    fn toggle_panel(&mut self, panel: ChromePanel, cx: &mut Context<Self>) {
        self.chrome.panel = match self.chrome.panel {
            Some(open) if open == panel => None,
            _ => Some(panel),
        };
        self.notify_ui_updated(cx);
    }

    fn close_panel(&mut self, cx: &mut Context<Self>) {
        if self.chrome.panel.take().is_some() {
            self.notify_ui_updated(cx);
        }
    }

    fn select_epic(&mut self, slug: String, cx: &mut Context<Self>) {
        let span =
            redesmyn_logging::redesmyn_info_span!("ui.chrome.select_epic", epic_slug = %slug);
        let _guard = span.enter();

        if self.chrome.selected_epic_slug.as_deref() == Some(slug.as_str()) {
            self.close_panel(cx);
            return;
        }

        self.chrome.selected_epic_slug = Some(slug.clone());
        self.chrome.panel = None;

        let selected = self
            .chrome
            .epics
            .iter()
            .find(|epic| epic.slug == slug)
            .cloned();

        let selected_for_session_pane = selected.clone();
        let slug_for_workspace = slug.clone();

        self.session_pane.update(cx, move |pane, cx| {
            pane.set_selected_epic(selected_for_session_pane, cx)
        });
        self.workspace_pane.update(cx, move |pane, cx| {
            pane.set_selected_epic(Some(slug_for_workspace), selected, cx)
        });
        self.notify_ui_updated(cx);
    }

    fn clear_refresh_error(&mut self, cx: &mut Context<Self>) {
        self.chrome.refresh.clear_error();
        self.notify_ui_updated(cx);
    }

    fn clear_theme_error(&mut self, cx: &mut Context<Self>) {
        self.chrome.theme_error = None;
        self.notify_ui_updated(cx);
    }

    fn set_theme_preference(&mut self, preference: ThemePreference, cx: &mut Context<Self>) {
        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.chrome.theme_preference_set",
            preference = ?preference
        );
        let _guard = span.enter();

        if cx.try_global::<UiContext>().is_none() {
            self.chrome.theme_error = Some("UI context unavailable; theme not saved.".into());
            self.notify_ui_updated(cx);
            return;
        }

        match cx
            .global_mut::<UiContext>()
            .set_theme_preference(preference)
        {
            Ok(()) => {
                self.chrome.theme_error = None;
                redesmyn_logging::tracing::info!("saved theme preference");
            }
            Err(error) => {
                redesmyn_logging::tracing::error!(
                    error = %error,
                    "failed to save theme preference"
                );
                self.chrome.theme_error = Some(error.to_string().into());
            }
        }

        self.notify_ui_updated(cx);
    }

    fn start_refresh(&mut self, reason: RefreshReason, cx: &mut Context<Self>) {
        if self.chrome.refresh.in_flight {
            return;
        }

        let span = redesmyn_logging::redesmyn_info_span!("ui.chrome.refresh", reason = ?reason);
        let _guard = span.enter();

        self.chrome.refresh.start();

        let client = self.model.read(cx).chrome_control_plane_client.clone();
        let tokio = client.as_ref().map(|client| client.tokio().clone());

        self.notify_ui_updated(cx);

        self.chrome.refresh_task = Some(cx.spawn(
            move |weak: gpui::WeakEntity<Self>, cx: &mut AsyncApp| {
                let cx = cx.clone();
                async move {
                    let Some(entity) = weak.upgrade() else {
                        return;
                    };

                    let Some(client) = client else {
                        if cx
                            .update(|cx| {
                                entity.update(cx, |this, cx| {
                                    this.chrome.refresh_task = None;
                                    this.chrome
                                        .refresh
                                        .fail("Control plane client unavailable (not embedded).");
                                    this.notify_ui_updated(cx);
                                })
                            })
                            .is_err()
                        {
                            return;
                        }
                        return;
                    };

                    let tokio = tokio.expect("tokio handle is present when client is present");
                    let task = tokio.spawn(async move {
                        let status = client.status().await;
                        let epics = client.list_epics().await;
                        (status, epics)
                    });

                    let (status, epics) = match task.await {
                        Ok(result) => result,
                        Err(error) => {
                            let _ = cx.update(|cx| {
                                entity.update(cx, |this, cx| {
                                    this.chrome.refresh_task = None;
                                    this.chrome.refresh.fail(format!("Refresh failed: {error}"));
                                    this.notify_ui_updated(cx);
                                })
                            });
                            return;
                        }
                    };

                    let _ = cx.update(|cx| {
                        entity.update(cx, |this, cx| {
                            this.chrome.refresh_task = None;

                            let mut errors = Vec::new();

                            match status {
                                Ok(status) => {
                                    this.chrome.control_plane_status = Some(status);
                                }
                                Err(error) => {
                                    redesmyn_logging::tracing::error!(
                                        error = %error,
                                        "control plane status request failed"
                                    );
                                    errors.push(format!("Status: {error}"));
                                }
                            }

                            match epics {
                                Ok(mut epics) => {
                                    epics.sort_by(|a, b| a.slug.cmp(&b.slug));
                                    this.chrome.epics = epics;

                                    if let Some(selected) = this.chrome.selected_epic_slug.as_deref()
                                    {
                                        let still_present = this
                                            .chrome
                                            .epics
                                            .iter()
                                            .any(|epic| epic.slug == selected);
                                        if !still_present {
                                            redesmyn_logging::tracing::warn!(
                                                epic_slug = %selected,
                                                "selected epic no longer present; clearing selection"
                                            );
                                            this.chrome.selected_epic_slug = None;
                                        }
                                    }

                                    let selected = this
                                        .chrome
                                        .selected_epic_slug
                                        .as_deref()
                                        .and_then(|slug| {
                                            this.chrome.epics.iter().find(|epic| epic.slug == slug)
                                        })
                                        .cloned();
                                    let selected_slug = this.chrome.selected_epic_slug.clone();
                                    let selected_for_session_pane = selected.clone();
                                    this.session_pane.update(cx, move |pane, cx| {
                                        pane.set_selected_epic(selected_for_session_pane, cx);
                                    });
                                    this.workspace_pane.update(cx, move |pane, cx| {
                                        pane.set_selected_epic(selected_slug, selected, cx);
                                    });
                                }
                                Err(error) => {
                                    redesmyn_logging::tracing::error!(
                                        error = %error,
                                        "control plane list epics request failed"
                                    );
                                    errors.push(format!("Epics: {error}"));
                                }
                            }

                            if errors.is_empty() {
                                this.chrome.refresh.succeed();
                            } else {
                                this.chrome.refresh.fail(errors.join(" · "));
                            }

                            this.notify_ui_updated(cx);
                        })
                    });
                }
            },
        ));
    }

    fn ui_snapshot(&self, cx: &App) -> UiSnapshot {
        let split_state = self.split_pane.read(cx).state();
        let pinned_chat_session_id = self.session_pane.read(cx).pinned_session_id;
        let pinned_chat_composer = self
            .session_pane
            .read(cx)
            .session_view
            .read(cx)
            .ui_composer_state();
        let workspace = self.workspace_pane.read(cx);

        let primary_view = if self.chrome.selected_epic_slug.is_some() {
            UiPrimaryView::EpicWorkspace
        } else {
            UiPrimaryView::EpicSelector
        };

        let graph_view = workspace.graph_view.clone();
        let graph_details = graph_view.read(cx).ui_graph_state();
        let graph = redesmyn_protocol::ui_driver::UiGraphState {
            load_state: workspace.graph_state,
            node_count: workspace.graph_node_count,
            edge_count: workspace.graph_edge_count,
            ..graph_details
        };

        let selected_task_id = graph.selected_node.and_then(|node| match node {
            redesmyn_protocol::ui_driver::UiGraphNodeId::Task(task_id) => Some(task_id),
            redesmyn_protocol::ui_driver::UiGraphNodeId::Trunk => None,
        });
        let selected_task_slug = selected_task_id
            .and_then(|task_id| graph_view.read(cx).task_slug_for_task_node(task_id))
            .unwrap_or_default();

        let selection = UiSelectionState {
            epic_id: None,
            epic_slug: self
                .chrome
                .selected_epic_slug
                .clone()
                .unwrap_or_else(String::new),
            task_id: selected_task_id,
            task_slug: selected_task_slug,
            // `edge_id` is a domain TaskRelationId selection (not a UI graph edge). Graph edge
            // selection is surfaced via `UiSnapshot.graph.selected_edge`.
            edge_id: None,
        };

        let mut errors = Vec::new();
        if let Some(err) = workspace.ui_settings_error.clone() {
            errors.push(UiErrorCallout {
                message: err.to_string(),
            });
        }
        if workspace.graph_state == redesmyn_protocol::ui_driver::UiGraphLoadState::Error {
            if let Some(err) = workspace.graph_error.clone() {
                errors.push(UiErrorCallout {
                    message: err.to_string(),
                });
            }
        }
        if let Some(err) = self.chrome.refresh.error.clone() {
            errors.push(UiErrorCallout {
                message: err.to_string(),
            });
        }
        if let Some(err) = self.chrome.theme_error.clone() {
            errors.push(UiErrorCallout {
                message: err.to_string(),
            });
        }
        if self.command_palette.is_open() {
            if let Some(err) = self.command_palette.visible_error() {
                errors.push(UiErrorCallout { message: err });
            }
        }
        if let Some(err) = self.ui_driver_action.error.clone() {
            errors.push(UiErrorCallout {
                message: err.to_string(),
            });
        }

        let mut in_flight = Vec::new();
        if workspace.graph_state == redesmyn_protocol::ui_driver::UiGraphLoadState::Loading {
            in_flight.push(UiInFlightAction {
                label: "Loading graph…".to_string(),
                command_id: None,
            });
        }
        if self.chrome.refresh.in_flight {
            in_flight.push(UiInFlightAction {
                label: "Refreshing…".to_string(),
                command_id: None,
            });
        }
        if self.command_palette.is_open() {
            if let Some(label) = self.command_palette.visible_in_flight_label() {
                in_flight.push(UiInFlightAction {
                    label,
                    command_id: None,
                });
            }
        }
        if self.ui_driver_action.in_flight {
            let label = self
                .ui_driver_action_label
                .clone()
                .map(|label| label.to_string())
                .unwrap_or_else(|| "Working…".to_string());
            in_flight.push(UiInFlightAction {
                label,
                command_id: None,
            });
        }

        UiSnapshot {
            captured_at: if redesmyn_ui::utils::ui_test_mode_enabled() {
                Timestamp::from_offset_date_time(time::OffsetDateTime::UNIX_EPOCH)
            } else {
                Timestamp::now_utc()
            },
            primary_view,
            left_pane: UiLeftPaneState {
                visible: true,
                collapsed: split_state.collapsed,
                width: split_state.primary_size_px.round().max(0.0) as u32,
            },
            selection,
            graph,
            in_flight,
            errors,
            pinned_chat_session_id,
            pinned_chat_composer,
        }
    }
}

fn ui_unavailable_snapshot() -> UiSnapshot {
    UiSnapshot {
        captured_at: Timestamp::now_utc(),
        primary_view: UiPrimaryView::EpicSelector,
        left_pane: UiLeftPaneState {
            visible: true,
            collapsed: true,
            width: 0,
        },
        selection: UiSelectionState {
            epic_id: None,
            epic_slug: String::new(),
            task_id: None,
            task_slug: String::new(),
            edge_id: None,
        },
        graph: redesmyn_protocol::ui_driver::UiGraphState::default(),
        in_flight: Vec::new(),
        errors: vec![UiErrorCallout {
            message: "UI unavailable.".to_string(),
        }],
        pinned_chat_session_id: None,
        pinned_chat_composer: UiComposerState::default(),
    }
}

fn snapshot_matches_predicate(snapshot: &UiSnapshot, predicate: &UiSnapshotPredicate) -> bool {
    if let Some(primary_view) = predicate.primary_view {
        if snapshot.primary_view != primary_view {
            return false;
        }
    }

    if !predicate.epic_slug.trim().is_empty() && snapshot.selection.epic_slug != predicate.epic_slug
    {
        return false;
    }

    if let Some(expected_empty) = predicate.in_flight_empty {
        if snapshot.in_flight.is_empty() != expected_empty {
            return false;
        }
    }

    if let Some(expected_task_id) = predicate.selected_task_id {
        if snapshot.selection.task_id != Some(expected_task_id) {
            return false;
        }
    }

    if let Some(expected_layout_settled) = predicate.graph_layout_settled {
        if snapshot.graph.layout_settled != expected_layout_settled {
            return false;
        }
    }

    if let Some(expected_selection_settled) = predicate.graph_selection_settled {
        if snapshot.graph.selection_settled != expected_selection_settled {
            return false;
        }
    }

    true
}

async fn wait_for_snapshot(
    root: &WeakEntity<RootView>,
    req: WaitForUiSnapshotRequest,
    cx: &AsyncApp,
) -> Result<UiSnapshot, ErrorEnvelope> {
    let timeout = if req.timeout_ms == 0 {
        Duration::from_millis(2_000)
    } else {
        Duration::from_millis(req.timeout_ms)
    };

    let mut updates_rx = cx
        .update(|cx| {
            let Some(root) = root.upgrade() else {
                return Err(());
            };
            Ok::<_, ()>(root.read(cx).subscribe_ui_updates())
        })
        .ok()
        .and_then(Result::ok)
        .ok_or_else(|| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))?;

    let (_idle_dummy_tx, idle_dummy_rx) = watch::channel(0_usize);
    let mut idle_rx = cx
        .update(|cx| {
            cx.try_global::<UiContext>()
                .map(|ui| ui.idle_tracker().subscribe())
        })
        .ok()
        .flatten()
        .unwrap_or(idle_dummy_rx);

    let timeout_timer = gpui::Timer::after(timeout);
    tokio::pin!(timeout_timer);

    loop {
        let snapshot = cx
            .update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(());
                };
                Ok::<_, ()>(root.read(cx).ui_snapshot(cx))
            })
            .ok()
            .and_then(Result::ok)
            .ok_or_else(|| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))?;

        if snapshot_matches_predicate(&snapshot, &req.predicate) {
            return Ok(snapshot);
        }

        tokio::select! {
            changed = updates_rx.changed() => {
                if changed.is_err() {
                    return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "UI update channel closed."));
                }
            }
            changed = idle_rx.changed() => {
                if changed.is_err() {
                    return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "UI idle channel closed."));
                }
            }
            _ = &mut timeout_timer => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Timed out waiting for UI snapshot predicate.",
                ));
            }
        }
    }
}

async fn wait_for_idle(
    root: &WeakEntity<RootView>,
    req: WaitForUiIdleRequest,
    cx: &AsyncApp,
) -> Result<(), ErrorEnvelope> {
    let timeout = if req.timeout_ms == 0 {
        Duration::from_millis(2_000)
    } else {
        Duration::from_millis(req.timeout_ms)
    };

    let quiescence = if req.quiescence_ms == 0 {
        Duration::from_millis(25)
    } else {
        Duration::from_millis(req.quiescence_ms)
    };

    let (mut updates_rx, idle_rx_opt) = cx
        .update(|cx| {
            let Some(root) = root.upgrade() else {
                return Err(());
            };

            Ok::<_, ()>((
                root.read(cx).subscribe_ui_updates(),
                cx.try_global::<UiContext>()
                    .map(|ui| ui.idle_tracker().subscribe()),
            ))
        })
        .ok()
        .and_then(Result::ok)
        .ok_or_else(|| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))?;

    let (_idle_dummy_tx, idle_dummy_rx) = watch::channel(0_usize);
    let mut idle_rx = idle_rx_opt.unwrap_or(idle_dummy_rx);

    let timeout_timer = gpui::Timer::after(timeout);
    tokio::pin!(timeout_timer);

    loop {
        let snapshot = cx
            .update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(());
                };
                Ok::<_, ()>(root.read(cx).ui_snapshot(cx))
            })
            .ok()
            .and_then(Result::ok)
            .ok_or_else(|| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))?;

        let active_transitions = *idle_rx.borrow();
        let is_idle = snapshot.in_flight.is_empty() && active_transitions == 0;

        if is_idle {
            let quiescence_timer = gpui::Timer::after(quiescence);
            tokio::pin!(quiescence_timer);

            tokio::select! {
                changed = updates_rx.changed() => {
                    if changed.is_err() {
                        return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "UI update channel closed."));
                    }
                    continue;
                }
                changed = idle_rx.changed() => {
                    if changed.is_err() {
                        return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "UI idle channel closed."));
                    }
                    continue;
                }
                _ = &mut quiescence_timer => return Ok(()),
                _ = &mut timeout_timer => {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "Timed out waiting for UI idle.",
                    ));
                }
            }
        }

        tokio::select! {
            changed = updates_rx.changed() => {
                if changed.is_err() {
                    return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "UI update channel closed."));
                }
            }
            changed = idle_rx.changed() => {
                if changed.is_err() {
                    return Err(ErrorEnvelope::new(ErrorCategory::Unavailable, "UI idle channel closed."));
                }
            }
            _ = &mut timeout_timer => {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "Timed out waiting for UI idle.",
                ));
            }
        }
    }
}

async fn handle_ui_driver_request(
    root: &WeakEntity<RootView>,
    payload: UiDriverRequestPayload,
    cx: &AsyncApp,
) -> UiDriverResponseResult {
    match payload {
        UiDriverRequestPayload::GetSnapshot(_) => {
            let snapshot = cx
                .update(|cx| {
                    let Some(root) = root.upgrade() else {
                        return Err(());
                    };
                    Ok::<_, ()>(root.read(cx).ui_snapshot(cx))
                })
                .ok()
                .and_then(Result::ok)
                .unwrap_or_else(ui_unavailable_snapshot);

            UiDriverResponseResult::GetSnapshot(
                redesmyn_protocol::ui_driver::GetUiSnapshotResponse { snapshot },
            )
        }
        UiDriverRequestPayload::OpenEpic(req) => {
            let slug = req.epic_slug;
            let ok = cx
                .update(|cx| {
                    let Some(root) = root.upgrade() else {
                        return Err(());
                    };
                    root.update(cx, |this, cx| this.select_epic(slug.clone(), cx));
                    Ok::<_, ()>(())
                })
                .ok()
                .and_then(Result::ok)
                .is_some();

            if ok {
                UiDriverResponseResult::OpenEpic(redesmyn_protocol::ui_driver::OpenEpicResponse {})
            } else {
                UiDriverResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ))
            }
        }
        UiDriverRequestPayload::SetLeftPaneCollapsed(req) => {
            let collapsed = req.collapsed;
            let ok = cx
                .update(|cx| {
                    let Some(root) = root.upgrade() else {
                        return Err(());
                    };
                    root.update(cx, |this, cx| this.set_left_pane_collapsed(collapsed, cx));
                    Ok::<_, ()>(())
                })
                .ok()
                .and_then(Result::ok)
                .is_some();

            if ok {
                UiDriverResponseResult::SetLeftPaneCollapsed(
                    redesmyn_protocol::ui_driver::SetLeftPaneCollapsedResponse {},
                )
            } else {
                UiDriverResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ))
            }
        }
        UiDriverRequestPayload::TriggerRefresh(_req) => {
            let ok = cx
                .update(|cx| {
                    let Some(root) = root.upgrade() else {
                        return Err(());
                    };
                    root.update(cx, |this, cx| this.start_refresh(RefreshReason::Manual, cx));
                    Ok::<_, ()>(())
                })
                .ok()
                .and_then(Result::ok)
                .is_some();

            if ok {
                UiDriverResponseResult::TriggerRefresh(TriggerRefreshResponse { command_id: None })
            } else {
                UiDriverResponseResult::Error(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ))
            }
        }
        UiDriverRequestPayload::CreateChatSession(req) => {
            let result: Result<CreateChatSessionResponse, ErrorEnvelope> =
                create_chat_session_via_control_plane(root, req.name_hint, cx).await;
            match result {
                Ok(resp) => UiDriverResponseResult::CreateChatSession(resp),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::CloseChatSession(req) => {
            let result: Result<(), ErrorEnvelope> =
                close_chat_session_via_control_plane(root, req.session_id, cx).await;
            match result {
                Ok(()) => UiDriverResponseResult::CloseChatSession(
                    redesmyn_protocol::ui_driver::CloseChatSessionResponse {},
                ),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::PinChatSession(req) => {
            let result: Result<(), ErrorEnvelope> =
                pin_chat_session_via_control_plane(root, req.session_id, cx).await;
            match result {
                Ok(()) => UiDriverResponseResult::PinChatSession(
                    redesmyn_protocol::ui_driver::PinChatSessionResponse {},
                ),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::UnpinChatSession(_req) => {
            let result: Result<(), ErrorEnvelope> =
                unpin_chat_session_via_control_plane(root, cx).await;
            match result {
                Ok(()) => UiDriverResponseResult::UnpinChatSession(
                    redesmyn_protocol::ui_driver::UnpinChatSessionResponse {},
                ),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::CaptureScreenshot(req) => {
            let result: Result<CaptureScreenshotResponse, ErrorEnvelope> = async {
                let label = crate::test_artifacts::sanitize_label(&req.name_hint);
                let include_decorations = req.include_decorations.unwrap_or(false);
                let window = req.window;

                let snapshot = cx
                    .update(|cx| {
                        let Some(root) = root.upgrade() else {
                            return Err(ErrorEnvelope::new(
                                ErrorCategory::Unavailable,
                                "UI is unavailable.",
                            ));
                        };
                        Ok::<_, ErrorEnvelope>(root.read(cx).ui_snapshot(cx))
                    })
                    .map_err(|_| {
                        ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable.")
                    })??;

                let target = cx
                    .update(|cx| crate::screenshot::select_target(window, cx))
                    .map_err(|_| {
                        ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable.")
                    })??;

                let Some(artifacts) = crate::test_artifacts::ui_test_artifacts() else {
                    let png_path = crate::test_artifacts::temp_png_path(&label);
                    let task = gpui::background_executor().spawn(async move {
                        crate::test_artifacts::ensure_parent_dir(&png_path).map_err(|err| {
                            ErrorEnvelope::new(
                                ErrorCategory::Unavailable,
                                format!("Failed to create screenshot directory: {err}"),
                            )
                        })?;
                        crate::screenshot::capture_png(target, include_decorations, &png_path)?;
                        let bytes = std::fs::read(&png_path).map_err(|err| {
                            ErrorEnvelope::new(
                                ErrorCategory::Unavailable,
                                format!("Failed to read screenshot data: {err}"),
                            )
                        })?;
                        let _ = std::fs::remove_file(&png_path);
                        Ok::<_, ErrorEnvelope>(bytes)
                    });

                    let png_data = task.await?;
                    return Ok(CaptureScreenshotResponse {
                        png_path: String::new(),
                        png_data,
                    });
                };

                let ui_snapshot_path = artifacts.ui_snapshot_path(&label);
                let screenshot_path = artifacts.screenshot_path(&label);
                let screenshot_path_str = screenshot_path.to_string_lossy().to_string();

                let task = gpui::background_executor().spawn(async move {
                    crate::test_artifacts::write_json_pretty(&ui_snapshot_path, &snapshot)
                        .map_err(|err| {
                            ErrorEnvelope::new(
                                ErrorCategory::Unavailable,
                                format!("Failed to write UI snapshot: {err}"),
                            )
                        })?;
                    crate::test_artifacts::ensure_parent_dir(&screenshot_path).map_err(|err| {
                        ErrorEnvelope::new(
                            ErrorCategory::Unavailable,
                            format!("Failed to create screenshot directory: {err}"),
                        )
                    })?;
                    crate::screenshot::capture_png(target, include_decorations, &screenshot_path)?;
                    Ok::<_, ErrorEnvelope>(())
                });

                task.await?;

                Ok(CaptureScreenshotResponse {
                    png_path: screenshot_path_str,
                    png_data: Vec::new(),
                })
            }
            .await;

            match result {
                Ok(resp) => UiDriverResponseResult::CaptureScreenshot(resp),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::WaitForSnapshot(req) => {
            match wait_for_snapshot(root, req, cx).await {
                Ok(snapshot) => {
                    UiDriverResponseResult::WaitForSnapshot(WaitForUiSnapshotResponse { snapshot })
                }
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::WaitForIdle(req) => match wait_for_idle(root, req, cx).await {
            Ok(()) => UiDriverResponseResult::WaitForIdle(WaitForUiIdleResponse {}),
            Err(err) => UiDriverResponseResult::Error(err),
        },
        UiDriverRequestPayload::GraphSelectNode(req) => {
            let task_id = req.task_id;
            let span =
                redesmyn_logging::redesmyn_info_span!("ui_driver.graph_select_node", %task_id);
            let _guard = span.enter();

            let result: Result<(), ErrorEnvelope> = match cx.update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                };

                let graph_view = {
                    let root_ref = root.read(cx);
                    root_ref.workspace_pane.read(cx).graph_view.clone()
                };

                if !graph_view.read(cx).contains_task_node(task_id) {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        format!("Graph does not contain task node {task_id}."),
                    ));
                }

                graph_view.update(cx, |this, cx| {
                    this.driver_select_node_by_task_id(task_id, cx)
                });

                root.update(cx, |this, cx| this.notify_ui_updated(cx));

                Ok(())
            }) {
                Ok(result) => result,
                Err(_) => {
                    return UiDriverResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                }
            };

            match result {
                Ok(()) => UiDriverResponseResult::GraphSelectNode(SelectGraphNodeResponse {}),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::GraphClearSelection(_req) => {
            let span = redesmyn_logging::redesmyn_info_span!("ui_driver.graph_clear_selection");
            let _guard = span.enter();

            let result: Result<(), ErrorEnvelope> = match cx.update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                };

                let graph_view = {
                    let root_ref = root.read(cx);
                    root_ref.workspace_pane.read(cx).graph_view.clone()
                };

                graph_view.update(cx, |this, cx| this.driver_clear_selection(cx));
                root.update(cx, |this, cx| this.notify_ui_updated(cx));

                Ok(())
            }) {
                Ok(result) => result,
                Err(_) => {
                    return UiDriverResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                }
            };

            match result {
                Ok(()) => UiDriverResponseResult::GraphClearSelection(ClearGraphSelectionResponse {}),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::GraphToggleExpandedTaskCard(req) => {
            let task_id = req.task_id;
            let span = redesmyn_logging::redesmyn_info_span!(
                "ui_driver.graph_toggle_expanded_task_card",
                %task_id
            );
            let _guard = span.enter();

            let result: Result<(), ErrorEnvelope> = match cx.update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                };

                let graph_view = {
                    let root_ref = root.read(cx);
                    root_ref.workspace_pane.read(cx).graph_view.clone()
                };

                if !graph_view.read(cx).contains_task_node(task_id) {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        format!("Graph does not contain task node {task_id}."),
                    ));
                }

                graph_view.update(cx, |this, cx| {
                    this.driver_toggle_expanded_task_card(task_id, cx)
                });

                root.update(cx, |this, cx| this.notify_ui_updated(cx));

                Ok(())
            }) {
                Ok(result) => result,
                Err(_) => {
                    return UiDriverResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                }
            };

            match result {
                Ok(()) => UiDriverResponseResult::GraphToggleExpandedTaskCard(
                    ToggleExpandedTaskCardResponse {},
                ),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::GraphMultiSelectAddNode(req) => {
            let task_id = req.task_id;
            let span = redesmyn_logging::redesmyn_info_span!(
                "ui_driver.graph_multi_select_add_node",
                %task_id
            );
            let _guard = span.enter();

            let result: Result<(), ErrorEnvelope> = match cx.update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                };

                let graph_view = {
                    let root_ref = root.read(cx);
                    root_ref.workspace_pane.read(cx).graph_view.clone()
                };

                if !graph_view.read(cx).contains_task_node(task_id) {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        format!("Graph does not contain task node {task_id}."),
                    ));
                }

                graph_view.update(cx, |this, cx| this.driver_multi_select_add_node(task_id, cx));
                root.update(cx, |this, cx| this.notify_ui_updated(cx));

                Ok(())
            }) {
                Ok(result) => result,
                Err(_) => {
                    return UiDriverResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                }
            };

            match result {
                Ok(()) => UiDriverResponseResult::GraphMultiSelectAddNode(MultiSelectAddNodeResponse {}),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        UiDriverRequestPayload::GraphMultiSelectRemoveNode(req) => {
            let task_id = req.task_id;
            let span = redesmyn_logging::redesmyn_info_span!(
                "ui_driver.graph_multi_select_remove_node",
                %task_id
            );
            let _guard = span.enter();

            let result: Result<(), ErrorEnvelope> = match cx.update(|cx| {
                let Some(root) = root.upgrade() else {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                };

                let graph_view = {
                    let root_ref = root.read(cx);
                    root_ref.workspace_pane.read(cx).graph_view.clone()
                };

                if !graph_view.read(cx).contains_task_node(task_id) {
                    return Err(ErrorEnvelope::new(
                        ErrorCategory::InvalidRequest,
                        format!("Graph does not contain task node {task_id}."),
                    ));
                }

                graph_view.update(cx, |this, cx| {
                    this.driver_multi_select_remove_node(task_id, cx)
                });
                root.update(cx, |this, cx| this.notify_ui_updated(cx));

                Ok(())
            }) {
                Ok(result) => result,
                Err(_) => {
                    return UiDriverResponseResult::Error(ErrorEnvelope::new(
                        ErrorCategory::Unavailable,
                        "UI is unavailable.",
                    ));
                }
            };

            match result {
                Ok(()) => UiDriverResponseResult::GraphMultiSelectRemoveNode(
                    MultiSelectRemoveNodeResponse {},
                ),
                Err(err) => UiDriverResponseResult::Error(err),
            }
        }
        other => UiDriverResponseResult::Error(ErrorEnvelope::new(
            ErrorCategory::InvalidRequest,
            format!("unimplemented ui driver request: {other:?}"),
        )),
    }
}

fn control_plane_error_to_envelope(
    err: crate::control_plane_client::ControlPlaneClientError,
) -> ErrorEnvelope {
    ErrorEnvelope::new(
        ErrorCategory::Unavailable,
        format!("Control plane request failed: {err}"),
    )
}

fn ui_driver_fail<T>(
    root: &WeakEntity<RootView>,
    cx: &AsyncApp,
    err: ErrorEnvelope,
) -> Result<T, ErrorEnvelope> {
    let root = root.clone();
    let message = err.message.clone();
    let _ = cx.update(move |cx| {
        if let Some(root) = root.upgrade() {
            root.update(cx, move |this, cx| {
                this.ui_driver_action.fail(message);
                this.ui_driver_action_label = None;
                this.notify_ui_updated(cx);
            });
        }
    });
    Err(err)
}

fn selected_epic_slug_or_fail(
    root: &WeakEntity<RootView>,
    cx: &AsyncApp,
    epic_slug: Option<String>,
    message: &'static str,
) -> Result<String, ErrorEnvelope> {
    let Some(epic_slug) = epic_slug.filter(|slug| !slug.trim().is_empty()) else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(ErrorCategory::InvalidRequest, message),
        );
    };
    Ok(epic_slug)
}

fn repo_scope_from_graph_or_fail(
    root: &WeakEntity<RootView>,
    cx: &AsyncApp,
    graph: &redesmyn_protocol::client::EpicGraph,
) -> Result<RepoScope, ErrorEnvelope> {
    let Some(workspace_id) = graph.workspace_id else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Workspace id unavailable for selected epic.",
            ),
        );
    };
    let Some(repo_id) = graph.repo_id else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Repo id unavailable for selected epic.",
            ),
        );
    };
    Ok(RepoScope::new(workspace_id, repo_id))
}

fn epic_id_from_graph_or_fail(
    root: &WeakEntity<RootView>,
    cx: &AsyncApp,
    graph: &redesmyn_protocol::client::EpicGraph,
) -> Result<redesmyn_ids::EpicId, ErrorEnvelope> {
    let Some(epic_id) = graph.epic_id else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Epic id unavailable for selected epic.",
            ),
        );
    };
    Ok(epic_id)
}

async fn create_chat_session_via_control_plane(
    root: &WeakEntity<RootView>,
    name_hint: String,
    cx: &AsyncApp,
) -> Result<CreateChatSessionResponse, ErrorEnvelope> {
    let title = name_hint.trim();
    let title = (!title.is_empty()).then_some(title.to_string());

    let (client, epic_slug) = cx
        .update(|cx| {
            let Some(root) = root.upgrade() else {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ));
            };

            let (client, epic_slug) = {
                let this = root.read(cx);
                (
                    this.model.read(cx).chrome_control_plane_client.clone(),
                    this.chrome.selected_epic_slug.clone(),
                )
            };

            root.update(cx, |this, cx| {
                this.ui_driver_action.start();
                this.ui_driver_action_label = Some("Creating chat…".into());
                this.notify_ui_updated(cx);
            });

            Ok::<_, ErrorEnvelope>((client, epic_slug))
        })
        .map_err(|_| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))??;

    let Some(client) = client else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Control plane client unavailable (not embedded).",
            ),
        );
    };

    let epic_slug = selected_epic_slug_or_fail(
        root,
        cx,
        epic_slug,
        "Select an epic before creating a chat session.",
    )?;

    let graph = match client.get_epic_graph(epic_slug).await {
        Ok(graph) => graph,
        Err(err) => return ui_driver_fail(root, cx, control_plane_error_to_envelope(err)),
    };
    let scope = repo_scope_from_graph_or_fail(root, cx, &graph)?;

    let resp = match client.create_chat_session(scope, title).await {
        Ok(resp) => resp,
        Err(err) => return ui_driver_fail(root, cx, control_plane_error_to_envelope(err)),
    };

    let _ = cx.update(|cx| {
        if let Some(root) = root.upgrade() {
            root.update(cx, |this, cx| {
                this.ui_driver_action.succeed();
                this.ui_driver_action_label = None;
                this.notify_ui_updated(cx);
            });
        }
    });

    Ok(CreateChatSessionResponse {
        session_id: resp.session_id,
    })
}

async fn close_chat_session_via_control_plane(
    root: &WeakEntity<RootView>,
    session_id: redesmyn_ids::SessionId,
    cx: &AsyncApp,
) -> Result<(), ErrorEnvelope> {
    let (client, epic_slug) = cx
        .update(|cx| {
            let Some(root) = root.upgrade() else {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ));
            };

            let (client, epic_slug) = {
                let this = root.read(cx);
                (
                    this.model.read(cx).chrome_control_plane_client.clone(),
                    this.chrome.selected_epic_slug.clone(),
                )
            };

            root.update(cx, |this, cx| {
                this.ui_driver_action.start();
                this.ui_driver_action_label = Some("Closing chat…".into());
                this.notify_ui_updated(cx);
            });

            Ok::<_, ErrorEnvelope>((client, epic_slug))
        })
        .map_err(|_| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))??;

    let Some(client) = client else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Control plane client unavailable (not embedded).",
            ),
        );
    };

    let epic_slug = selected_epic_slug_or_fail(
        root,
        cx,
        epic_slug,
        "Select an epic before closing a chat session.",
    )?;

    let graph = match client.get_epic_graph(epic_slug).await {
        Ok(graph) => graph,
        Err(err) => return ui_driver_fail(root, cx, control_plane_error_to_envelope(err)),
    };
    let scope = repo_scope_from_graph_or_fail(root, cx, &graph)?;

    if let Err(err) = client.close_chat_session(scope, session_id).await {
        return ui_driver_fail(root, cx, control_plane_error_to_envelope(err));
    }

    let _ = cx.update(|cx| {
        if let Some(root) = root.upgrade() {
            root.update(cx, |this, cx| {
                this.ui_driver_action.succeed();
                this.ui_driver_action_label = None;
                this.notify_ui_updated(cx);
            });
        }
    });

    Ok(())
}

async fn pin_chat_session_via_control_plane(
    root: &WeakEntity<RootView>,
    session_id: redesmyn_ids::SessionId,
    cx: &AsyncApp,
) -> Result<(), ErrorEnvelope> {
    let (client, epic_slug) = cx
        .update(|cx| {
            let Some(root) = root.upgrade() else {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ));
            };

            let (client, epic_slug) = {
                let this = root.read(cx);
                (
                    this.model.read(cx).chrome_control_plane_client.clone(),
                    this.chrome.selected_epic_slug.clone(),
                )
            };

            root.update(cx, |this, cx| {
                this.ui_driver_action.start();
                this.ui_driver_action_label = Some("Pinning chat…".into());
                this.notify_ui_updated(cx);
            });

            Ok::<_, ErrorEnvelope>((client, epic_slug))
        })
        .map_err(|_| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))??;

    let Some(client) = client else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Control plane client unavailable (not embedded).",
            ),
        );
    };

    let epic_slug =
        selected_epic_slug_or_fail(root, cx, epic_slug, "Select an epic before pinning.")?;

    let graph = match client.get_epic_graph(epic_slug).await {
        Ok(graph) => graph,
        Err(err) => return ui_driver_fail(root, cx, control_plane_error_to_envelope(err)),
    };
    let scope = repo_scope_from_graph_or_fail(root, cx, &graph)?;
    let epic_id = epic_id_from_graph_or_fail(root, cx, &graph)?;

    if let Err(err) = client
        .pin_chat_session_to_epic(scope, epic_id, session_id)
        .await
    {
        return ui_driver_fail(root, cx, control_plane_error_to_envelope(err));
    }

    let _ = cx.update(|cx| {
        if let Some(root) = root.upgrade() {
            root.update(cx, |this, cx| {
                this.ui_driver_action.succeed();
                this.ui_driver_action_label = None;
                this.notify_ui_updated(cx);
            });
        }
    });

    Ok(())
}

async fn unpin_chat_session_via_control_plane(
    root: &WeakEntity<RootView>,
    cx: &AsyncApp,
) -> Result<(), ErrorEnvelope> {
    let (client, epic_slug) = cx
        .update(|cx| {
            let Some(root) = root.upgrade() else {
                return Err(ErrorEnvelope::new(
                    ErrorCategory::Unavailable,
                    "UI is unavailable.",
                ));
            };

            let (client, epic_slug) = {
                let this = root.read(cx);
                (
                    this.model.read(cx).chrome_control_plane_client.clone(),
                    this.chrome.selected_epic_slug.clone(),
                )
            };

            root.update(cx, |this, cx| {
                this.ui_driver_action.start();
                this.ui_driver_action_label = Some("Unpinning chat…".into());
                this.notify_ui_updated(cx);
            });

            Ok::<_, ErrorEnvelope>((client, epic_slug))
        })
        .map_err(|_| ErrorEnvelope::new(ErrorCategory::Unavailable, "UI is unavailable."))??;

    let Some(client) = client else {
        return ui_driver_fail(
            root,
            cx,
            ErrorEnvelope::new(
                ErrorCategory::Unavailable,
                "Control plane client unavailable (not embedded).",
            ),
        );
    };

    let epic_slug =
        selected_epic_slug_or_fail(root, cx, epic_slug, "Select an epic before unpinning.")?;

    let graph = match client.get_epic_graph(epic_slug).await {
        Ok(graph) => graph,
        Err(err) => return ui_driver_fail(root, cx, control_plane_error_to_envelope(err)),
    };
    let scope = repo_scope_from_graph_or_fail(root, cx, &graph)?;
    let epic_id = epic_id_from_graph_or_fail(root, cx, &graph)?;

    if let Err(err) = client.unpin_chat_session_from_epic(scope, epic_id).await {
        return ui_driver_fail(root, cx, control_plane_error_to_envelope(err));
    }

    let _ = cx.update(|cx| {
        if let Some(root) = root.upgrade() {
            root.update(cx, |this, cx| {
                this.ui_driver_action.succeed();
                this.ui_driver_action_label = None;
                this.notify_ui_updated(cx);
            });
        }
    });

    Ok(())
}

impl Focusable for RootView {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl Render for RootView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        crate::vsync::ensure_vsync(window);
        let theme = theme_for_window(window, cx);

        if !self.chrome.did_startup_refresh {
            self.chrome.did_startup_refresh = true;
            self.start_refresh(RefreshReason::Startup, cx);
        }

        let model = self.model.read(cx);
        let split_state = self.split_pane.read(cx).state();
        let toggle_icon = if split_state.collapsed { "⟩" } else { "⟨" };
        let toggle_tooltip = if split_state.collapsed {
            "Show sessions pane"
        } else {
            "Hide sessions pane"
        };

        let root = cx.entity();

        let selected_epic = self
            .chrome
            .selected_epic_slug
            .as_deref()
            .and_then(|slug| self.chrome.epics.iter().find(|epic| epic.slug == slug))
            .cloned();
        let selected_epic_name = selected_epic.as_ref().map(|epic| epic.name.clone().into());

        let epic_button_label: SharedString = match selected_epic.as_ref() {
            Some(epic) => epic.slug.clone().into(),
            None if self.chrome.epics.is_empty() => "No epics".into(),
            None => "Select epic".into(),
        };

        let epic_button_disabled = self.chrome.epics.is_empty();
        let epic_button_disabled_reason = if self.chrome.refresh.in_flight {
            "Loading epics…"
        } else {
            "No epics available"
        };

        let preference = cx
            .try_global::<UiContext>()
            .map(|ui| ui.theme_preference())
            .unwrap_or(ThemePreference::System);

        let theme_pref_button = |id, label: &'static str, pref| {
            let kind = if preference == pref {
                ButtonKind::Primary
            } else {
                ButtonKind::Secondary
            };

            TextButton::new((id, cx.entity_id()), label)
                .kind(kind)
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.set_theme_preference(pref, cx));
                    }
                })
        };

        let status_dot_color = if self.chrome.control_plane_status.is_some() {
            theme.colors.ring
        } else if self.chrome.refresh.in_flight {
            theme.colors.border.opacity(0.7)
        } else {
            theme.colors.danger
        };

        let control_plane_label = if self.chrome.control_plane_status.is_some() {
            "Control plane: running"
        } else if self.chrome.refresh.in_flight {
            "Control plane: checking…"
        } else if model.chrome_control_plane_client.is_some()
            || model.config.desktop.embed_control_plane
        {
            "Control plane: unavailable"
        } else {
            "Control plane: external"
        };

        let daemon_label: SharedString = if model.config.desktop.embed_daemon {
            match model.daemon_host_id {
                Some(id) => format!("Daemon: embedded · host {id}").into(),
                None => "Daemon: embedded".into(),
            }
        } else {
            "Daemon: external (start: rn daemon run)".into()
        };

        let refresh_button = IconButton::new(("chrome_refresh", cx.entity_id()), div().child("↻"))
            .tooltip("Refresh")
            .disabled(self.chrome.refresh.in_flight)
            .disabled_reason("Refreshing…")
            .on_click({
                let root = root.clone();
                move |_, _, cx| {
                    root.update(cx, |this, cx| this.start_refresh(RefreshReason::Manual, cx));
                }
            });

        let settings_button =
            IconButton::new(("chrome_settings", cx.entity_id()), div().child("⚙"))
                .tooltip("Settings")
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.toggle_panel(ChromePanel::Settings, cx));
                    }
                });

        let theme_toggle_button =
            IconButton::new(("chrome_theme_toggle", cx.entity_id()), div().child("◐"))
                .tooltip("Toggle theme")
                .on_click({
                    let root = root.clone();
                    move |_, window, cx| {
                        let mode = theme_for_window(window, cx).mode;
                        let next = match mode {
                            redesmyn_ui::styles::ThemeMode::Dark => ThemePreference::Light,
                            redesmyn_ui::styles::ThemeMode::Light => ThemePreference::Dark,
                        };
                        root.update(cx, |this, cx| this.set_theme_preference(next, cx));
                    }
                });

        let epic_button =
            TextButton::new(("chrome_epic_selector", cx.entity_id()), epic_button_label)
                .kind(ButtonKind::Ghost)
                .disabled(epic_button_disabled)
                .disabled_reason(epic_button_disabled_reason)
                .trailing(div().child("▾"))
                .on_click({
                    let root = root.clone();
                    move |_, _, cx| {
                        root.update(cx, |this, cx| this.toggle_panel(ChromePanel::EpicMenu, cx));
                    }
                });

        let header = div()
            .h(px(44.0))
            .px(theme.spacing.md)
            .flex()
            .flex_row()
            .items_center()
            .justify_between()
            .gap(theme.spacing.lg)
            .bg(theme.colors.surface_elevated)
            .border_b_1()
            .border_color(theme.colors.border.opacity(0.6))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .child(
                        IconButton::new(
                            ("chrome_toggle_sessions", cx.entity_id()),
                            div().child(toggle_icon),
                        )
                        .tooltip(toggle_tooltip)
                        .on_click({
                            let root = root.clone();
                            move |_, _, cx| {
                                root.update(cx, |this, cx| this.toggle_sessions_pane(cx));
                            }
                        }),
                    )
                    .child(epic_button)
                    .when_some(selected_epic_name, |this, name: SharedString| {
                        this.child(
                            div()
                                .text_sm()
                                .text_color(theme.colors.foreground_muted)
                                .child(name),
                        )
                    }),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .child(
                        div()
                            .flex()
                            .flex_row()
                            .items_center()
                            .gap(theme.spacing.xs)
                            .child(div().size(px(8.0)).rounded(px(999.0)).bg(status_dot_color))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(theme.colors.foreground_muted)
                                    .child(control_plane_label),
                            ),
                    )
                    .child(
                        div()
                            .w(px(1.0))
                            .h(px(18.0))
                            .bg(theme.colors.border.opacity(0.6)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child(daemon_label),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .when(self.chrome.refresh.in_flight, |this| {
                        this.child(ProgressPill::new("Refreshing"))
                    })
                    .child(refresh_button)
                    .child(theme_toggle_button)
                    .child(settings_button),
            );

        let mut chrome_extras = div().flex().flex_col().gap(theme.spacing.sm);
        let epic_menu_top = px(44.0) + theme.spacing.xs;
        let epic_menu_left = theme.spacing.md + px(28.0) + theme.spacing.sm;

        if let Some(error) = self.chrome.refresh.error.clone() {
            chrome_extras = chrome_extras.child(
                div().px(theme.spacing.md).child(
                    Callout::new(error)
                        .kind(CalloutKind::Danger)
                        .title("Refresh failed")
                        .action(
                            TextButton::new(("refresh_error_dismiss", cx.entity_id()), "Dismiss")
                                .kind(ButtonKind::Ghost)
                                .on_click({
                                    let root = root.clone();
                                    move |_, _, cx| {
                                        root.update(cx, |this, cx| this.clear_refresh_error(cx));
                                    }
                                }),
                        ),
                ),
            );
        }

        if let Some(error) = self.chrome.theme_error.clone() {
            chrome_extras = chrome_extras.child(
                div().px(theme.spacing.md).child(
                    Callout::new(error)
                        .kind(CalloutKind::Warning)
                        .title("Theme save failed")
                        .action(
                            TextButton::new(("theme_error_dismiss", cx.entity_id()), "Dismiss")
                                .kind(ButtonKind::Ghost)
                                .on_click({
                                    let root = root.clone();
                                    move |_, _, cx| {
                                        root.update(cx, |this, cx| this.clear_theme_error(cx));
                                    }
                                }),
                        ),
                ),
            );
        }

        let epic_menu_overlay = if matches!(self.chrome.panel, Some(ChromePanel::EpicMenu)) {
            let mut menu_body = div().flex().flex_col().gap(theme.spacing.sm).child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Select an epic"),
            );

            if self.chrome.epics.is_empty() {
                menu_body = menu_body.child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("No epics available. Ensure the control plane can discover epics."),
                );
            } else {
                let scroll = self.chrome.epic_scroll.clone();
                let root_for_items = root.clone();
                let selected_slug = self.chrome.selected_epic_slug.clone();
                let in_flight = self.chrome.refresh.in_flight;
                let entity_id = cx.entity_id();

                let list = ScrollArea::new(("chrome_epic_scroll", cx.entity_id()), scroll)
                    .scrollbar_width(px(8.0))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .w_full()
                            .gap(theme.spacing.xs)
                            .children(self.chrome.epics.iter().cloned().map(|epic| {
                                let epic_slug = epic.slug.clone();
                                let epic_name = epic.name.clone();
                                let selected = selected_slug.as_deref() == Some(epic_slug.as_str());
                                let kind = if selected {
                                    ButtonKind::Secondary
                                } else {
                                    ButtonKind::Ghost
                                };
                                let item_id = (
                                    gpui::ElementId::from(("chrome_epic_item", entity_id)),
                                    epic_slug.clone(),
                                );
                                TextButton::new(item_id, epic_slug.clone())
                                    .menu_item()
                                    .kind(kind)
                                    .disabled(in_flight)
                                    .disabled_reason("Refreshing…")
                                    .tooltip(epic_name)
                                    .on_click({
                                        let root = root_for_items.clone();
                                        let slug = epic_slug.clone();
                                        move |_, _, cx| {
                                            root.update(cx, |this, cx| {
                                                this.select_epic(slug.clone(), cx)
                                            });
                                        }
                                    })
                            })),
                    );

                menu_body =
                    menu_body.child(div().h(px(220.0)).w_full().overflow_hidden().child(list));
            }

            Some(
                div()
                    .absolute()
                    .inset_0()
                    .child(div().absolute().inset_0().occlude().on_mouse_down(
                        gpui::MouseButton::Left,
                        {
                            let root = root.clone();
                            move |_, _, cx| {
                                root.update(cx, |this, cx| this.close_panel(cx));
                                cx.stop_propagation();
                            }
                        },
                    ))
                    .child(
                        div()
                            .absolute()
                            .top(epic_menu_top)
                            .left(epic_menu_left)
                            .child(
                                div()
                                    .w(px(360.0))
                                    .rounded(theme.radius.md)
                                    .shadow_md()
                                    .occlude()
                                    .child(
                                        div()
                                            .w_full()
                                            .p(theme.spacing.md)
                                            .rounded(theme.radius.md)
                                            .bg(theme.colors.surface)
                                            .border_1()
                                            .border_color(theme.colors.border.opacity(0.5))
                                            .overflow_hidden()
                                            .child(menu_body),
                                    ),
                            ),
                    ),
            )
        } else {
            None
        };

        if matches!(self.chrome.panel, Some(ChromePanel::Settings)) {
            let settings_panel = div()
                .p(theme.spacing.md)
                .rounded(theme.radius.md)
                .bg(theme.colors.surface)
                .border_1()
                .border_color(theme.colors.border.opacity(0.5))
                .child(
                    div()
                        .flex()
                        .flex_row()
                        .items_center()
                        .justify_between()
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.colors.foreground)
                                .child("Settings"),
                        )
                        .child(
                            IconButton::new(("settings_close", cx.entity_id()), div().child("×"))
                                .tooltip("Close")
                                .on_click({
                                    let root = root.clone();
                                    move |_, _, cx| {
                                        root.update(cx, |this, cx| this.close_panel(cx));
                                    }
                                }),
                        ),
                )
                .child(
                    div()
                        .pt(theme.spacing.md)
                        .flex()
                        .gap(theme.spacing.xs)
                        .child(theme_pref_button(
                            "theme_light",
                            "Light",
                            ThemePreference::Light,
                        ))
                        .child(theme_pref_button(
                            "theme_dark",
                            "Dark",
                            ThemePreference::Dark,
                        ))
                        .child(theme_pref_button(
                            "theme_system",
                            "System",
                            ThemePreference::System,
                        )),
                )
                .child(
                    div()
                        .pt(theme.spacing.md)
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("More settings coming soon."),
                );

            chrome_extras = chrome_extras.child(
                div()
                    .px(theme.spacing.md)
                    .pb(theme.spacing.md)
                    .child(settings_panel),
            );
        }

        let mut root_container = div()
            .id(("desktop_root", cx.entity_id()))
            .relative()
            .key_context("Desktop")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::toggle_command_palette))
            .on_action(cx.listener(Self::close_command_palette))
            .on_action(cx.listener(Self::select_previous_command))
            .on_action(cx.listener(Self::select_next_command))
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
            .capture_key_down({
                let root = root.clone();
                move |event, _, cx| {
                    if event.keystroke.key != "escape" {
                        return;
                    }

                    let did_close = root.update(cx, |this, cx| {
                        let was_open = this.chrome.panel.is_some();
                        if was_open {
                            this.close_panel(cx);
                        }
                        was_open
                    });

                    if did_close {
                        cx.stop_propagation();
                    }
                }
            })
            .child(header)
            .child(chrome_extras)
            .child(div().flex_1().min_h(px(0.0)).child(self.split_pane.clone()));

        if let Some(epic_menu_overlay) = epic_menu_overlay {
            root_container = root_container.child(epic_menu_overlay);
        }

        if self.command_palette.is_open() {
            root_container = root_container.child(self.command_palette.render(window, cx));
        }

        root_container
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EpicSessionPanePanel {
    PinExisting,
}

struct EpicSessionPaneHost {
    session_view: Entity<SessionView>,
    control_plane_client: Option<ControlPlaneClient>,
    fixture: Option<SessionViewerFixtureEmitter>,
    fallback_session_id: Option<redesmyn_ids::SessionId>,
    selected_epic: Option<redesmyn_protocol::client::EpicSummary>,
    pinned_session_id: Option<redesmyn_ids::SessionId>,
    chat_sessions: Vec<redesmyn_protocol::client::AgentSessionSummary>,
    panel: Option<EpicSessionPanePanel>,
    scroll: ScrollHandle,
    load: UserActionState,
    create_and_pin: UserActionState,
    pin_existing: UserActionState,
    unpin: UserActionState,
    close_session: UserActionState,
    emit_demo_task: Option<Task<()>>,
    emit_demo_in_flight: bool,
    emit_demo_error: Option<SharedString>,
    load_task: Option<Task<()>>,
    action_task: Option<Task<()>>,
    selection_generation: u64,
}

impl EpicSessionPaneHost {
    fn new(model: Entity<DesktopModel>, cx: &mut Context<Self>) -> Self {
        let (session_client, fixture) = model.update(cx, |model, _cx| {
            (
                model.take_control_plane_client(),
                model.take_session_viewer_fixture(),
            )
        });
        let control_plane_client = model.read(cx).chrome_control_plane_client.clone();
        let fallback_session_id = fixture.as_ref().map(|fixture| fixture.session_id());
        let session_view = cx.new(|cx| SessionView::new(session_client, fallback_session_id, cx));
        Self {
            session_view,
            control_plane_client,
            fixture,
            fallback_session_id,
            selected_epic: None,
            pinned_session_id: None,
            chat_sessions: Vec::new(),
            panel: None,
            scroll: ScrollHandle::new(),
            load: UserActionState::default(),
            create_and_pin: UserActionState::default(),
            pin_existing: UserActionState::default(),
            unpin: UserActionState::default(),
            close_session: UserActionState::default(),
            emit_demo_task: None,
            emit_demo_in_flight: false,
            emit_demo_error: None,
            load_task: None,
            action_task: None,
            selection_generation: 0,
        }
    }

    fn set_selected_epic(
        &mut self,
        epic: Option<redesmyn_protocol::client::EpicSummary>,
        cx: &mut Context<Self>,
    ) {
        if self.selected_epic == epic {
            return;
        }

        self.selection_generation = self.selection_generation.wrapping_add(1);
        self.selected_epic = epic;
        self.pinned_session_id = None;
        self.chat_sessions.clear();
        self.panel = None;
        self.load_task = None;
        self.action_task = None;

        self.load.in_flight = false;
        self.load.clear_error();
        self.create_and_pin.in_flight = false;
        self.create_and_pin.clear_error();
        self.pin_existing.in_flight = false;
        self.pin_existing.clear_error();
        self.unpin.in_flight = false;
        self.unpin.clear_error();
        self.close_session.in_flight = false;
        self.close_session.clear_error();

        let session_id = if self.selected_epic.is_some() {
            None
        } else {
            self.fallback_session_id
        };
        self.session_view
            .update(cx, |view, cx| view.set_session_id(session_id, cx));

        if self.selected_epic.is_some() {
            self.refresh(cx);
        } else {
            cx.notify();
        }
    }

    fn is_current_selection(&self, generation: u64, epic_slug: &str) -> bool {
        if self.selection_generation != generation {
            return false;
        }
        matches!(
            self.selected_epic.as_ref(),
            Some(epic) if epic.slug == epic_slug
        )
    }

    fn emit_demo_message(
        &mut self,
        _event: &ClickEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.emit_demo_in_flight {
            return;
        }

        let Some(fixture) = self.fixture.clone() else {
            return;
        };

        let span = redesmyn_logging::redesmyn_info_span!(
            "ui.session_viewer_fixture.emit_demo_message",
            session_id = %fixture.session_id()
        );
        let _guard = span.enter();

        self.emit_demo_in_flight = true;
        self.emit_demo_error = None;
        cx.notify();

        self.emit_demo_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let tokio = fixture.tokio();
                let task = tokio.spawn({
                    let fixture = fixture.clone();
                    async move { fixture.emit_demo_message().await }
                });

                let error = match task.await {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error.to_string()),
                    Err(error) => Some(format!("Demo event task failed: {error}")),
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        this.emit_demo_task = None;
                        this.emit_demo_in_flight = false;
                        this.emit_demo_error = error.map(Into::into);
                        cx.notify();
                    })
                });
            }
        }));
    }

    fn refresh(&mut self, cx: &mut Context<Self>) {
        if self.load.in_flight {
            return;
        }

        let Some(client) = self.control_plane_client.clone() else {
            self.load.fail("Control plane client unavailable.");
            cx.notify();
            return;
        };

        let Some(selected_epic) = self.selected_epic.clone() else {
            return;
        };

        let generation = self.selection_generation;
        let epic_slug = selected_epic.slug.clone();
        let epic_slug_for_task = epic_slug.clone();

        self.load.start();
        cx.notify();

        let tokio = client.tokio().clone();
        self.load_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let task = tokio.spawn(async move {
                    let graph = client.get_epic_graph(epic_slug_for_task).await?;
                    let Some(workspace_id) = graph.workspace_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Workspace id unavailable for selected epic.".to_string(),
                        });
                    };
                    let Some(repo_id) = graph.repo_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Repo id unavailable for selected epic.".to_string(),
                        });
                    };

                    let scope = RepoScope::new(workspace_id, repo_id);
                    let epic_id = selected_epic.epic_id.or(graph.epic_id).ok_or_else(|| {
                        ControlPlaneClientError::Server {
                            message: "Epic id unavailable for selected epic.".to_string(),
                        }
                    })?;

                    let pinned_session_id =
                        client.get_epic_pinned_chat_session(scope, epic_id).await?;
                    let chat_sessions = client.list_chat_sessions(scope, true, 100).await?;

                    Ok::<_, ControlPlaneClientError>((pinned_session_id, chat_sessions))
                });

                let result = match task.await {
                    Ok(result) => result,
                    Err(error) => {
                        let _ = cx.update(|cx| {
                            entity.update(cx, |this, cx| {
                                if !this.is_current_selection(generation, &epic_slug) {
                                    return;
                                }

                                this.load_task = None;
                                this.load.fail(format!("Refresh failed: {error}"));
                                cx.notify();
                            })
                        });
                        return;
                    }
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        if !this.is_current_selection(generation, &epic_slug) {
                            return;
                        }

                        this.load_task = None;

                        match result {
                            Ok((pinned_session_id, chat_sessions)) => {
                                this.pinned_session_id = pinned_session_id;
                                this.chat_sessions = chat_sessions;
                                this.load.succeed();
                                this.session_view.update(cx, |view, cx| {
                                    view.set_session_id(pinned_session_id, cx);
                                });
                            }
                            Err(err) => this.load.fail(err.to_string()),
                        }

                        cx.notify();
                    })
                });
            }
        }));
    }

    fn create_and_pin_chat(&mut self, cx: &mut Context<Self>) {
        if self.create_and_pin.in_flight {
            return;
        }

        let Some(client) = self.control_plane_client.clone() else {
            self.create_and_pin
                .fail("Control plane client unavailable.");
            cx.notify();
            return;
        };

        let Some(selected_epic) = self.selected_epic.clone() else {
            self.create_and_pin
                .fail("Select an epic before creating a chat session.");
            cx.notify();
            return;
        };

        let generation = self.selection_generation;
        let epic_slug = selected_epic.slug.clone();
        let epic_slug_for_task = epic_slug.clone();

        self.create_and_pin.start();
        cx.notify();

        let tokio = client.tokio().clone();
        self.action_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let task = tokio.spawn(async move {
                    let graph = client.get_epic_graph(epic_slug_for_task).await?;
                    let Some(workspace_id) = graph.workspace_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Workspace id unavailable for selected epic.".to_string(),
                        });
                    };
                    let Some(repo_id) = graph.repo_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Repo id unavailable for selected epic.".to_string(),
                        });
                    };

                    let scope = RepoScope::new(workspace_id, repo_id);
                    let epic_id = selected_epic.epic_id.or(graph.epic_id).ok_or_else(|| {
                        ControlPlaneClientError::Server {
                            message: "Epic id unavailable for selected epic.".to_string(),
                        }
                    })?;

                    let resp = client.create_chat_session(scope, None).await?;
                    let session_id = resp.session_id;

                    client
                        .pin_chat_session_to_epic(scope, epic_id, session_id)
                        .await?;

                    let chat_sessions = client.list_chat_sessions(scope, true, 100).await?;

                    Ok::<_, ControlPlaneClientError>((session_id, chat_sessions))
                });

                let result = match task.await {
                    Ok(result) => result,
                    Err(error) => {
                        let _ = cx.update(|cx| {
                            entity.update(cx, |this, cx| {
                                if !this.is_current_selection(generation, &epic_slug) {
                                    return;
                                }

                                this.action_task = None;
                                this.create_and_pin.fail(format!("Create failed: {error}"));
                                cx.notify();
                            })
                        });
                        return;
                    }
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        if !this.is_current_selection(generation, &epic_slug) {
                            return;
                        }

                        this.action_task = None;

                        match result {
                            Ok((session_id, sessions)) => {
                                this.create_and_pin.succeed();
                                this.pinned_session_id = Some(session_id);
                                this.chat_sessions = sessions;
                                this.session_view.update(cx, |view, cx| {
                                    view.set_session_id(Some(session_id), cx);
                                    view.request_focus_composer(cx);
                                });
                            }
                            Err(err) => this.create_and_pin.fail(err.to_string()),
                        }

                        cx.notify();
                    })
                });
            }
        }));
    }

    fn open_pin_existing(&mut self, cx: &mut Context<Self>) {
        self.panel = Some(EpicSessionPanePanel::PinExisting);
        cx.notify();
    }

    fn close_panel(&mut self, cx: &mut Context<Self>) {
        if self.panel.take().is_some() {
            cx.notify();
        }
    }

    fn pin_existing_chat(&mut self, session_id: redesmyn_ids::SessionId, cx: &mut Context<Self>) {
        if self.pin_existing.in_flight {
            return;
        }

        let Some(client) = self.control_plane_client.clone() else {
            self.pin_existing.fail("Control plane client unavailable.");
            cx.notify();
            return;
        };

        let Some(selected_epic) = self.selected_epic.clone() else {
            self.pin_existing.fail("Select an epic before pinning.");
            cx.notify();
            return;
        };

        let generation = self.selection_generation;
        let epic_slug = selected_epic.slug.clone();
        let epic_slug_for_task = epic_slug.clone();

        self.pin_existing.start();
        cx.notify();

        let tokio = client.tokio().clone();
        self.action_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let task = tokio.spawn(async move {
                    let graph = client.get_epic_graph(epic_slug_for_task).await?;
                    let Some(workspace_id) = graph.workspace_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Workspace id unavailable for selected epic.".to_string(),
                        });
                    };
                    let Some(repo_id) = graph.repo_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Repo id unavailable for selected epic.".to_string(),
                        });
                    };

                    let scope = RepoScope::new(workspace_id, repo_id);
                    let epic_id = selected_epic.epic_id.or(graph.epic_id).ok_or_else(|| {
                        ControlPlaneClientError::Server {
                            message: "Epic id unavailable for selected epic.".to_string(),
                        }
                    })?;

                    client
                        .pin_chat_session_to_epic(scope, epic_id, session_id)
                        .await?;

                    let chat_sessions = client.list_chat_sessions(scope, true, 100).await?;

                    Ok::<_, ControlPlaneClientError>(chat_sessions)
                });

                let result = match task.await {
                    Ok(result) => result,
                    Err(error) => {
                        let _ = cx.update(|cx| {
                            entity.update(cx, |this, cx| {
                                if !this.is_current_selection(generation, &epic_slug) {
                                    return;
                                }

                                this.action_task = None;
                                this.pin_existing.fail(format!("Pin failed: {error}"));
                                cx.notify();
                            })
                        });
                        return;
                    }
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        if !this.is_current_selection(generation, &epic_slug) {
                            return;
                        }

                        this.action_task = None;

                        match result {
                            Ok(sessions) => {
                                this.pin_existing.succeed();
                                this.panel = None;
                                this.pinned_session_id = Some(session_id);
                                this.chat_sessions = sessions;
                                this.session_view.update(cx, |view, cx| {
                                    view.set_session_id(Some(session_id), cx);
                                    view.request_focus_composer(cx);
                                });
                            }
                            Err(err) => this.pin_existing.fail(err.to_string()),
                        }

                        cx.notify();
                    })
                });
            }
        }));
    }

    fn unpin_chat(&mut self, cx: &mut Context<Self>) {
        if self.unpin.in_flight {
            return;
        }

        let Some(client) = self.control_plane_client.clone() else {
            self.unpin.fail("Control plane client unavailable.");
            cx.notify();
            return;
        };

        let Some(selected_epic) = self.selected_epic.clone() else {
            self.unpin.fail("Select an epic before unpinning.");
            cx.notify();
            return;
        };

        let generation = self.selection_generation;
        let epic_slug = selected_epic.slug.clone();
        let epic_slug_for_task = epic_slug.clone();

        self.unpin.start();
        cx.notify();

        let tokio = client.tokio().clone();
        self.action_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let task = tokio.spawn(async move {
                    let graph = client.get_epic_graph(epic_slug_for_task).await?;
                    let Some(workspace_id) = graph.workspace_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Workspace id unavailable for selected epic.".to_string(),
                        });
                    };
                    let Some(repo_id) = graph.repo_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Repo id unavailable for selected epic.".to_string(),
                        });
                    };

                    let scope = RepoScope::new(workspace_id, repo_id);
                    let epic_id = selected_epic.epic_id.or(graph.epic_id).ok_or_else(|| {
                        ControlPlaneClientError::Server {
                            message: "Epic id unavailable for selected epic.".to_string(),
                        }
                    })?;

                    client.unpin_chat_session_from_epic(scope, epic_id).await?;

                    Ok::<_, ControlPlaneClientError>(())
                });

                let result = match task.await {
                    Ok(result) => result,
                    Err(error) => {
                        let _ = cx.update(|cx| {
                            entity.update(cx, |this, cx| {
                                if !this.is_current_selection(generation, &epic_slug) {
                                    return;
                                }

                                this.action_task = None;
                                this.unpin.fail(format!("Unpin failed: {error}"));
                                cx.notify();
                            })
                        });
                        return;
                    }
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        if !this.is_current_selection(generation, &epic_slug) {
                            return;
                        }

                        this.action_task = None;

                        match result {
                            Ok(()) => {
                                this.unpin.succeed();
                                this.pinned_session_id = None;
                                this.session_view.update(cx, |view, cx| {
                                    view.set_session_id(None, cx);
                                });
                            }
                            Err(err) => this.unpin.fail(err.to_string()),
                        }

                        cx.notify();
                    })
                });
            }
        }));
    }

    fn close_current_chat(&mut self, cx: &mut Context<Self>) {
        if self.close_session.in_flight {
            return;
        }

        let Some(client) = self.control_plane_client.clone() else {
            self.close_session.fail("Control plane client unavailable.");
            cx.notify();
            return;
        };

        let Some(selected_epic) = self.selected_epic.clone() else {
            self.close_session.fail("Select an epic before closing.");
            cx.notify();
            return;
        };

        let Some(session_id) = self.pinned_session_id else {
            return;
        };

        let generation = self.selection_generation;
        let epic_slug = selected_epic.slug.clone();
        let epic_slug_for_task = epic_slug.clone();

        self.close_session.start();
        cx.notify();

        let tokio = client.tokio().clone();
        self.action_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let task = tokio.spawn(async move {
                    let graph = client.get_epic_graph(epic_slug_for_task).await?;
                    let Some(workspace_id) = graph.workspace_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Workspace id unavailable for selected epic.".to_string(),
                        });
                    };
                    let Some(repo_id) = graph.repo_id else {
                        return Err(ControlPlaneClientError::Server {
                            message: "Repo id unavailable for selected epic.".to_string(),
                        });
                    };

                    let scope = RepoScope::new(workspace_id, repo_id);

                    client.close_chat_session(scope, session_id).await?;
                    let sessions = client.list_chat_sessions(scope, true, 100).await?;
                    Ok::<_, ControlPlaneClientError>(sessions)
                });

                let result = match task.await {
                    Ok(result) => result,
                    Err(error) => {
                        let _ = cx.update(|cx| {
                            entity.update(cx, |this, cx| {
                                if !this.is_current_selection(generation, &epic_slug) {
                                    return;
                                }

                                this.action_task = None;
                                this.close_session.fail(format!("Close failed: {error}"));
                                cx.notify();
                            })
                        });
                        return;
                    }
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        if !this.is_current_selection(generation, &epic_slug) {
                            return;
                        }

                        this.action_task = None;

                        match result {
                            Ok(sessions) => {
                                this.close_session.succeed();
                                this.chat_sessions = sessions;
                            }
                            Err(err) => this.close_session.fail(err.to_string()),
                        }

                        cx.notify();
                    })
                });
            }
        }));
    }

    fn pinned_chat_summary(&self) -> Option<&redesmyn_protocol::client::AgentSessionSummary> {
        let pinned = self.pinned_session_id?;
        self.chat_sessions
            .iter()
            .find(|session| session.session_id == pinned)
    }
}

impl Render for EpicSessionPaneHost {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);
        let view = cx.entity();

        let header_title = match self.selected_epic.as_ref() {
            None => "Chat".to_string(),
            Some(epic) => match self.pinned_session_id {
                None => epic.slug.clone(),
                Some(_) => self
                    .pinned_chat_summary()
                    .and_then(|summary| summary.title.clone())
                    .or_else(|| self.pinned_session_id.map(|id| id.to_string()))
                    .unwrap_or_else(|| "Pinned chat".to_string()),
            },
        };

        let in_flight_label = if self.load.in_flight {
            Some("Loading chat")
        } else if self.create_and_pin.in_flight {
            Some("Creating chat")
        } else if self.pin_existing.in_flight {
            Some("Pinning chat")
        } else if self.unpin.in_flight {
            Some("Unpinning")
        } else if self.close_session.in_flight {
            Some("Closing")
        } else {
            None
        };

        let mut header_actions = div().flex().flex_row().items_center().gap(theme.spacing.xs);

        if let Some(label) = in_flight_label {
            header_actions = header_actions.child(ProgressPill::new(label));
        }

        let can_act_on_epic = self.selected_epic.is_some() && !self.load.in_flight;
        let has_pinned = self.pinned_session_id.is_some();

        let create_button = IconButton::new(
            ("chat_create", cx.entity_id()),
            div()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .child("+"),
        )
        .tooltip("Create chat")
        .disabled(!can_act_on_epic || self.create_and_pin.in_flight)
        .disabled_reason("Loading…")
        .on_click({
            let view = view.clone();
            move |_, _, cx| view.update(cx, |this, cx| this.create_and_pin_chat(cx))
        });

        let pin_existing_button = IconButton::new(
            ("chat_pin_existing", cx.entity_id()),
            div()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .child("☰"),
        )
        .tooltip("Pin existing chat…")
        .disabled(!can_act_on_epic || self.pin_existing.in_flight)
        .disabled_reason("Loading…")
        .on_click({
            let view = view.clone();
            move |_, _, cx| view.update(cx, |this, cx| this.open_pin_existing(cx))
        });

        let pinned_button = IconButton::new(
            ("chat_unpin", cx.entity_id()),
            div()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .child("★"),
        )
        .tooltip("Unpin chat")
        .active(true)
        .disabled(!can_act_on_epic || !has_pinned || self.unpin.in_flight)
        .disabled_reason("No pinned chat")
        .on_click({
            let view = view.clone();
            move |_, _, cx| view.update(cx, |this, cx| this.unpin_chat(cx))
        });

        header_actions = header_actions
            .child(create_button)
            .child(pin_existing_button)
            .when(has_pinned, |this| this.child(pinned_button));
        if self.fixture.is_some() {
            let label = if self.emit_demo_in_flight {
                "Emitting…"
            } else {
                "Emit demo message"
            };
            header_actions = header_actions.child(
                TextButton::new(("session_fixture_emit_demo", cx.entity_id()), label)
                    .kind(ButtonKind::Ghost)
                    .disabled(self.emit_demo_in_flight)
                    .on_click(cx.listener(Self::emit_demo_message)),
            );
        }

        let header = div()
            .h(px(44.0))
            .px(theme.spacing.md)
            .flex()
            .items_center()
            .justify_between()
            .bg(theme.colors.surface_elevated)
            .child(
                div().flex().min_w_0().child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground)
                        .truncate()
                        .child(header_title),
                ),
            )
            .child(header_actions);

        let mut body = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .px(theme.spacing.md)
            .pb(theme.spacing.md)
            .gap(theme.spacing.sm);

        let error = self
            .load
            .error
            .clone()
            .or_else(|| self.create_and_pin.error.clone())
            .or_else(|| self.pin_existing.error.clone())
            .or_else(|| self.unpin.error.clone())
            .or_else(|| self.close_session.error.clone())
            .or_else(|| self.emit_demo_error.clone());

        if let Some(error) = error {
            body = body.child(
                Callout::new(error)
                    .kind(CalloutKind::Danger)
                    .title("Chat action failed"),
            );
        }

        match self.selected_epic.as_ref() {
            None => {}
            Some(_) if self.load.in_flight => {
                body = body.child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("Loading pinned chat…"),
                );
            }
            Some(_) if self.pinned_session_id.is_none() => {
                body = body.child(
                    Callout::new("No chat is pinned to this epic yet.")
                        .kind(CalloutKind::Info)
                        .title("Pinned chat")
                        .action(
                            TextButton::new(("chat_empty_create", cx.entity_id()), "Create chat")
                                .kind(ButtonKind::Secondary)
                                .disabled(self.create_and_pin.in_flight)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| this.create_and_pin_chat(cx))
                                    }
                                }),
                        ),
                );
            }
            Some(_) => {
                let closed = self
                    .pinned_chat_summary()
                    .and_then(|summary| summary.closed_at)
                    .is_some();

                if closed {
                    body = body.child(
                        Callout::new("Closing a chat does not unpin it from the epic.")
                            .kind(CalloutKind::Warning)
                            .title("Pinned chat is closed")
                            .action(
                                TextButton::new(
                                    ("chat_closed_create", cx.entity_id()),
                                    "Create new chat",
                                )
                                .kind(ButtonKind::Secondary)
                                .disabled(self.create_and_pin.in_flight)
                                .on_click({
                                    let view = view.clone();
                                    move |_, _, cx| {
                                        view.update(cx, |this, cx| this.create_and_pin_chat(cx))
                                    }
                                }),
                            ),
                    );
                }
            }
        }

        body = body.child(
            div()
                .flex_1()
                .min_h(px(0.0))
                .child(self.session_view.clone()),
        );

        let mut container = div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.surface)
            .child(header)
            .child(body)
            .relative();

        if matches!(self.panel, Some(EpicSessionPanePanel::PinExisting)) {
            let entity_id = cx.entity_id();
            let pinned = self.pinned_session_id;
            let in_flight = self.pin_existing.in_flight;

            let mut menu_body = div().flex().flex_col().gap(theme.spacing.sm).child(
                div()
                    .text_sm()
                    .text_color(theme.colors.foreground)
                    .child("Pin an existing chat"),
            );

            if self.chat_sessions.is_empty() {
                menu_body = menu_body.child(
                    div()
                        .text_sm()
                        .text_color(theme.colors.foreground_muted)
                        .child("No chat sessions exist yet."),
                );
            } else {
                let scroll = self.scroll.clone();
                let view_for_items = view.clone();
                let list = ScrollArea::new(("chat_pin_scroll", cx.entity_id()), scroll)
                    .scrollbar_width(px(8.0))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .w_full()
                            .gap(theme.spacing.xs)
                            .children(self.chat_sessions.iter().map(|session| {
                                let title = session
                                    .title
                                    .clone()
                                    .unwrap_or_else(|| "Untitled chat".to_string());
                                let closed = session.closed_at.is_some();
                                let label = if closed {
                                    format!("{title} (closed)")
                                } else {
                                    title
                                };

                                let is_pinned = pinned == Some(session.session_id);
                                let kind = if is_pinned {
                                    ButtonKind::Secondary
                                } else {
                                    ButtonKind::Ghost
                                };

                                let item_id = (
                                    gpui::ElementId::from(("chat_pin_item", entity_id)),
                                    session.session_id.to_string(),
                                );

                                TextButton::new(item_id, label)
                                    .menu_item()
                                    .kind(kind)
                                    .disabled(in_flight)
                                    .disabled_reason("Pinning…")
                                    .on_click({
                                        let view = view_for_items.clone();
                                        let session_id = session.session_id;
                                        move |_, _, cx| {
                                            view.update(cx, |this, cx| {
                                                this.pin_existing_chat(session_id, cx)
                                            });
                                        }
                                    })
                            })),
                    );

                menu_body =
                    menu_body.child(div().h(px(260.0)).w_full().overflow_hidden().child(list));
            }

            let overlay = div()
                .absolute()
                .inset_0()
                .child(div().absolute().inset_0().occlude().on_mouse_down(
                    gpui::MouseButton::Left,
                    {
                        let view = view.clone();
                        move |_, _, cx| {
                            view.update(cx, |this, cx| this.close_panel(cx));
                            cx.stop_propagation();
                        }
                    },
                ))
                .child(
                    div()
                        .absolute()
                        .top(px(44.0) + theme.spacing.xs)
                        .left(theme.spacing.md)
                        .child(
                            div()
                                .w(px(360.0))
                                .p(theme.spacing.md)
                                .rounded_md()
                                .bg(theme.colors.surface_elevated)
                                .border_1()
                                .border_color(theme.colors.border.opacity(0.4))
                                .child(menu_body),
                        ),
                );

            container = container.child(overlay);
        }

        container
    }
}

struct WorkspacePaneHost {
    focus_handle: FocusHandle,
    task_filters_focus_handle: FocusHandle,
    ui_updates: UiUpdateCounter,
    model: Entity<DesktopModel>,
    task_session_view: Entity<SessionView>,
    graph_view: Entity<GraphView>,
    task_filters: TaskFilters,
    task_filters_open: bool,
    task_filters_active_category: TaskFilterCategory,
    task_filters_hovered_category: Option<TaskFilterCategory>,
    task_filters_focus: TaskFiltersFocus,
    task_filters_input_source: TaskFiltersInputSource,
    task_filters_active_chip_index: usize,
    task_filters_active_value_index: usize,
    task_filters_search: SharedString,
    task_filters_search_input: Entity<TextInput>,
    task_filters_value_search: SharedString,
    task_filters_value_search_input: Entity<TextInput>,
    task_filters_menu_opacity: TransitionMap<&'static str>,
    sessions_collapsed: bool,
    ui_settings_error: Option<SharedString>,
    selected_epic_slug: Option<String>,
    selected_epic: Option<redesmyn_protocol::client::EpicSummary>,
    graph_state: redesmyn_protocol::ui_driver::UiGraphLoadState,
    graph_node_count: u32,
    graph_edge_count: u32,
    graph_error: Option<SharedString>,
    graph_task: Option<Task<()>>,
    selection_generation: u64,
    _subscriptions: Vec<Subscription>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskFiltersFocus {
    Categories,
    Values,
    Chips,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TaskFiltersInputSource {
    Mouse,
    Keyboard,
}

impl Focusable for WorkspacePaneHost {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl WorkspacePaneHost {
    fn new(
        model: Entity<DesktopModel>,
        sessions_collapsed: bool,
        ui_updates: UiUpdateCounter,
        cx: &mut Context<Self>,
    ) -> Self {
        let task_filters_search_input = cx.new(|cx| {
            TextInput::new(cx)
                .placeholder("Add filter…")
                .small()
                .menu_search()
        });
        let task_filters_value_search_input = cx.new(|cx| {
            TextInput::new(cx)
                .placeholder("Filter…")
                .small()
                .menu_search()
        });
        let mut subscriptions = Vec::new();
        subscriptions.push(cx.subscribe(
            &task_filters_search_input,
            |this, _, event, cx| match event {
                TextInputEvent::Changed(text) => {
                    this.task_filters_search = text.clone();
                    this.ui_updates.bump();
                    cx.notify();
                }
                TextInputEvent::Submitted(_) => {
                    if !this.task_filters_open {
                        return;
                    }

                    match this.task_filters_focus {
                        TaskFiltersFocus::Categories | TaskFiltersFocus::Chips => {
                            this.task_filters_focus = TaskFiltersFocus::Values;
                            this.task_filters_hovered_category =
                                Some(this.task_filters_active_category);
                            this.task_filters_active_value_index = 0;
                            this.ui_updates.bump();
                            cx.notify();
                        }
                        TaskFiltersFocus::Values => {
                            this.toggle_active_filter_value(cx);
                        }
                    }
                }
            },
        ));
        subscriptions.push(
            cx.subscribe(
                &task_filters_value_search_input,
                |this, _, event, cx| match event {
                    TextInputEvent::Changed(text) => {
                        this.task_filters_value_search = text.clone();
                        let visible_count = this
                            .visible_task_filter_value_indices(
                                this.task_filters_hovered_category
                                    .unwrap_or(this.task_filters_active_category),
                            )
                            .len();
                        if visible_count == 0 {
                            this.task_filters_active_value_index = 0;
                        } else if this.task_filters_active_value_index >= visible_count {
                            this.task_filters_active_value_index = 0;
                        }
                        this.ui_updates.bump();
                        cx.notify();
                    }
                    TextInputEvent::Submitted(_) => {
                        if !this.task_filters_open {
                            return;
                        }

                        match this.task_filters_focus {
                            TaskFiltersFocus::Categories | TaskFiltersFocus::Chips => {
                                this.task_filters_focus = TaskFiltersFocus::Values;
                                this.task_filters_hovered_category =
                                    Some(this.task_filters_active_category);
                                this.task_filters_active_value_index = 0;
                                this.ui_updates.bump();
                                cx.notify();
                            }
                            TaskFiltersFocus::Values => {
                                this.toggle_active_filter_value(cx);
                            }
                        }
                    }
                },
            ),
        );

        let task_session_client =
            model.update(cx, |model, _cx| model.take_task_control_plane_client());
        let task_session_view = cx.new(|cx| SessionView::new(task_session_client, None, cx));
        let graph_session_view = task_session_view.clone();
        Self {
            focus_handle: cx.focus_handle(),
            task_filters_focus_handle: cx.focus_handle(),
            ui_updates,
            model,
            task_session_view,
            graph_view: cx.new(|cx| GraphView::new_empty(graph_session_view, cx)),
            task_filters: TaskFilters::default(),
            task_filters_open: false,
            task_filters_active_category: TaskFilterCategory::TaskState,
            task_filters_hovered_category: None,
            task_filters_focus: TaskFiltersFocus::Categories,
            task_filters_input_source: TaskFiltersInputSource::Mouse,
            task_filters_active_chip_index: 0,
            task_filters_active_value_index: 0,
            task_filters_search: "".into(),
            task_filters_search_input,
            task_filters_value_search: "".into(),
            task_filters_value_search_input,
            task_filters_menu_opacity: TransitionMap::new(),
            sessions_collapsed,
            ui_settings_error: None,
            selected_epic_slug: None,
            selected_epic: None,
            graph_state: redesmyn_protocol::ui_driver::UiGraphLoadState::Unselected,
            graph_node_count: 0,
            graph_edge_count: 0,
            graph_error: None,
            graph_task: None,
            selection_generation: 0,
            _subscriptions: subscriptions,
        }
    }

    fn set_selected_epic(
        &mut self,
        epic_slug: Option<String>,
        epic: Option<redesmyn_protocol::client::EpicSummary>,
        cx: &mut Context<Self>,
    ) {
        if self.selected_epic_slug == epic_slug && self.selected_epic == epic {
            return;
        }

        let slug_changed = self.selected_epic_slug != epic_slug;
        self.selected_epic_slug = epic_slug;
        self.selected_epic = epic;
        if slug_changed {
            self.selection_generation = self.selection_generation.wrapping_add(1);
            self.graph_task = None;
            self.graph_state = redesmyn_protocol::ui_driver::UiGraphLoadState::Unselected;
            self.graph_node_count = 0;
            self.graph_edge_count = 0;
            self.graph_error = None;
            let task_session_view = self.task_session_view.clone();
            self.graph_view = cx.new(|cx| GraphView::new_empty(task_session_view, cx));
            if self.task_filters.is_active() {
                let filters = self.task_filters.clone();
                self.graph_view
                    .update(cx, |view, cx| view.set_task_filters(filters, cx));
            }

            if self.selected_epic_slug.is_some() {
                self.refresh_graph(cx);
            } else {
                self.ui_updates.bump();
                cx.notify();
            }
            return;
        }

        self.ui_updates.bump();
        cx.notify();
    }

    fn set_sessions_collapsed(&mut self, collapsed: bool, cx: &mut Context<Self>) {
        if self.sessions_collapsed == collapsed {
            return;
        }
        self.sessions_collapsed = collapsed;
        cx.notify();
    }

    fn open_task_filters(
        &mut self,
        _: &OpenTaskFilters,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let span = redesmyn_logging::redesmyn_info_span!("ui.workspace.filters.open");
        let _guard = span.enter();

        self.task_filters_open = true;
        self.task_filters_hovered_category = None;
        self.task_filters_focus = TaskFiltersFocus::Categories;
        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.task_filters_active_chip_index = 0;
        self.task_filters_active_value_index = 0;
        self.task_filters_search = "".into();
        self.task_filters_search_input
            .update(cx, |input, cx| input.set_text("", cx));
        self.task_filters_value_search = "".into();
        self.task_filters_value_search_input
            .update(cx, |input, cx| input.set_text("", cx));
        window.focus(&self.task_filters_search_input.focus_handle(cx));
        self.ui_updates.bump();
        cx.notify();
    }

    fn close_task_filters(
        &mut self,
        _: &CloseTaskFilters,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        let span = redesmyn_logging::redesmyn_info_span!("ui.workspace.filters.close");
        let _guard = span.enter();

        self.task_filters_open = false;
        self.task_filters_hovered_category = None;
        self.task_filters_focus = TaskFiltersFocus::Categories;
        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.task_filters_active_chip_index = 0;
        self.task_filters_active_value_index = 0;
        self.task_filters_search = "".into();
        self.task_filters_search_input
            .update(cx, |input, cx| input.set_text("", cx));
        self.task_filters_value_search = "".into();
        self.task_filters_value_search_input
            .update(cx, |input, cx| input.set_text("", cx));
        window.focus(&self.focus_handle);
        self.ui_updates.bump();
        cx.notify();
    }

    fn task_filters_toggle_chip_focus(
        &mut self,
        _: &TaskFiltersToggleChipFocus,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        let active_chips = self.active_task_filter_chip_categories();
        if active_chips.is_empty() {
            return;
        }

        match self.task_filters_focus {
            TaskFiltersFocus::Chips => {
                self.task_filters_focus = TaskFiltersFocus::Categories;
                window.focus(&self.task_filters_search_input.focus_handle(cx));
            }
            TaskFiltersFocus::Categories | TaskFiltersFocus::Values => {
                self.task_filters_focus = TaskFiltersFocus::Chips;
                self.task_filters_hovered_category = None;
                self.task_filters_active_chip_index = self
                    .task_filters_active_chip_index
                    .min(active_chips.len().saturating_sub(1));
                window.focus(&self.task_filters_focus_handle);
            }
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn task_filters_activate(
        &mut self,
        _: &TaskFiltersActivate,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        match self.task_filters_focus {
            TaskFiltersFocus::Categories => {
                self.task_filters_focus = TaskFiltersFocus::Values;
                self.task_filters_hovered_category = Some(self.task_filters_active_category);
                self.task_filters_value_search = "".into();
                self.task_filters_value_search_input
                    .update(cx, |input, cx| input.set_text("", cx));
                self.task_filters_active_value_index = 0;
                window.focus(&self.task_filters_focus_handle);
            }
            TaskFiltersFocus::Values => {
                self.toggle_active_filter_value(cx);
            }
            TaskFiltersFocus::Chips => {
                let Some(category) = self.active_task_filter_chip_category() else {
                    return;
                };
                self.task_filters_active_category = category;
                self.task_filters_hovered_category = Some(category);
                self.task_filters_focus = TaskFiltersFocus::Values;
                self.task_filters_active_value_index = 0;
                self.task_filters_value_search = "".into();
                self.task_filters_value_search_input
                    .update(cx, |input, cx| input.set_text("", cx));
                window.focus(&self.task_filters_focus_handle);
            }
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn task_filters_clear_focused_chip(
        &mut self,
        _: &TaskFiltersClearFocusedChip,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open || self.task_filters_focus != TaskFiltersFocus::Chips {
            return;
        }

        let Some(category) = self.active_task_filter_chip_category() else {
            return;
        };

        self.clear_filter_category(category, cx);
        self.task_filters_hovered_category = None;

        let active_chips = self.active_task_filter_chip_categories();
        if active_chips.is_empty() {
            self.task_filters_focus = TaskFiltersFocus::Categories;
            self.task_filters_active_chip_index = 0;
            window.focus(&self.task_filters_search_input.focus_handle(cx));
        } else if self.task_filters_active_chip_index >= active_chips.len() {
            self.task_filters_active_chip_index = active_chips.len().saturating_sub(1);
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn visible_task_filter_categories(&self) -> Vec<TaskFilterCategory> {
        let query = self.task_filters_search.trim().to_ascii_lowercase();
        TaskFilterCategory::ALL
            .into_iter()
            .filter(|category| {
                query.is_empty() || category.title().to_ascii_lowercase().contains(&query)
            })
            .collect()
    }

    fn visible_task_filter_value_indices(&self, category: TaskFilterCategory) -> Vec<usize> {
        let query = self.task_filters_value_search.trim().to_ascii_lowercase();
        match category {
            TaskFilterCategory::TaskState => TASK_STATE_OPTIONS
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, value)| {
                    let label = task_state_title(value);
                    if query.is_empty() || label.to_ascii_lowercase().contains(&query) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect(),
            TaskFilterCategory::MergeReadiness => MERGE_READINESS_OPTIONS
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, value)| {
                    let label = merge_readiness_title(value);
                    if query.is_empty() || label.to_ascii_lowercase().contains(&query) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect(),
            TaskFilterCategory::AgentStatus => TaskAgentStatus::ALL
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, value)| {
                    let label = value.title();
                    if query.is_empty() || label.to_ascii_lowercase().contains(&query) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    fn active_task_filter_chip_categories(&self) -> Vec<TaskFilterCategory> {
        TaskFilterCategory::ALL
            .into_iter()
            .filter(|category| self.task_filters.selected_count(*category) > 0)
            .collect()
    }

    fn active_task_filter_chip_category(&self) -> Option<TaskFilterCategory> {
        let active = self.active_task_filter_chip_categories();
        active.get(self.task_filters_active_chip_index).copied()
    }

    fn move_task_filter_chip(&mut self, delta: isize) {
        let active = self.active_task_filter_chip_categories();
        if active.is_empty() {
            self.task_filters_active_chip_index = 0;
            return;
        }

        let next_index = (self.task_filters_active_chip_index as isize + delta)
            .rem_euclid(active.len() as isize) as usize;
        self.task_filters_active_chip_index = next_index;
    }

    fn toggle_active_filter_value(&mut self, cx: &mut Context<Self>) {
        let visible = self.visible_task_filter_value_indices(self.task_filters_active_category);
        let Some(source_index) = visible.get(self.task_filters_active_value_index).copied() else {
            return;
        };

        match self.task_filters_active_category {
            TaskFilterCategory::TaskState => {
                let Some(value) = TASK_STATE_OPTIONS.get(source_index).copied() else {
                    return;
                };
                self.toggle_task_state_filter(value, cx);
            }
            TaskFilterCategory::MergeReadiness => {
                let Some(value) = MERGE_READINESS_OPTIONS.get(source_index).copied() else {
                    return;
                };
                self.toggle_merge_readiness_filter(value, cx);
            }
            TaskFilterCategory::AgentStatus => {
                let Some(value) = TaskAgentStatus::ALL.get(source_index).copied() else {
                    return;
                };
                self.toggle_agent_status_filter(value, cx);
            }
        };
    }

    fn move_task_filter_category(&mut self, delta: isize, cx: &mut Context<Self>) {
        let visible = self.visible_task_filter_categories();
        if visible.is_empty() {
            return;
        }

        let current_index = visible
            .iter()
            .position(|category| *category == self.task_filters_active_category)
            .unwrap_or(0);

        let next_index =
            (current_index as isize + delta).rem_euclid(visible.len() as isize) as usize;
        let next_category = visible[next_index];

        if self.task_filters_active_category != next_category {
            self.task_filters_value_search = "".into();
            self.task_filters_value_search_input
                .update(cx, |input, cx| input.set_text("", cx));
        }

        self.task_filters_active_category = next_category;
        self.task_filters_active_value_index = 0;
        if self.task_filters_hovered_category.is_some() {
            self.task_filters_hovered_category = Some(next_category);
        }
    }

    fn move_task_filter_value(&mut self, delta: isize) {
        let value_count = self
            .visible_task_filter_value_indices(self.task_filters_active_category)
            .len();
        if value_count == 0 {
            return;
        }

        let next_index = (self.task_filters_active_value_index as isize + delta)
            .rem_euclid(value_count as isize) as usize;
        self.task_filters_active_value_index = next_index;
    }

    fn task_filters_move_up(
        &mut self,
        _: &TaskFiltersMoveUp,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        match self.task_filters_focus {
            TaskFiltersFocus::Categories => self.move_task_filter_category(-1, cx),
            TaskFiltersFocus::Values => self.move_task_filter_value(-1),
            TaskFiltersFocus::Chips => {}
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn task_filters_move_down(
        &mut self,
        _: &TaskFiltersMoveDown,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        match self.task_filters_focus {
            TaskFiltersFocus::Categories => self.move_task_filter_category(1, cx),
            TaskFiltersFocus::Values => self.move_task_filter_value(1),
            TaskFiltersFocus::Chips => {}
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn task_filters_move_left(
        &mut self,
        _: &TaskFiltersMoveLeft,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        match self.task_filters_focus {
            TaskFiltersFocus::Categories => {
                self.task_filters_hovered_category = None;
            }
            TaskFiltersFocus::Values => {
                self.task_filters_focus = TaskFiltersFocus::Categories;
                self.task_filters_hovered_category = None;
                self.task_filters_value_search = "".into();
                self.task_filters_value_search_input
                    .update(cx, |input, cx| input.set_text("", cx));
                window.focus(&self.task_filters_search_input.focus_handle(cx));
            }
            TaskFiltersFocus::Chips => {
                self.move_task_filter_chip(-1);
            }
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn task_filters_move_right(
        &mut self,
        _: &TaskFiltersMoveRight,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters_open {
            return;
        }

        match self.task_filters_focus {
            TaskFiltersFocus::Categories => {
                self.task_filters_focus = TaskFiltersFocus::Values;
                self.task_filters_hovered_category = Some(self.task_filters_active_category);
                self.task_filters_value_search = "".into();
                self.task_filters_value_search_input
                    .update(cx, |input, cx| input.set_text("", cx));
                self.task_filters_active_value_index = self.task_filters_active_value_index.min(
                    self.visible_task_filter_value_indices(self.task_filters_active_category)
                        .len()
                        .saturating_sub(1),
                );
                window.focus(&self.task_filters_focus_handle);
            }
            TaskFiltersFocus::Values => {}
            TaskFiltersFocus::Chips => {
                self.move_task_filter_chip(1);
            }
        }

        self.task_filters_input_source = TaskFiltersInputSource::Keyboard;
        self.ui_updates.bump();
        cx.notify();
    }

    fn apply_task_filters(&mut self, cx: &mut Context<Self>) {
        let filters = self.task_filters.clone();
        self.graph_view
            .update(cx, |view, cx| view.set_task_filters(filters, cx));
        self.ui_updates.bump();
        cx.notify();
    }

    fn toggle_task_state_filter(
        &mut self,
        value: redesmyn_protocol::client::TaskState,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters.task_states.insert(value) {
            self.task_filters.task_states.remove(&value);
        }
        self.apply_task_filters(cx);
    }

    fn toggle_merge_readiness_filter(
        &mut self,
        value: redesmyn_protocol::client::MergeReadiness,
        cx: &mut Context<Self>,
    ) {
        if !self.task_filters.merge_readiness.insert(value) {
            self.task_filters.merge_readiness.remove(&value);
        }
        self.apply_task_filters(cx);
    }

    fn toggle_agent_status_filter(&mut self, value: TaskAgentStatus, cx: &mut Context<Self>) {
        if !self.task_filters.agent_statuses.insert(value) {
            self.task_filters.agent_statuses.remove(&value);
        }
        self.apply_task_filters(cx);
    }

    fn clear_filter_category(&mut self, category: TaskFilterCategory, cx: &mut Context<Self>) {
        match category {
            TaskFilterCategory::TaskState => self.task_filters.task_states.clear(),
            TaskFilterCategory::MergeReadiness => self.task_filters.merge_readiness.clear(),
            TaskFilterCategory::AgentStatus => self.task_filters.agent_statuses.clear(),
        }

        self.apply_task_filters(cx);
    }

    fn clear_all_filters(&mut self, cx: &mut Context<Self>) {
        if !self.task_filters.is_active() {
            return;
        }
        self.task_filters = TaskFilters::default();
        self.apply_task_filters(cx);
    }

    fn is_current_selection(&self, generation: u64, epic_slug: &str) -> bool {
        if self.selection_generation != generation {
            return false;
        }
        matches!(
            self.selected_epic.as_ref(),
            Some(epic) if epic.slug == epic_slug
        )
    }

    fn set_ui_settings_error(&mut self, error: Option<SharedString>, cx: &mut Context<Self>) {
        if self.ui_settings_error == error {
            return;
        }
        self.ui_settings_error = error;
        self.ui_updates.bump();
        cx.notify();
    }

    fn refresh_graph(&mut self, cx: &mut Context<Self>) {
        if self.graph_task.is_some() {
            return;
        }

        let Some(client) = self.model.read(cx).chrome_control_plane_client.clone() else {
            self.graph_state = redesmyn_protocol::ui_driver::UiGraphLoadState::Error;
            self.graph_error = Some("Control plane client unavailable.".into());
            self.ui_updates.bump();
            cx.notify();
            return;
        };

        let Some(epic_slug) = self.selected_epic_slug.clone() else {
            return;
        };

        let generation = self.selection_generation;
        let epic_slug_for_task = epic_slug.clone();

        let span = redesmyn_logging::redesmyn_info_span!("ui.workspace.graph.refresh", epic_slug = %epic_slug);
        let _guard = span.enter();

        self.graph_state = redesmyn_protocol::ui_driver::UiGraphLoadState::Loading;
        self.graph_error = None;
        self.ui_updates.bump();
        cx.notify();

        let tokio = client.tokio().clone();
        self.graph_task = Some(cx.spawn(move |weak: WeakEntity<Self>, cx: &mut AsyncApp| {
            let cx = cx.clone();
            async move {
                let Some(entity) = weak.upgrade() else {
                    return;
                };

                let task =
                    tokio.spawn(async move { client.get_epic_graph(epic_slug_for_task).await });
                let result = match task.await {
                    Ok(result) => result,
                    Err(error) => Err(ControlPlaneClientError::Server {
                        message: format!("Graph request failed: {error}"),
                    }),
                };

                let _ = cx.update(|cx| {
                    entity.update(cx, |this, cx| {
                        if !this.is_current_selection(generation, &epic_slug) {
                            return;
                        }

                        this.graph_task = None;

                        match result {
                            Ok(graph) => {
                                this.graph_node_count =
                                    graph.nodes.len().min(u32::MAX as usize) as u32;
                                this.graph_edge_count =
                                    graph.edges.len().min(u32::MAX as usize) as u32;
                                this.graph_state = if graph.nodes.is_empty() {
                                    redesmyn_protocol::ui_driver::UiGraphLoadState::Empty
                                } else {
                                    redesmyn_protocol::ui_driver::UiGraphLoadState::Loaded
                                };
                                this.graph_error = None;
                                this.graph_view.update(cx, |view, cx| {
                                    view.replace_from_epic_graph(&graph, cx);
                                });
                            }
                            Err(err) => {
                                this.graph_state =
                                    redesmyn_protocol::ui_driver::UiGraphLoadState::Error;
                                this.graph_error = Some(err.to_string().into());
                            }
                        }

                        this.ui_updates.bump();
                        cx.notify();
                    })
                });
            }
        }));
        self.ui_updates.bump();
        cx.notify();
    }
}

impl Render for WorkspacePaneHost {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = theme_for_window(window, cx);

        let entity_id = cx.entity_id();
        let workspace = cx.entity();

        let filter_shortcut_enabled = window.is_action_available(&OpenTaskFilters, cx);
        let filter_tab_height = px(28.0);
        let keycap = |label: &'static str| {
            div()
                .px(theme.spacing.xs)
                .py(px(1.0))
                .rounded(theme.radius.sm)
                .bg(theme.colors.surface.opacity(0.92))
                .border_1()
                .border_color(theme.colors.border.opacity(0.35))
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child(label)
        };
        let filter_keycap = keycap("F").opacity(if filter_shortcut_enabled { 1.0 } else { 0.4 });

        let filter_button = TextButton::new(("workspace_filters_button", entity_id), "Filter")
            .kind(ButtonKind::Ghost)
            .compact()
            .trailing(
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.xs)
                    .child(filter_keycap)
                    .child("▾"),
            )
            .on_click({
                let workspace = workspace.clone();
                move |_, window, cx| {
                    let focus = workspace.update(cx, |this, cx| {
                        this.task_filters_open = !this.task_filters_open;
                        this.task_filters_hovered_category = None;
                        this.task_filters_focus = TaskFiltersFocus::Categories;
                        this.task_filters_input_source = TaskFiltersInputSource::Mouse;
                        this.task_filters_active_value_index = 0;
                        this.task_filters_search = "".into();
                        this.task_filters_search_input
                            .update(cx, |input, cx| input.set_text("", cx));

                        if this.task_filters_open {
                            Some(this.task_filters_search_input.focus_handle(cx))
                        } else {
                            Some(this.focus_handle.clone())
                        }
                    });

                    if let Some(focus) = focus {
                        window.focus(&focus);
                    }
                }
            });

        let mut chips_row = div()
            .flex()
            .flex_row()
            .items_center()
            .gap(px(2.0))
            .min_w_0();
        let mut has_chips = false;
        let mut next_chip_index = 0_usize;

        for category in TaskFilterCategory::ALL {
            let Some(label) = self.task_filters.chip_label(category) else {
                continue;
            };
            has_chips = true;
            let chip_index = next_chip_index;
            next_chip_index = next_chip_index.saturating_add(1);
            let highlighted = self.task_filters_focus == TaskFiltersFocus::Chips
                && self.task_filters_active_chip_index == chip_index;

            let open_label = div()
                .min_w_0()
                .text_xs()
                .text_color(theme.colors.foreground)
                .truncate()
                .child(label)
                .cursor_pointer()
                .on_mouse_down(gpui::MouseButton::Left, {
                    let workspace = workspace.clone();
                    move |_, window, cx| {
                        let focus = workspace.update(cx, |this, cx| {
                            this.task_filters_open = true;
                            this.task_filters_active_category = category;
                            this.task_filters_hovered_category = Some(category);
                            this.task_filters_focus = TaskFiltersFocus::Categories;
                            this.task_filters_active_value_index = 0;
                            this.task_filters_search = "".into();
                            this.task_filters_search_input
                                .update(cx, |input, cx| input.set_text("", cx));
                            this.task_filters_value_search = "".into();
                            this.task_filters_value_search_input
                                .update(cx, |input, cx| input.set_text("", cx));
                            Some(this.task_filters_search_input.focus_handle(cx))
                        });
                        if let Some(focus) = focus {
                            window.focus(&focus);
                        }
                    }
                });

            let clear_button = div()
                .text_xs()
                .text_color(theme.colors.foreground_muted)
                .child("×")
                .cursor_pointer()
                .hover(|this| this.text_color(theme.colors.foreground))
                .on_mouse_down(gpui::MouseButton::Left, {
                    let workspace = workspace.clone();
                    move |_, _, cx| {
                        workspace.update(cx, |this, cx| this.clear_filter_category(category, cx));
                    }
                });

            let chip = overlay_surface(&theme, OverlaySurfaceKind::Chrome, px(999.0))
                .h(filter_tab_height)
                .flex()
                .flex_row()
                .items_center()
                .gap(theme.spacing.xs)
                .px(theme.spacing.sm)
                .when(highlighted, |this| {
                    this.border_color(theme.colors.ring.opacity(0.7))
                })
                .child(open_label)
                .child(clear_button);

            chips_row = chips_row.child(chip);
        }

        let mut body = div().flex().flex_col().size_full();

        if let Some(error) = self.ui_settings_error.clone() {
            body = body.child(
                div().px(theme.spacing.md).pt(theme.spacing.md).child(
                    Callout::new(error)
                        .kind(CalloutKind::Warning)
                        .title("Unable to save UI settings"),
                ),
            );
        }

        let graph_host = {
            let mut host = div()
                .flex_1()
                .min_h(px(0.0))
                .relative()
                .child(self.graph_view.clone());

            host = match self.graph_state {
                redesmyn_protocol::ui_driver::UiGraphLoadState::Unselected => host.child(
                    div()
                        .absolute()
                        .inset_0()
                        .px(theme.spacing.lg)
                        .py(theme.spacing.lg)
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            div().max_w(px(520.0)).child(
                                Callout::new("Select an epic to load its task graph.")
                                    .kind(CalloutKind::Info)
                                    .title("No epic selected"),
                            ),
                        ),
                ),
                redesmyn_protocol::ui_driver::UiGraphLoadState::Loading => host.child(
                    div()
                        .absolute()
                        .inset_0()
                        .px(theme.spacing.lg)
                        .py(theme.spacing.lg)
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(ProgressPill::new("Loading graph")),
                ),
                redesmyn_protocol::ui_driver::UiGraphLoadState::Empty => host.child(
                    div()
                        .absolute()
                        .inset_0()
                        .px(theme.spacing.lg)
                        .py(theme.spacing.lg)
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            div().max_w(px(560.0)).child(
                                Callout::new("This epic has no tasks yet.")
                                    .kind(CalloutKind::Info)
                                    .title("No tasks")
                                    .action(
                                        TextButton::new(
                                            ("graph_empty_reload", entity_id),
                                            "Reload",
                                        )
                                        .kind(ButtonKind::Secondary)
                                        .on_click({
                                            let graph = cx.entity();
                                            move |_, _, cx| {
                                                graph.update(cx, |this, cx| this.refresh_graph(cx))
                                            }
                                        }),
                                    ),
                            ),
                        ),
                ),
                redesmyn_protocol::ui_driver::UiGraphLoadState::Error => {
                    let message = self
                        .graph_error
                        .clone()
                        .unwrap_or_else(|| "Graph load failed.".into());

                    host.child(
                        div()
                            .absolute()
                            .inset_0()
                            .px(theme.spacing.lg)
                            .py(theme.spacing.lg)
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(
                                div().max_w(px(560.0)).child(
                                    Callout::new(message)
                                        .kind(CalloutKind::Danger)
                                        .title("Unable to load graph")
                                        .action(
                                            TextButton::new(
                                                ("graph_error_retry", entity_id),
                                                "Retry",
                                            )
                                            .kind(ButtonKind::Secondary)
                                            .on_click(
                                                {
                                                    let graph = cx.entity();
                                                    move |_, _, cx| {
                                                        graph.update(cx, |this, cx| {
                                                            this.refresh_graph(cx)
                                                        })
                                                    }
                                                },
                                            ),
                                        ),
                                ),
                            ),
                    )
                }
                redesmyn_protocol::ui_driver::UiGraphLoadState::Loaded
                | redesmyn_protocol::ui_driver::UiGraphLoadState::Unknown => host,
            };

            let show_filter_ui = matches!(
                self.graph_state,
                redesmyn_protocol::ui_driver::UiGraphLoadState::Loaded
                    | redesmyn_protocol::ui_driver::UiGraphLoadState::Unknown
            );

            if show_filter_ui {
                let tab_top =
                    theme.spacing.md + self.graph_view.read(cx).selection_bar_reserved_top_offset();
                let tab_left = theme.spacing.md;

                let menu_opacity = self.task_filters_menu_opacity.opacity_for_render(
                    "task_filters_menu",
                    self.task_filters_open,
                    ui_test_mode_animation_duration(
                        theme
                            .animation
                            .fast
                            .saturating_sub(Duration::from_millis(80)),
                    ),
                    window,
                );

                let filter_button_pill =
                    overlay_surface(&theme, OverlaySurfaceKind::Chrome, px(999.0))
                        .h(filter_tab_height)
                        .flex()
                        .flex_row()
                        .items_center()
                        .when(self.task_filters_open, |this| {
                            this.border_color(theme.colors.ring.opacity(0.7))
                        })
                        .child(filter_button);

                let filter_tab = div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(px(2.0))
                    .child(filter_button_pill)
                    .when(has_chips, |this| this.child(chips_row));

                host = host.child(
                    div()
                        .absolute()
                        .top(tab_top)
                        .left(tab_left)
                        .flex()
                        .flex_row()
                        .items_center()
                        .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| {
                            cx.stop_propagation();
                        })
                        .child(filter_tab),
                );

                if menu_opacity > 1e-3 {
                    let primary_menu_width = px(260.0);
                    let submenu_width = px(240.0);
                    let submenu_overlap = theme.spacing.xs;
                    let menu_item_height = px(34.0);
                    let menu_search_height = menu_item_height;

                    let query = self.task_filters_search.trim().to_ascii_lowercase();
                    let visible_categories: Vec<TaskFilterCategory> = TaskFilterCategory::ALL
                        .into_iter()
                        .filter(|category| {
                            query.is_empty()
                                || category.title().to_ascii_lowercase().contains(&query)
                        })
                        .collect();

                    let mut category_list = div().flex().flex_col().gap(px(0.0));
                    if visible_categories.is_empty() {
                        category_list = category_list.child(
                            div()
                                .text_xs()
                                .text_color(theme.colors.foreground_muted)
                                .child("No matching filters."),
                        );
                    } else {
                        for category in visible_categories.iter().copied() {
                            let count = self.task_filters.selected_count(category);
                            let active = self.task_filters_active_category == category;
                            let hovered = self.task_filters_hovered_category == Some(category);
                            let highlighted =
                                hovered || (self.task_filters_hovered_category.is_none() && active);

                            let badge = if count > 0 {
                                Some(
                                    div()
                                        .px(theme.spacing.xs)
                                        .py(px(1.0))
                                        .rounded(theme.radius.sm)
                                        .bg(theme.colors.surface.opacity(0.85))
                                        .border_1()
                                        .border_color(theme.colors.border.opacity(0.25))
                                        .text_xs()
                                        .text_color(theme.colors.foreground_muted)
                                        .child(format!("{count}")),
                                )
                            } else {
                                None
                            };

                            let arrow = div()
                                .text_xs()
                                .text_color(theme.colors.foreground_muted)
                                .child("›");

                            let row = div()
                                .id((
                                    gpui::ElementId::from(("task_filters_category", entity_id)),
                                    category.title(),
                                ))
                                .flex()
                                .flex_row()
                                .w_full()
                                .items_center()
                                .gap(theme.spacing.xs)
                                .px(theme.spacing.sm)
                                .h(menu_item_height)
                                .rounded(theme.radius.sm)
                                .when(highlighted, |this| this.bg(theme.colors.accent))
                                .when(
                                    self.task_filters_input_source == TaskFiltersInputSource::Mouse,
                                    |this| {
                                        this.hover(|this| {
                                            this.bg(theme.colors.accent.opacity(0.75))
                                        })
                                    },
                                )
                                .cursor_pointer()
                                .on_mouse_move(cx.listener(move |this, _, _, cx| {
                                    let mut did_change = false;
                                    if this.task_filters_input_source
                                        != TaskFiltersInputSource::Mouse
                                    {
                                        this.task_filters_input_source =
                                            TaskFiltersInputSource::Mouse;
                                        did_change = true;
                                    }
                                    if this.task_filters_active_category != category {
                                        this.task_filters_active_category = category;
                                        did_change = true;
                                    }
                                    if this.task_filters_hovered_category != Some(category) {
                                        this.task_filters_hovered_category = Some(category);
                                        did_change = true;
                                    }
                                    if this.task_filters_focus != TaskFiltersFocus::Categories {
                                        this.task_filters_focus = TaskFiltersFocus::Categories;
                                        did_change = true;
                                    }

                                    if did_change {
                                        this.task_filters_active_value_index = 0;
                                        this.task_filters_value_search = "".into();
                                        this.task_filters_value_search_input
                                            .update(cx, |input, cx| input.set_text("", cx));
                                        cx.notify();
                                    }
                                }))
                                .on_mouse_down(gpui::MouseButton::Left, {
                                    let workspace = workspace.clone();
                                    move |_, window, cx| {
                                        let focus = workspace.update(cx, |this, cx| {
                                            this.task_filters_active_category = category;
                                            this.task_filters_hovered_category = Some(category);
                                            this.task_filters_focus = TaskFiltersFocus::Values;
                                            this.task_filters_active_value_index = 0;
                                            this.task_filters_value_search = "".into();
                                            this.task_filters_value_search_input
                                                .update(cx, |input, cx| input.set_text("", cx));
                                            cx.notify();
                                            this.task_filters_focus_handle.clone()
                                        });
                                        window.focus(&focus);
                                    }
                                })
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .text_xs()
                                        .text_color(theme.colors.foreground)
                                        .truncate()
                                        .child(category.title()),
                                )
                                .when_some(badge, |this, badge| this.child(badge))
                                .child(arrow);

                            category_list = category_list.child(row);
                        }
                    }

                    let actions_row = div()
                        .pt(theme.spacing.xs)
                        .flex()
                        .flex_row()
                        .items_center()
                        .justify_between()
                        .w_full()
                        .child(
                            TextButton::new(("task_filters_clear_all", entity_id), "Clear all")
                                .kind(ButtonKind::Ghost)
                                .small()
                                .disabled(!self.task_filters.is_active())
                                .on_click({
                                    let workspace = workspace.clone();
                                    move |_, _, cx| {
                                        workspace.update(cx, |this, cx| this.clear_all_filters(cx));
                                    }
                                }),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_row()
                                .items_center()
                                .gap(theme.spacing.xs)
                                .px(theme.spacing.sm)
                                .py(theme.spacing.xs)
                                .rounded(theme.radius.sm)
                                .text_xs()
                                .text_color(theme.colors.foreground_muted)
                                .child(keycap("Esc"))
                                .child("Close"),
                        );

                    let primary_menu =
                        overlay_surface(&theme, OverlaySurfaceKind::Menu, theme.radius.lg)
                            .w(primary_menu_width)
                            .pt(theme.spacing.sm)
                            .pb(theme.spacing.md)
                            .px(theme.spacing.sm)
                            .shadow_md()
                            .occlude()
                            .child(
                                div()
                                    .h(menu_search_height)
                                    .flex()
                                    .items_center()
                                    .w_full()
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w_0()
                                            .child(self.task_filters_search_input.clone()),
                                    ),
                            )
                            .child(div().pt(theme.spacing.sm).child(category_list))
                            .child(actions_row);

                    let submenu_state = self.task_filters_hovered_category.and_then(|category| {
                        let row_index = visible_categories
                            .iter()
                            .position(|candidate| *candidate == category)?;
                        Some((category, row_index))
                    });

                    let submenu_top = submenu_state.map(|(_category, row_index)| {
                        let categories_top =
                            theme.spacing.sm + menu_search_height + theme.spacing.sm;
                        let row_top = categories_top + (menu_item_height * row_index as f32);
                        let submenu_padding_top = theme.spacing.sm;
                        (row_top - submenu_padding_top).max(px(0.0))
                    });

                    let submenu = submenu_state.map(|(category, _row_index)| {
                        let visible_values = self.visible_task_filter_value_indices(category);
                        let mut value_list = div().flex().flex_col().gap(px(0.0));
                        let checkbox = |selected: bool| {
                            div()
                                .size(px(16.0))
                                .flex()
                                .items_center()
                                .justify_center()
                                .rounded(theme.radius.sm)
                                .border_1()
                                .border_color(theme.colors.border.opacity(0.45))
                                .when(selected, |this| {
                                    this.border_color(theme.colors.ring.opacity(0.7))
                                        .text_color(theme.colors.ring)
                                })
                                .text_xs()
                                .text_color(theme.colors.foreground_muted)
                                .child(if selected { "✓" } else { "" })
                        };

                        if visible_values.is_empty() {
                            value_list = value_list.child(
                                div()
                                    .text_xs()
                                    .text_color(theme.colors.foreground_muted)
                                    .child("No matching values."),
                            );
                        } else {
                            match category {
                                TaskFilterCategory::TaskState => {
                                    for (display_index, source_index) in
                                        visible_values.into_iter().enumerate()
                                    {
                                        let Some(value) =
                                            TASK_STATE_OPTIONS.get(source_index).copied()
                                        else {
                                            continue;
                                        };
                                        let selected =
                                            self.task_filters.task_states.contains(&value);
                                        let label = task_state_title(value);
                                        let highlighted = self.task_filters_focus
                                            == TaskFiltersFocus::Values
                                            && self.task_filters_active_value_index
                                                == display_index;

                                        let row_id = (
                                            gpui::ElementId::from((
                                                "task_filters_value",
                                                entity_id,
                                            )),
                                            format!("task_state:{label}"),
                                        );

                                        let row = div()
                                            .id(row_id)
                                            .flex()
                                            .flex_row()
                                            .w_full()
                                            .items_center()
                                            .gap(theme.spacing.sm)
                                            .px(theme.spacing.sm)
                                            .h(menu_item_height)
                                            .rounded(theme.radius.sm)
                                            .when(highlighted, |this| this.bg(theme.colors.accent))
                                            .when(
                                                self.task_filters_input_source
                                                    == TaskFiltersInputSource::Mouse,
                                                |this| {
                                                    this.hover(|this| {
                                                        this.bg(theme.colors.accent.opacity(0.75))
                                                    })
                                                },
                                            )
                                            .cursor_pointer()
                                            .on_mouse_move(cx.listener(move |this, _, _, cx| {
                                                let mut did_change = false;
                                                if this.task_filters_input_source
                                                    != TaskFiltersInputSource::Mouse
                                                {
                                                    this.task_filters_input_source =
                                                        TaskFiltersInputSource::Mouse;
                                                    did_change = true;
                                                }
                                                if this.task_filters_active_value_index
                                                    != display_index
                                                {
                                                    this.task_filters_active_value_index =
                                                        display_index;
                                                    did_change = true;
                                                }
                                                if this.task_filters_focus
                                                    != TaskFiltersFocus::Values
                                                {
                                                    this.task_filters_focus =
                                                        TaskFiltersFocus::Values;
                                                    did_change = true;
                                                }
                                                if did_change {
                                                    cx.notify();
                                                }
                                            }))
                                            .on_mouse_down(gpui::MouseButton::Left, {
                                                let workspace = workspace.clone();
                                                move |_, _, cx| {
                                                    workspace.update(cx, |this, cx| {
                                                        this.task_filters_focus =
                                                            TaskFiltersFocus::Values;
                                                        this.task_filters_active_value_index =
                                                            display_index;
                                                        this.toggle_task_state_filter(value, cx);
                                                    });
                                                }
                                            })
                                            .child(checkbox(selected))
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .text_xs()
                                                    .text_color(theme.colors.foreground)
                                                    .truncate()
                                                    .child(label),
                                            );

                                        value_list = value_list.child(row);
                                    }
                                }
                                TaskFilterCategory::MergeReadiness => {
                                    for (display_index, source_index) in
                                        visible_values.into_iter().enumerate()
                                    {
                                        let Some(value) =
                                            MERGE_READINESS_OPTIONS.get(source_index).copied()
                                        else {
                                            continue;
                                        };
                                        let selected =
                                            self.task_filters.merge_readiness.contains(&value);
                                        let label = merge_readiness_title(value);
                                        let highlighted = self.task_filters_focus
                                            == TaskFiltersFocus::Values
                                            && self.task_filters_active_value_index
                                                == display_index;

                                        let row_id = (
                                            gpui::ElementId::from((
                                                "task_filters_value",
                                                entity_id,
                                            )),
                                            format!("merge_readiness:{label}"),
                                        );

                                        let row = div()
                                            .id(row_id)
                                            .flex()
                                            .flex_row()
                                            .w_full()
                                            .items_center()
                                            .gap(theme.spacing.sm)
                                            .px(theme.spacing.sm)
                                            .h(menu_item_height)
                                            .rounded(theme.radius.sm)
                                            .when(highlighted, |this| this.bg(theme.colors.accent))
                                            .when(
                                                self.task_filters_input_source
                                                    == TaskFiltersInputSource::Mouse,
                                                |this| {
                                                    this.hover(|this| {
                                                        this.bg(theme.colors.accent.opacity(0.75))
                                                    })
                                                },
                                            )
                                            .cursor_pointer()
                                            .on_mouse_move(cx.listener(move |this, _, _, cx| {
                                                let mut did_change = false;
                                                if this.task_filters_input_source
                                                    != TaskFiltersInputSource::Mouse
                                                {
                                                    this.task_filters_input_source =
                                                        TaskFiltersInputSource::Mouse;
                                                    did_change = true;
                                                }
                                                if this.task_filters_active_value_index
                                                    != display_index
                                                {
                                                    this.task_filters_active_value_index =
                                                        display_index;
                                                    did_change = true;
                                                }
                                                if this.task_filters_focus
                                                    != TaskFiltersFocus::Values
                                                {
                                                    this.task_filters_focus =
                                                        TaskFiltersFocus::Values;
                                                    did_change = true;
                                                }
                                                if did_change {
                                                    cx.notify();
                                                }
                                            }))
                                            .on_mouse_down(gpui::MouseButton::Left, {
                                                let workspace = workspace.clone();
                                                move |_, _, cx| {
                                                    workspace.update(cx, |this, cx| {
                                                        this.task_filters_focus =
                                                            TaskFiltersFocus::Values;
                                                        this.task_filters_active_value_index =
                                                            display_index;
                                                        this.toggle_merge_readiness_filter(
                                                            value, cx,
                                                        );
                                                    });
                                                }
                                            })
                                            .child(checkbox(selected))
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .text_xs()
                                                    .text_color(theme.colors.foreground)
                                                    .truncate()
                                                    .child(label),
                                            );

                                        value_list = value_list.child(row);
                                    }
                                }
                                TaskFilterCategory::AgentStatus => {
                                    for (display_index, source_index) in
                                        visible_values.into_iter().enumerate()
                                    {
                                        let Some(value) =
                                            TaskAgentStatus::ALL.get(source_index).copied()
                                        else {
                                            continue;
                                        };
                                        let selected =
                                            self.task_filters.agent_statuses.contains(&value);
                                        let label = value.title();
                                        let highlighted = self.task_filters_focus
                                            == TaskFiltersFocus::Values
                                            && self.task_filters_active_value_index
                                                == display_index;

                                        let row_id = (
                                            gpui::ElementId::from((
                                                "task_filters_value",
                                                entity_id,
                                            )),
                                            format!("agent_status:{label}"),
                                        );

                                        let row = div()
                                            .id(row_id)
                                            .flex()
                                            .flex_row()
                                            .w_full()
                                            .items_center()
                                            .gap(theme.spacing.sm)
                                            .px(theme.spacing.sm)
                                            .h(menu_item_height)
                                            .rounded(theme.radius.sm)
                                            .when(highlighted, |this| this.bg(theme.colors.accent))
                                            .when(
                                                self.task_filters_input_source
                                                    == TaskFiltersInputSource::Mouse,
                                                |this| {
                                                    this.hover(|this| {
                                                        this.bg(theme.colors.accent.opacity(0.75))
                                                    })
                                                },
                                            )
                                            .cursor_pointer()
                                            .on_mouse_move(cx.listener(move |this, _, _, cx| {
                                                let mut did_change = false;
                                                if this.task_filters_input_source
                                                    != TaskFiltersInputSource::Mouse
                                                {
                                                    this.task_filters_input_source =
                                                        TaskFiltersInputSource::Mouse;
                                                    did_change = true;
                                                }
                                                if this.task_filters_active_value_index
                                                    != display_index
                                                {
                                                    this.task_filters_active_value_index =
                                                        display_index;
                                                    did_change = true;
                                                }
                                                if this.task_filters_focus
                                                    != TaskFiltersFocus::Values
                                                {
                                                    this.task_filters_focus =
                                                        TaskFiltersFocus::Values;
                                                    did_change = true;
                                                }
                                                if did_change {
                                                    cx.notify();
                                                }
                                            }))
                                            .on_mouse_down(gpui::MouseButton::Left, {
                                                let workspace = workspace.clone();
                                                move |_, _, cx| {
                                                    workspace.update(cx, |this, cx| {
                                                        this.task_filters_focus =
                                                            TaskFiltersFocus::Values;
                                                        this.task_filters_active_value_index =
                                                            display_index;
                                                        this.toggle_agent_status_filter(value, cx);
                                                    });
                                                }
                                            })
                                            .child(checkbox(selected))
                                            .child(
                                                div()
                                                    .flex_1()
                                                    .min_w_0()
                                                    .text_xs()
                                                    .text_color(theme.colors.foreground)
                                                    .truncate()
                                                    .child(label),
                                            );

                                        value_list = value_list.child(row);
                                    }
                                }
                            }
                        }

                        let values_body = div().flex().flex_col().child(value_list);

                        overlay_surface(&theme, OverlaySurfaceKind::Menu, theme.radius.lg)
                            .w(submenu_width)
                            .pt(theme.spacing.sm)
                            .pb(theme.spacing.sm)
                            .px(theme.spacing.sm)
                            .shadow_md()
                            .occlude()
                            .child(values_body)
                    });

                    let menu_container = div()
                        .key_context("TaskFilters")
                        .track_focus(&self.task_filters_focus_handle)
                        .on_mouse_down(gpui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                        .child(primary_menu)
                        .relative()
                        .when_some(submenu, move |this, submenu| {
                            let top = submenu_top.unwrap_or(px(0.0));
                            this.child(
                                div()
                                    .absolute()
                                    .top(top)
                                    .left(primary_menu_width - submenu_overlap)
                                    .child(submenu),
                            )
                        });

                    host = host
                        .on_mouse_down(gpui::MouseButton::Left, {
                            let workspace = workspace.clone();
                            move |_, _, cx| {
                                workspace.update(cx, |this, cx| {
                                    if !this.task_filters_open {
                                        return;
                                    }

                                    this.task_filters_open = false;
                                    this.task_filters_hovered_category = None;
                                    this.ui_updates.bump();
                                    cx.notify();
                                });
                            }
                        })
                        .child(
                            div()
                                .absolute()
                                .top(tab_top + filter_tab_height + theme.spacing.xs)
                                .left(tab_left)
                                .opacity(menu_opacity)
                                .child(menu_container),
                        );
                }
            }

            host
        };

        div()
            .flex()
            .flex_col()
            .size_full()
            .relative()
            .bg(theme.colors.background)
            .key_context("Workspace")
            .on_action(cx.listener(Self::open_task_filters))
            .on_action(cx.listener(Self::close_task_filters))
            .on_action(cx.listener(Self::task_filters_toggle_chip_focus))
            .on_action(cx.listener(Self::task_filters_move_up))
            .on_action(cx.listener(Self::task_filters_move_down))
            .on_action(cx.listener(Self::task_filters_move_left))
            .on_action(cx.listener(Self::task_filters_move_right))
            .on_action(cx.listener(Self::task_filters_activate))
            .on_action(cx.listener(Self::task_filters_clear_focused_chip))
            .child(body.child(graph_host))
            .track_focus(&self.focus_handle(cx))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefreshReason {
    Startup,
    Manual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChromePanel {
    EpicMenu,
    Settings,
}

struct ChromeState {
    did_startup_refresh: bool,
    panel: Option<ChromePanel>,
    epics: Vec<redesmyn_protocol::client::EpicSummary>,
    selected_epic_slug: Option<String>,
    control_plane_status: Option<redesmyn_protocol::client::StatusResponse>,
    refresh: UserActionState,
    refresh_task: Option<Task<()>>,
    theme_error: Option<SharedString>,
    epic_scroll: ScrollHandle,
}

impl ChromeState {
    fn new() -> Self {
        Self {
            did_startup_refresh: false,
            panel: None,
            epics: Vec::new(),
            selected_epic_slug: None,
            control_plane_status: None,
            refresh: UserActionState::default(),
            refresh_task: None,
            theme_error: None,
            epic_scroll: ScrollHandle::new(),
        }
    }
}
