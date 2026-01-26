mod command_palette_overlay;

use std::sync::Arc;
use std::time::Duration;

use gpui::{
    App, AsyncApp, ClickEvent, Context, Entity, FocusHandle, Focusable, Render, ScrollHandle,
    SharedString, Subscription, Task, WeakEntity, Window, div, prelude::*, px,
};
use tokio::sync::{mpsc, watch};

use redesmyn_protocol::ui_driver::{
    CaptureScreenshotResponse, CreateChatSessionResponse, TriggerRefreshResponse, UiComposerState,
    UiDriverRequestPayload, UiDriverResponse, UiDriverResponseResult, UiErrorCallout,
    UiInFlightAction, UiLeftPaneState, UiPrimaryView, UiSelectionState, UiSnapshot,
    UiSnapshotPredicate, WaitForUiIdleRequest, WaitForUiIdleResponse, WaitForUiSnapshotRequest,
    WaitForUiSnapshotResponse,
};
use redesmyn_protocol::{ErrorCategory, ErrorEnvelope, RepoScope, Timestamp};
use redesmyn_transport::client::in_proc::InProcEndpoint as ClientInProcEndpoint;
use redesmyn_ui_session::SessionView;

use redesmyn_ui::UiContext;
use redesmyn_ui::components::{
    ButtonKind, Callout, CalloutKind, IconButton, ProgressPill, ScrollArea, SplitPane,
    SplitPaneAxis, SplitPaneEvent, SplitPaneState, TextButton,
};
use redesmyn_ui::settings::ThemePreference;
use redesmyn_ui::utils::{UserActionState, theme_for_window};
use redesmyn_ui_graph::GraphView;

use crate::app::SessionViewerFixtureEmitter;
use crate::command_palette::{
    CloseCommandPalette, SelectNextCommand, SelectPreviousCommand, ToggleCommandPalette,
};
use crate::control_plane_client::{ControlPlaneClient, ControlPlaneClientError};

use self::command_palette_overlay::CommandPaletteOverlay;

#[derive(Debug)]
pub struct DesktopModel {
    config: Arc<redesmyn_config::RustConfig>,
    daemon_host_id: Option<redesmyn_ids::HostId>,
    session_control_plane_client: Option<ClientInProcEndpoint>,
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
        chrome_control_plane_client: Option<ClientInProcEndpoint>,
        session_viewer_fixture: Option<SessionViewerFixtureEmitter>,
        ui_driver_rx: Option<mpsc::UnboundedReceiver<crate::ui_driver::UiDriverCommand>>,
    ) -> Self {
        Self {
            config,
            daemon_host_id,
            session_control_plane_client,
            chrome_control_plane_client: chrome_control_plane_client
                .map(|conn| ControlPlaneClient::new(tokio_handle, conn)),
            session_viewer_fixture,
            ui_driver_rx,
        }
    }

    pub fn take_control_plane_client(&mut self) -> Option<ClientInProcEndpoint> {
        self.session_control_plane_client.take()
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

        self.session_pane
            .update(cx, move |pane, cx| pane.set_selected_epic(selected_for_session_pane, cx));
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

        let selection = UiSelectionState {
            epic_id: None,
            epic_slug: self
                .chrome
                .selected_epic_slug
                .clone()
                .unwrap_or_else(String::new),
            task_id: None,
            task_slug: String::new(),
            edge_id: None,
        };

        let graph = redesmyn_protocol::ui_driver::UiGraphState {
            load_state: workspace.graph_state,
            node_count: workspace.graph_node_count,
            edge_count: workspace.graph_edge_count,
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

        let epic_slug = self
            .selected_epic
            .as_ref()
            .map(|epic| epic.slug.clone())
            .unwrap_or_else(|| "Select epic".to_string());

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

        let mut header_actions = div().flex().flex_row().items_center().gap(theme.spacing.sm);

        if let Some(label) = in_flight_label {
            header_actions = header_actions.child(ProgressPill::new(label));
        }

        let can_act_on_epic = self.selected_epic.is_some() && !self.load.in_flight;
        let has_pinned = self.pinned_session_id.is_some();

        let create_button = TextButton::new(("chat_create", cx.entity_id()), "Create chat")
            .kind(ButtonKind::Secondary)
            .disabled(!can_act_on_epic || self.create_and_pin.in_flight)
            .disabled_reason("Loading…")
            .on_click({
                let view = view.clone();
                move |_, _, cx| view.update(cx, |this, cx| this.create_and_pin_chat(cx))
            });

        let pin_existing_button =
            TextButton::new(("chat_pin_existing", cx.entity_id()), "Pin existing…")
                .kind(ButtonKind::Ghost)
                .disabled(!can_act_on_epic || self.pin_existing.in_flight)
                .disabled_reason("Loading…")
                .on_click({
                    let view = view.clone();
                    move |_, _, cx| view.update(cx, |this, cx| this.open_pin_existing(cx))
                });

        let unpin_button = TextButton::new(("chat_unpin", cx.entity_id()), "Unpin")
            .kind(ButtonKind::Ghost)
            .disabled(!can_act_on_epic || !has_pinned || self.unpin.in_flight)
            .disabled_reason("No pinned chat")
            .on_click({
                let view = view.clone();
                move |_, _, cx| view.update(cx, |this, cx| this.unpin_chat(cx))
            });

        let close_button = TextButton::new(("chat_close", cx.entity_id()), "Close")
            .kind(ButtonKind::Ghost)
            .disabled(!can_act_on_epic || !has_pinned || self.close_session.in_flight)
            .disabled_reason("No pinned chat")
            .on_click({
                let view = view.clone();
                move |_, _, cx| view.update(cx, |this, cx| this.close_current_chat(cx))
            });

        header_actions = header_actions
            .child(create_button)
            .child(pin_existing_button)
            .when(has_pinned, |this| {
                this.child(unpin_button).child(close_button)
            });
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
                div()
                    .flex()
                    .flex_row()
                    .items_center()
                    .gap(theme.spacing.sm)
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground)
                            .child("Chat"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .text_color(theme.colors.foreground_muted)
                            .child(epic_slug),
                    ),
            )
            .child(header_actions);

        let mut body = div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .px(theme.spacing.md)
            .py(theme.spacing.md)
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
            None => {
                body = body.child(
                    Callout::new("Select an epic to view or pin a chat session.")
                        .kind(CalloutKind::Info)
                        .title("Chat"),
                );
            }
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
                let title = self
                    .pinned_chat_summary()
                    .and_then(|summary| summary.title.clone())
                    .unwrap_or_else(|| "Pinned chat".to_string());
                let closed = self
                    .pinned_chat_summary()
                    .and_then(|summary| summary.closed_at)
                    .is_some();

                let subtitle = self
                    .pinned_session_id
                    .map(|id| {
                        if closed {
                            format!("{id} · closed")
                        } else {
                            id.to_string()
                        }
                    })
                    .unwrap_or_default();

                body = body.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(theme.spacing.xs)
                        .child(
                            div()
                                .text_sm()
                                .text_color(theme.colors.foreground)
                                .child(title),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(theme.colors.foreground_muted)
                                .child(subtitle),
                        ),
                );

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
    ui_updates: UiUpdateCounter,
    model: Entity<DesktopModel>,
    graph_view: Entity<GraphView>,
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
        Self {
            focus_handle: cx.focus_handle(),
            ui_updates,
            model,
            graph_view: cx.new(GraphView::new_empty),
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
            self.graph_view = cx.new(GraphView::new_empty);

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

        let entity_id = cx.entity_id();
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

            host
        };

        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(theme.colors.background)
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
