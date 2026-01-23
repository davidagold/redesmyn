mod app;
mod command_palette;
mod control_plane_client;
mod foundations_demo;
mod root_view;
mod screenshot;
mod test_artifacts;
mod ui_driver;

use gpui::{AppContext as _, Focusable as _};

use crate::foundations_demo::FoundationsDemo;

fn main() {
    redesmyn_logging::init();

    let span = redesmyn_logging::redesmyn_info_span!("startup", component = "redesmyn-desktop");
    redesmyn_logging::span::record_run_id(&span, std::process::id());

    let _guard = span.enter();
    redesmyn_logging::tracing::info!("starting");

    if std::env::var_os("REDESMYN_UI_FOUNDATIONS_DEMO").is_some() {
        redesmyn_logging::tracing::info!("starting ui foundations demo");
        run_foundations_demo();
        return;
    }

    let config = match redesmyn_config::load_rust_config(Default::default()) {
        Ok(config) => config,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to load config");
            std::process::exit(2);
        }
    };

    let mut desktop = match crate::app::DesktopApp::start(config) {
        Ok(desktop) => desktop,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to start desktop app");
            std::process::exit(2);
        }
    };

    let ui_config = desktop.config().clone();
    let daemon_host_id = desktop.daemon_host_id();
    let tokio_handle = desktop.tokio_handle();
    let session_control_plane_client = desktop.take_control_plane_client();
    let session_viewer_fixture = desktop.take_session_viewer_fixture();
    let chrome_control_plane_client = desktop.connect_control_plane_client(64);
    let (ui_driver_rx, _ui_driver_server) = match std::env::var_os("REDESMYN_UI_DRIVER_SOCKET_PATH")
    {
        Some(path) => {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let socket_path = std::path::PathBuf::from(path);

            let server = match crate::ui_driver::start_ui_driver_server(
                crate::ui_driver::UiDriverServerConfig {
                    socket_path,
                    max_frame_len: crate::ui_driver::DEFAULT_MAX_FRAME_LEN,
                },
                tx,
            ) {
                Ok(server) => Some(server),
                Err(err) => {
                    redesmyn_logging::tracing::error!(error = %err, "failed to start ui driver server");
                    std::process::exit(2);
                }
            };

            (Some(rx), server)
        }
        None => (None, None),
    };

    gpui::Application::new().run(move |cx| {
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                redesmyn_logging::tracing::info!("last window closed; quitting");
                cx.quit();
            }
        })
        .detach();

        if let Err(error) = redesmyn_ui::UiContext::init(cx) {
            redesmyn_logging::tracing::error!(error = %error, "failed to init ui context");
        }
        redesmyn_ui::components::bind_text_input_keys(cx);
        crate::command_palette::bind_command_palette_keys(cx);

        let window_size = gpui::Size::new(
            gpui::px(ui_config.desktop.window.width as f32),
            gpui::px(ui_config.desktop.window.height as f32),
        );

        let options = gpui::WindowOptions {
            window_bounds: Some(gpui::WindowBounds::centered(window_size, cx)),
            ..Default::default()
        };

        let window = cx
            .open_window(options, move |_, cx| {
                let model = cx.new(|_| {
                    crate::root_view::DesktopModel::new(
                        ui_config.clone(),
                        daemon_host_id,
                        tokio_handle.clone(),
                        session_control_plane_client,
                        chrome_control_plane_client,
                        session_viewer_fixture,
                        ui_driver_rx,
                    )
                });
                cx.new(|cx| crate::root_view::RootView::new(model, cx))
            })
            .expect("window open should succeed");

        window
            .update(cx, |view: &mut crate::root_view::RootView, window, cx| {
                window.focus(&view.focus_handle(cx));
                cx.activate(true);
            })
            .ok();
    });

    desktop.shutdown();
}

fn run_foundations_demo() {
    gpui::Application::new().run(|cx| {
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                redesmyn_logging::tracing::info!("last window closed; quitting");
                cx.quit();
            }
        })
        .detach();

        if let Err(error) = redesmyn_ui::UiContext::init(cx) {
            redesmyn_logging::tracing::error!(error = %error, "failed to init ui context");
        }
        redesmyn_ui::components::bind_text_input_keys(cx);

        let bounds =
            gpui::Bounds::centered(None, gpui::size(gpui::px(1120.0), gpui::px(760.0)), cx);
        let window = cx
            .open_window(
                gpui::WindowOptions {
                    window_bounds: Some(gpui::WindowBounds::Windowed(bounds)),
                    ..Default::default()
                },
                |_, cx| cx.new(FoundationsDemo::new),
            )
            .expect("open_window failed");

        window
            .update(cx, |view: &mut FoundationsDemo, window, cx| {
                window.focus(&view.text_input_focus_handle(cx));
                cx.activate(true);
            })
            .ok();
    });
}
