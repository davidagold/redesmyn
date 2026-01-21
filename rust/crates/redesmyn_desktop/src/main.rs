mod app;
mod command_palette;
mod foundations_demo;
mod root_view;

use gpui::AppContext as _;

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
    let control_plane_client = desktop.take_control_plane_client();

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

        let _window = cx
            .open_window(options, move |_, cx| {
                let model = cx.new(|_| {
                    crate::root_view::DesktopModel::new(
                        ui_config.clone(),
                        daemon_host_id,
                        control_plane_client,
                    )
                });
                cx.new(|cx| crate::root_view::RootView::new(model, cx))
            })
            .expect("window open should succeed");
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
