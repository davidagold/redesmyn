mod app;
mod root_view;

use gpui::AppContext as _;

fn main() {
    redesmyn_logging::init();

    let span = redesmyn_logging::redesmyn_info_span!("startup", component = "redesmyn-desktop");
    redesmyn_logging::span::record_run_id(&span, std::process::id());

    let _guard = span.enter();
    redesmyn_logging::tracing::info!("starting");

    let config = match redesmyn_config::load_rust_config(Default::default()) {
        Ok(config) => config,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to load config");
            std::process::exit(2);
        }
    };

    let desktop = match crate::app::DesktopApp::start(config) {
        Ok(desktop) => desktop,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to start desktop app");
            std::process::exit(2);
        }
    };

    let ui_config = desktop.config().clone();
    let daemon_host_id = desktop.daemon_host_id();

    gpui::Application::new().run(move |cx| {
        cx.on_window_closed(|cx| {
            if cx.windows().is_empty() {
                redesmyn_logging::tracing::info!("last window closed; quitting");
                cx.quit();
            }
        })
        .detach();

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
                let model = cx.new(|_| crate::root_view::DesktopModel::new(ui_config.clone(), daemon_host_id));
                cx.new(|_| crate::root_view::RootView::new(model))
            })
            .expect("window open should succeed");
    });

    desktop.shutdown();
}
