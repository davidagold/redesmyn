fn main() {
    redesmyn_logging::init();

    let span = redesmyn_logging::redesmyn_info_span!("startup", component = "redesmyn-daemon");
    redesmyn_logging::span::record_run_id(&span, std::process::id());

    let _guard = span.enter();
    redesmyn_logging::tracing::info!("starting");

    let _config = match redesmyn_config::load_rust_config(Default::default()) {
        Ok(config) => config,
        Err(err) => {
            redesmyn_logging::tracing::error!(error = %err, "failed to load config");
            std::process::exit(2);
        }
    };
}
