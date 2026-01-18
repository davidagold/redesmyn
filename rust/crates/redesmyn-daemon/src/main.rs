fn main() {
    redesmyn_logging::init();

    let span = redesmyn_logging::redesmyn_info_span!("startup", component = "redesmyn-daemon");
    redesmyn_logging::span::record_run_id(&span, std::process::id());

    let _guard = span.enter();
    redesmyn_logging::tracing::info!("starting");
}
