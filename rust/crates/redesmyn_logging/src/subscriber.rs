use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt as _;

use crate::{LogFormat, LoggingConfig};

const DEFAULT_FILTER_DIRECTIVES: &str = "info";

pub(crate) fn build_dispatch<W>(config: LoggingConfig, make_writer: W) -> tracing::Dispatch
where
    W: for<'a> tracing_subscriber::fmt::MakeWriter<'a> + Send + Sync + 'static,
{
    let filter = build_env_filter(&config);

    match config.format {
        LogFormat::Pretty => {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_writer(make_writer)
                .with_file(true)
                .with_line_number(true)
                .with_thread_names(true)
                .with_target(false)
                .with_ansi(false)
                .compact();

            tracing::Dispatch::new(tracing_subscriber::registry().with(filter).with(fmt_layer))
        }
        LogFormat::Json => {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_writer(make_writer)
                .with_file(true)
                .with_line_number(true)
                .with_thread_names(true)
                .with_target(false)
                .with_ansi(false)
                .json()
                .with_current_span(true)
                .with_span_list(true);

            tracing::Dispatch::new(tracing_subscriber::registry().with(filter).with(fmt_layer))
        }
    }
}

fn build_env_filter(config: &LoggingConfig) -> EnvFilter {
    match config.filter_directives.as_deref() {
        Some(directives) if !directives.trim().is_empty() => EnvFilter::try_new(directives)
            .unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER_DIRECTIVES)),
        _ => EnvFilter::new(DEFAULT_FILTER_DIRECTIVES),
    }
}
