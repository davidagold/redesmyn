use std::env;

use actix_web::{web, App, HttpServer};
use pyo3::Python;
use rs_model_server::predictions::{self, ServiceError};
use tokio::{
    signal,
    sync::{mpsc, oneshot},
    task::JoinError,
};
use tracing::{event, instrument, Level};
use tracing_subscriber::{self, fmt::format::FmtSpan, layer::SubscriberExt, prelude::*, EnvFilter};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
#[instrument]
async fn main() -> Result<(), ServiceError> {
    let subscribe_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_span_events(FmtSpan::CLOSE);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(subscribe_layer)
        .init();

    pyo3::prepare_freethreaded_python();
    match Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let version = sys.getattr("version")?.extract::<String>()?;
        let python_path = sys.getattr("path")?.extract::<Vec<String>>()?;
        event!(Level::INFO, "Found Python version: {}", version);
        event!(Level::INFO, "Found Python path: {:?}", python_path);
        sys.getattr("path")?
            .getattr("insert")
            .and_then(move |insert| insert.call((0, env::current_dir()?), None))?;

        Ok::<(), std::io::Error>(())
    }) {
        Ok(_) => Ok(()),
        Err(err) => Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to initialize Python process: {}", err),
        )),
    }?;

    event!(
        Level::INFO,
        "Starting `main` in working directory {:?}",
        env::current_dir()?
    );

    let (tx, rx) = mpsc::unbounded_channel();
    let server = HttpServer::new(move || {
        let app_state = predictions::AppState::new(tx.clone());
        App::new()
            .app_data(web::Data::new(app_state))
            .service(predictions::submit_prediction_request)
    })
    .disable_signals()
    .workers(256)
    .bind("127.0.0.1:8080")?
    .run();
    let server_handle = server.handle();
    tokio::spawn(async { server.await });

    let (tx_abort, rx_abort) = oneshot::channel::<()>();
    let predict_loop_handle = tokio::spawn(predictions::batch_predict_loop(rx, rx_abort));
    tokio::spawn(async move {
        let _ = signal::ctrl_c().await;
        event!(Level::INFO, "Received shutdown signal.");
        if let Err(_) = tx_abort.send(()) {
            event!(Level::ERROR, "Failed to send cancel signal.");
        }
        server_handle.stop(true).await;
    });
    predict_loop_handle.await.map_err(JoinError::into)
}
