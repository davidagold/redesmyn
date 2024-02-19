use std::env;

use actix_web::{dev::ServerHandle, web, App, HttpServer};
use pyo3::{PyResult, Python};
use redesmyn::predictions::{self, ToyRecord};
use tokio::{
    signal,
    sync::{mpsc, oneshot},
};
use tracing::instrument;
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
#[instrument]
async fn main() -> () {
    let subscribe_layer = tracing_subscriber::fmt::layer().json();
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(subscribe_layer)
        .init();

    let pwd = match env::current_dir() {
        Ok(pwd) => pwd,
        Err(err) => {
            return tracing::error!("Failed to get working directory: {}", err);
        }
    };
    tracing::info!("Starting `main` from directory {:?}", pwd);

    pyo3::prepare_freethreaded_python();
    if let Err(err) = Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let version = sys.getattr("version")?.extract::<String>()?;
        let python_path = sys.getattr("path")?.extract::<Vec<String>>()?;
        tracing::info!("Found Python version: {}", version);
        tracing::info!("Found Python path: {:?}", python_path);
        sys.getattr("path")?
            .getattr("insert")
            .and_then(move |insert| insert.call((0, pwd), None))?;

        PyResult::<()>::Ok(())
    }) {
        return tracing::error!("Failed to initialize Python process: {}", err);
    };

    let (tx, rx) = mpsc::unbounded_channel();
    let http_server = HttpServer::new(move || {
        let app_state = predictions::PredictionService::<ToyRecord>::new(tx.clone());
        App::new()
            .app_data(web::Data::new(app_state))
            .service(predictions::submit_prediction_request)
    })
    .disable_signals();

    let server = match http_server.bind("127.0.0.1:8080") {
        Ok(http_server) => http_server.run(),
        Err(err) => {
            return tracing::error!("Failed to start server: {}", err);
        }
    };
    let server_handle = server.handle();
    tokio::spawn(server);

    let (tx_abort, rx_abort) = oneshot::channel::<()>();
    let predict_loop_handle = tokio::spawn(predictions::batch_predict_loop(rx, rx_abort));
    tokio::spawn(await_shutdown(server_handle, tx_abort));
    match predict_loop_handle.await {
        Ok(()) => tracing::info!("Successfully exited predict loop."),
        Err(err) => tracing::error!("Failure while exiting predict loop: {}", err),
    };
}

async fn await_shutdown(server_handle: ServerHandle, tx_abort: oneshot::Sender<()>) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    if tx_abort.send(()).is_err() {
        tracing::error!("Failed to send cancel signal.");
    }
    server_handle.stop(true).await;
}
