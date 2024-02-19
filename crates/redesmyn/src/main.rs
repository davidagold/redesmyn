use std::env;

use actix_web::{dev::ServerHandle, web, App, HttpServer};
use pyo3::{PyResult, Python};
use redesmyn::predictions::{self, ServiceError, ToyRecord};
use tokio::{
    signal,
    sync::{mpsc, oneshot},
};
use tracing::instrument;
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

#[tokio::main(flavor = "multi_thread", worker_threads = 8)]
#[instrument]
async fn main() -> Result<(), ServiceError> {
    let subscribe_layer = tracing_subscriber::fmt::layer().json();
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(subscribe_layer)
        .init();

    let pwd = env::current_dir()?;
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
        let msg = format!("Failed to initialize Python process: {}", err);
        tracing::error!("{}", msg);
        return Err(ServiceError::Error(msg))
    };

    let (tx, rx) = mpsc::unbounded_channel();
    let server = match HttpServer::new(move || {
        let app_state = predictions::PredictionService::<ToyRecord>::new(tx.clone());
        App::new()
            .app_data(web::Data::new(app_state))
            .service(predictions::submit_prediction_request)
    })
    .disable_signals()
    .bind("127.0.0.1:8080") {
        Ok(server) => server.run(),
        Err(err) => {
            tracing::error!("Failed to start server: {}", err);
            return Err(err.into())
        }
    };

    let server_handle = server.handle();
    tokio::spawn(server);

    let (tx_abort, rx_abort) = oneshot::channel::<()>();
    let predict_loop_handle = tokio::spawn(predictions::batch_predict_loop(rx, rx_abort));
    tokio::spawn(await_shutdown(server_handle, tx_abort));
    predict_loop_handle.await.map_err(|err| err.into())
}

async fn await_shutdown(server_handle: ServerHandle, tx_abort: oneshot::Sender<()>) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    if tx_abort.send(()).is_err() {
        tracing::error!("Failed to send cancel signal.");
    }
    server_handle.stop(true).await;
}
