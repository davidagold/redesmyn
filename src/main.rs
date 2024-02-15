use std::env;

use actix_web::{web, App, HttpServer};
use pyo3::Python;
use rs_model_server::predictions::{self, ServiceError};
use tokio::sync::mpsc;
use tracing::{event, instrument, Level};
use tracing_subscriber::{self, fmt::format::FmtSpan, layer::SubscriberExt, prelude::*, EnvFilter};

#[tokio::main]
#[instrument]
async fn main() -> Result<(), ServiceError> {
    // let path_venv = "/Users/davidgold/.local/share/virtualenvs/notebooks-C37d9m95";
    // env::set_var("", format!("{}/bin/python3", path_venv));
    // env::set_var("PYTHONHOME", format!("{}/lib/", path_venv));
    // event!(Level::INFO, "Found Python virtual environment {}", path_python_venv);

    // env::set_var("PYTHONPATH", format!("{}/lib/python3.11/site-packages", path_venv));

    let subscribe_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_span_events(FmtSpan::CLOSE);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(subscribe_layer)
        .init();

    // let mlflow_tracking_dir = env::var("MLFLOW_TRACKING_DIR")?;
    // let mlflow_registry_dir = env::var("MLFLOW_REGISTRY_DIR")?;
    // event!(Level::INFO, "Found MLFLOW_TRACKING_DIR={}", mlflow_tracking_dir);
    // event!(Level::INFO, "Found MLFLOW_REGISTRY_DIR={}", mlflow_registry_dir);

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

    let (tx, rx) = mpsc::channel(512);

    tokio::spawn(predictions::batch_predict_loop(rx));

    HttpServer::new(move || {
        let app_state = predictions::AppState::new(tx.clone());
        App::new()
            .app_data(web::Data::new(app_state))
            .service(predictions::submit_prediction_request)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
    .map_err(std::io::Error::into)
}
