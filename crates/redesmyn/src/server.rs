use crate::predictions::Handle;

use super::error::ServiceError;
use super::predictions::{PredictionService, Schema};
use actix_web::{dev::ServerHandle, web, HttpServer};
use pyo3::{PyResult, Python};
use serde::Deserialize;
use std::collections::VecDeque;
use std::env;
use std::sync::Arc;
use tokio::{signal, task::JoinHandle};
use tracing::error;
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

trait Serves {
    type R: Schema<Self::R> + Sync + Send + 'static + for<'a> Deserialize<'a>;

    fn add_service(self, handler: Arc<dyn Handle<R = Self::R> + Sync + Send>) -> Self;

    fn serve(self) -> Result<JoinHandle<()>, ServiceError>;
}

pub struct Server<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    include_with: VecDeque<Arc<dyn Handle<R = R> + Sync + Send>>,
}

impl<R> Default for Server<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        Server { include_with: VecDeque::new() }
    }
}

impl<R> Serves for Server<R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    type R = R;

    fn add_service(mut self, service: Arc<dyn Handle<R = R> + Sync + Send>) -> Self {
        self.include_with.push_back(service);
        self
    }

    fn serve(self) -> Result<JoinHandle<()>, ServiceError> {
        let http_server = HttpServer::new(move || {
            let app = self.include_with.iter().fold(actix_web::App::new(), |app, service| {
                app.service(
                    web::resource((*service).path())
                        // .app_data(web::Data::new(self))
                        .route(web::post().to(<PredictionService<Self::R> as Handle>::invoke::<R>)),
                )
            });
            app
        })
        .disable_signals();

        let server = http_server.bind("127.0.0.1:8080")?.run();

        let subscribe_layer = tracing_subscriber::fmt::layer().json();
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(subscribe_layer)
            .init();

        let pwd = match env::current_dir() {
            Ok(pwd) => pwd,
            Err(err) => {
                error!("Failed to get working directory: {err}");
                return Err(ServiceError::IoError(err));
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
            error!("Failed to initialize Python process: {err}");
            return Err(err.into());
        };

        let server_handle = server.handle();
        tokio::spawn(async move { await_shutdown(server_handle).await });
        let handle = tokio::spawn(async move {
            server.await;
        });
        Ok(handle)
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
