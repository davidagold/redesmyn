use crate::predictions::{Endpoint, Service};

use super::error::ServiceError;
use super::predictions::{BatchPredictor, Schema};
use actix_web::Resource;
use actix_web::{dev::ServerHandle, web, HttpServer};
use pyo3::{PyResult, Python};
use serde::Deserialize;
use tokio::task::JoinError;
use std::collections::VecDeque;
use std::env;
use tokio::{signal, task::JoinHandle};
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

pub trait Serves {
    fn register<S>(&mut self, endpoint: S) -> &Self
    where
        S: ResourceFactory + Clone + Send + 'static;

        fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError>;
}

pub struct Server {
    resource_factories: VecDeque<BoxedResourceFactory>,
}

impl Default for Server {
    fn default() -> Self {
        Server { resource_factories: VecDeque::new() }
    }
}

pub trait ResourceFactory: Send {
    fn new_resource(&self) -> Resource;

    fn clone_boxed(&self) -> Box<dyn ResourceFactory>;
}

impl<R> ResourceFactory for Endpoint<BatchPredictor<R>, R>
where
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
{
    fn new_resource(&self) -> Resource {
        let mut service = BatchPredictor::<R>::new(&self.path.clone());
        service.run();
        web::resource(self.path.clone())
            .app_data(web::Data::new(service))
            .route(web::post().to(<BatchPredictor<R> as Service>::invoke))
    }

    fn clone_boxed(&self) -> Box<dyn ResourceFactory> {
        Box::new(self.clone())
    }
}

struct BoxedResourceFactory(Box<dyn ResourceFactory>);

impl Clone for BoxedResourceFactory {
    fn clone(&self) -> Self {
        BoxedResourceFactory(self.0.clone_boxed())
    }
}

impl Serves for Server {
    fn register<S>(&mut self, endpoint: S) -> &Self
    where
        S: ResourceFactory + Clone + Send + 'static,
    {
        info!("Registering endpoint with path: ...");
        self.resource_factories
            .push_back(BoxedResourceFactory(Box::new(endpoint)));
        self
    }

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError> {
        let http_server = HttpServer::new(move || {
            let app = self
                .resource_factories
                .clone()
                .into_iter()
                .fold(actix_web::App::new(), |app, resource_factory| {
                    app.service(resource_factory.0.new_resource())
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

        info!("Starting server...");
        let server_handle = server.handle();
        tokio::spawn(async move { await_shutdown(server_handle).await });
        Ok(tokio::spawn(async move { server.await }))
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
