use crate::cache::Cache;
use crate::common::LogConfig;
use crate::metrics::{EmfInterest, EmfMetrics};
use crate::predictions::{HandlerArgs, PredictionJob, Service};
use crate::schema::Schema;

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use pyo3::{PyResult, Python};
use serde::Deserialize;
use std::collections::VecDeque;
use std::env;
use tokio::sync::mpsc;
use tokio::{signal, task::JoinHandle};
use tracing::instrument;
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

trait ResourceFactory: Sync + Send {
    fn new_resource(&mut self, path: &str) -> Result<Resource, ServiceError>;

    fn clone_boxed(&self) -> Box<dyn ResourceFactory>;

    fn start_service(&mut self) -> Result<JoinHandle<()>, ServiceError>;
}

pub(crate) struct BoxedResourceFactory {
    factory: Box<dyn ResourceFactory>,
    path: String,
}

impl Clone for BoxedResourceFactory {
    fn clone(&self) -> Self {
        BoxedResourceFactory {
            factory: self.factory.clone_boxed(),
            path: self.path.clone(),
        }
    }
}

impl<S, O> ResourceFactory for S
where
    S: Service + Clone + Sync + Send + 'static,
    S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
    S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
    S::H: Handler<HandlerArgs<S::R, S::T>, Output = O>,
    O: Responder + 'static,
{
    fn new_resource(&mut self, path: &str) -> Result<Resource, ServiceError> {
        let handler = self.get_handler_fn();
        let resource = web::resource(path)
            // .app_data(web::Data::<Self>::new(self.clone()))
            .app_data(web::Data::<mpsc::Sender<PredictionJob<S::T, S::R>>>::new(self.job_sender()))
            .app_data(web::Data::<Schema>::new(self.get_schema()))
            .route(web::post().to(handler));
        Ok(resource)
    }

    fn clone_boxed(&self) -> Box<dyn ResourceFactory> {
        Box::new(self.clone())
    }

    fn start_service(&mut self) -> Result<JoinHandle<()>, ServiceError> {
        self.run()
    }
}

#[derive(Default)]
pub struct Server {
    pub(crate) factories: VecDeque<BoxedResourceFactory>,
    pythonpath: Vec<String>,
    config_log: LogConfig,
}

impl Clone for Server {
    fn clone(&self) -> Self {
        let mut server = Server::default();
        for factory in self.factories.iter() {
            server.factories.push_back(factory.clone());
        }
        for path in self.pythonpath.iter() {
            server.pythonpath.push(path.clone());
        }
        server.log_config(self.config_log.clone());
        server
    }
}

impl Server {
    pub fn push_pythonpath(&mut self, path: &str) {
        self.pythonpath.push(path.to_string());
    }

    pub fn log_config(&mut self, config: LogConfig) {
        self.config_log = config;
    }
}

impl Server {
    pub fn register<S, O>(&mut self, service: S) -> &Self
    where
        S: Service + Clone + Sync + Send + 'static,
        S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
        S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
        S::H: Handler<HandlerArgs<S::R, S::T>, Output = O> + Sync + Send,
        O: Responder + 'static,
    {
        info!("Registering endpoint with path: ...");
        let path = service.get_path();
        self.factories.push_back(BoxedResourceFactory { factory: Box::new(service), path });
        self
    }

    #[instrument(skip_all)]
    pub fn serve(&mut self) -> Result<actix_web::dev::Server, ServiceError> {
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(self.config_log.layer().with_filter(EmfInterest::Never))
            .with(EmfMetrics::new(10, "./metrics.log".into()))
            .init();

        let pwd = match env::current_dir() {
            Ok(pwd) => pwd,
            Err(err) => {
                error!("Failed to get working directory: {err}");
                return Err(ServiceError::IoError(err));
            }
        };
        println!("Starting `main` from directory {}", pwd.to_str().unwrap());
        tracing::info!("Starting `main` from directory {}", pwd.to_str().unwrap());

        pyo3::prepare_freethreaded_python();
        if let Err(err) = Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let version = sys.getattr("version")?.extract::<String>()?;
            tracing::info!("Found Python version: {}", version);

            let insert = sys.getattr("path")?.getattr("insert")?;
            let additional_paths = [&(pwd.to_str().unwrap().to_string())];
            for path in self.pythonpath.iter().chain(additional_paths) {
                insert.call((0, path), None)?;
            }

            let pythonpath = sys.getattr("path")?.extract::<Vec<String>>()?;
            let str_python_path =
                serde_json::to_string_pretty(&pythonpath).expect("Failed to serialize `sys.path`.");
            println!("Found Python path: {str_python_path}");
            info!(pythonpath = format!("{}", str_python_path.as_str()));
            PyResult::<()>::Ok(())
        }) {
            error!("{}", format!("Failed to initialize Python process: {err}"));
            return Err(err.into());
        };

        // The `factory` argument to `HttpServer::new()` is invoked for each worker,
        // hence we must start the services before moving them into the `factory` closure
        // to avoid creating a separate long-running prediction task for each worker.
        let mut factories = self.factories.clone();
        for boxed_factory in factories.iter_mut() {
            boxed_factory.factory.start_service()?;
        }
        let http_server = HttpServer::new(move || {
            match factories.clone().into_iter().try_fold(
                actix_web::App::new(),
                |app, mut boxed_factory| {
                    let resource = boxed_factory.factory.new_resource(&boxed_factory.path)?;
                    Result::<actix_web::App<_>, ServiceError>::Ok(app.service(resource))
                },
            ) {
                Ok(app) => app,
                Err(err) => panic!("{err}"),
            }
        })
        .disable_signals();

        info!("Starting server...");
        let server = http_server.bind("127.0.0.1:8080")?.run();
        let server_handle = server.handle();
        tokio::spawn(async move { await_shutdown(server_handle).await });
        Ok(server)
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
