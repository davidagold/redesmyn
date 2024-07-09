use crate::common::LogConfig;
use crate::predictions::{EndpointHandle, HandlerArgs, Service, ServiceCore};

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use pyo3::{PyResult, Python};
use serde::Deserialize;
use std::collections::{BTreeMap, VecDeque};
use std::env;
use tokio::signal;
use tracing::instrument;
use tracing::{error, info};

pub(crate) trait ResourceFactory: Sync + Send {
    fn new_resource(&mut self) -> Result<Resource, ServiceError>;

    fn clone_boxed(&self) -> Box<dyn ResourceFactory>;
}

impl Clone for Box<dyn ResourceFactory> {
    fn clone(&self) -> Self {
        self.clone_boxed()
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
    fn new_resource(&mut self) -> Result<Resource, ServiceError> {
        let handler = self.get_handler_fn();
        let resource = web::resource(self.path())
            .app_data(web::Data::<EndpointHandle<S::T, S::R>>::new(self.handle()?))
            .route(web::post().to(handler));
        Ok(resource)
    }

    fn clone_boxed(&self) -> Box<dyn ResourceFactory> {
        Box::new(self.clone())
    }
}

#[derive(Default)]
pub struct Server {
    pub(crate) services_by_path: BTreeMap<String, Box<dyn ServiceCore + Send + 'static>>,
    pythonpath: Vec<String>,
    config_log: LogConfig,
}

// We need this (at least in the current design) in order to create a Python future from `Server.serve`
impl Clone for Server {
    fn clone(&self) -> Self {
        let mut server = Server::default();
        for path in self.pythonpath.iter() {
            server.pythonpath.push(path.clone());
        }
        // for (path, service) in self.services_by_path.iter() {
        //     server.services_by_path.insert(path, value)
        // }
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
        S: Service + Sync + Send + 'static,
        S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
        S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
        S::H: Handler<HandlerArgs<S::R, S::T>, Output = O> + Sync + Send,
        O: Responder + 'static,
    {
        info!("Registering endpoint with path: {}", service.path());
        let path = service.path();
        self.services_by_path.insert(path, Box::new(service));
        info!("n services: {}", self.services_by_path.len());
        self
    }

    #[instrument(skip_all)]
    pub fn serve(&mut self) -> Result<actix_web::dev::Server, ServiceError> {
        let pwd = match env::current_dir() {
            Ok(pwd) => pwd,
            Err(err) => {
                error!("Failed to get working directory: {err}");
                return Err(ServiceError::IoError(err));
            }
        };
        tracing::info!("Starting server from directory {}", pwd.to_str().unwrap());

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

        // TODO: Handle failures
        info!("n services: {}", self.services_by_path.len());
        let handles: VecDeque<Box<dyn ResourceFactory>> = self
            .services_by_path
            .iter_mut()
            .filter_map(|(_, service)| match service.start() {
                Ok(factory) => {
                    info!("Successfully started service with path `{}`", service.path());
                    Some(factory)
                }
                Err(err) => {
                    error!("Failed to start service with path `{}`: {}", service.path(), err);
                    None
                }
            })
            .collect();

        let http_server = HttpServer::new(move || {
            match handles.clone().into_iter().try_fold(
                actix_web::App::new(),
                |app, mut boxed_factory| {
                    let resource = boxed_factory.new_resource()?;
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
