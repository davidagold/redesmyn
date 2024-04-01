use crate::common::LogConfig;
use crate::predictions::{HandlerArgs, Service};
use crate::schema::Schema;

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use pyo3::{PyResult, Python};
use serde::Deserialize;
use std::collections::VecDeque;
use std::env;
use tokio::{signal, task::JoinHandle};
use tracing::instrument;
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

trait ResourceFactory: Sync + Send {
    fn new_resource(&mut self, path: &str) -> Result<Resource, ServiceError>;

    fn clone_boxed(&self) -> Box<dyn ResourceFactory>;

    fn start_service(&mut self) -> Result<JoinHandle<()>, ServiceError>;
}

pub(crate) struct BoxedResourceFactory(Box<dyn ResourceFactory>, String);

impl Clone for BoxedResourceFactory {
    fn clone(&self) -> Self {
        BoxedResourceFactory(self.0.clone_boxed(), self.1.clone())
    }
}

impl<R, H, O, S> ResourceFactory for S
where
    S: Service<R = R, H = H> + Clone + Sync + Send + 'static,
    S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
    R: Relation<Serialized = S::T> + Sync + Send + 'static,
    H: Handler<HandlerArgs<S::R, S::T>, Output = O>,
    O: Responder + 'static,
{
    fn new_resource(&mut self, path: &str) -> Result<Resource, ServiceError> {
        let handler = self.get_handler_fn();
        let resource = web::resource(path)
            .app_data(web::Data::<Self>::new(self.clone()))
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
        self.factories.push_back(BoxedResourceFactory(Box::new(service), path));
        self
    }

    #[instrument(skip_all)]
    pub fn serve(&mut self) -> Result<actix_web::dev::Server, ServiceError> {
        println!("Log config: {:#?}", self.config_log);

        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(self.config_log.layer())
            .init();

        let available_parallelism = std::thread::available_parallelism()?;
        info!(%available_parallelism);

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
            sys.getattr("path")?.getattr("insert").map(|insert| {
                let additional_paths = [&(pwd.to_str().unwrap().to_string())];
                for path in self.pythonpath.iter().chain(additional_paths) {
                    if let Err(err) = insert.call((0, path), None) {
                        error!("{err}");
                    };
                }
            })?;
            let python_path = sys.getattr("path")?.extract::<Vec<String>>()?;
            let str_python_path = serde_json::to_string_pretty(&python_path)
                .expect("Failed to serialize `sys.path`.");
            println!("Found Python path: {str_python_path}");
            info!(python_path = format!("{}", str_python_path.as_str()));
            PyResult::<()>::Ok(())
        }) {
            error!("{}", format!("Failed to initialize Python process: {err}"));
            return Err(err.into());
        };

        // Start the services before moving
        let mut factories = self.factories.clone();
        for factory in factories.iter_mut() {
            factory.0.start_service()?;
        }
        let http_server = HttpServer::new(move || {
            match factories.clone().into_iter().try_fold(
                actix_web::App::new(),
                |app, mut factory| {
                    let resource = factory.0.new_resource(&factory.1)?;
                    Result::<actix_web::App<_>, ServiceError>::Ok(app.service(resource))
                },
            ) {
                Ok(app) => app,
                Err(err) => panic!("{err}"),
            }
        })
        .disable_signals();

        let server = http_server.bind("127.0.0.1:8080")?.run();

        info!("Starting server...");
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
