use crate::common::{build_runtime, consume_and_log_err, include_python_paths, TOKIO_RUNTIME};
use crate::logging::LogConfig;
use crate::predictions::{EndpointHandle, HandlerArgs, Service, ServiceCore};

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
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

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n_services = self.services_by_path.len();
        let n_pythonpath_entries = self.pythonpath.len();
        f.write_fmt(format_args!(
            "<Server, [{} services; {} entries in PYTHONPATH>",
            n_services, n_pythonpath_entries
        ))
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
        let runtime = TOKIO_RUNTIME.get_or_init(build_runtime);
        let pwd = env::current_dir().map_err(|err| {
            ServiceError::from(format!("Failed to obtain current working directory: {}", err))
        })?;
        tracing::info!("Starting server from directory {}", pwd.to_str().unwrap());

        let pythonpaths =
            self.pythonpath.iter().map(String::as_str).chain(vec![(pwd.to_str().unwrap())]);
        consume_and_log_err(include_python_paths(pythonpaths));

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
        runtime.spawn(async move { await_shutdown(server_handle).await });
        Ok(server)
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
