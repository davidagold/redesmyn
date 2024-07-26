use crate::common::{build_runtime, consume_and_log_err, include_python_paths, TOKIO_RUNTIME};
use crate::error::ServiceResult;
use crate::logging::LogConfig;
use crate::predictions::{EndpointHandle, HandlerArgs, Service, ServiceCore};
use crate::{config_methods, do_in};

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use pyo3::{pyclass, pymethods, PyResult};
use serde::Deserialize;
use std::cell::OnceCell;
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::sync::mpsc::channel;
use std::sync::{Arc, OnceLock};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::{select, signal};
use tracing::{error, info};
use tracing::{instrument, warn};

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

pub enum ServerCommand {
    SetInternalHandle {
        handle: actix_web::dev::ServerHandle,
        tx: oneshot::Sender<ServiceResult<()>>,
    },
    Stop {
        graceful: Option<bool>,
        tx: oneshot::Sender<ServiceResult<()>>,
    },
}

pub struct Server {
    pub(crate) services_by_path: BTreeMap<String, Box<dyn ServiceCore + Send + Sync + 'static>>,
    // TODO: Consolidate Python-related functionality away from particular constructs
    pythonpath: Vec<String>,
    tx_cmd: Arc<mpsc::Sender<ServerCommand>>,
    task: JoinHandle<ServiceResult<()>>,
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
    pub fn new(pythonpath: Option<Vec<String>>) -> Server {
        let (tx_cmd, rx_cmd) = mpsc::channel(512);
        let runtime = TOKIO_RUNTIME.get_or_init(build_runtime);
        let task = runtime.spawn(Server::task(rx_cmd));
        Server {
            services_by_path: BTreeMap::default(),
            pythonpath: pythonpath.unwrap_or_default(),
            task,
            tx_cmd: tx_cmd.into(),
        }
    }

    pub fn push_pythonpath(&mut self, path: &str) {
        self.pythonpath.push(path.to_string());
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
        let (tx, rx) = oneshot::channel();
        let cmd = ServerCommand::SetInternalHandle { handle: server.handle(), tx };
        consume_and_log_err(self.tx_cmd.try_send(cmd));

        Ok(server)
    }

    async fn task(mut rx_cmd: mpsc::Receiver<ServerCommand>) -> ServiceResult<()> {
        let server_handle = OnceCell::<actix_web::dev::ServerHandle>::new();
        loop {
            select! {
                _ = signal::ctrl_c() => {
                    tracing::info!("Received shutdown signal.");
                    match server_handle.get() {
                        Some(handle) => {
                            handle.stop(true).await;
                        }
                        None => {
                            warn!("Failed to obtain server handle");
                            break Err(ServiceError::from("Failed to shut down gracefully"))
                        }
                    }
                    break Ok(())
                },
                cmd = rx_cmd.recv() => {
                    match cmd {
                        Some(ServerCommand::Stop{ tx, graceful }) => {
                            tracing::info!("Received shutdown request.");
                            match server_handle.get() {
                                Some(handle) => {
                                    handle.stop(graceful.unwrap_or(true)).await;
                                    consume_and_log_err(tx.send(Ok(())));
                                }
                                None => {
                                    let err_msg = "Failed to shut down gracefully: Failed to obtain server handle";
                                    warn!(err_msg);
                                    consume_and_log_err(tx.send(Err(ServiceError::from(err_msg))));
                                    break Err(ServiceError::from(err_msg));
                                }
                            }
                            break Ok(())
                        }
                        Some(ServerCommand::SetInternalHandle { handle, .. }) => {
                            consume_and_log_err(server_handle.set(handle));
                        }
                        None => {
                            break Ok(())
                        },
                    }
                }
            }
        }
    }

    pub fn handle(&self) -> ServerHandle {
        ServerHandle { tx_cmd: self.tx_cmd.clone() }
    }
}

#[pyclass]
pub struct ServerHandle {
    tx_cmd: Arc<mpsc::Sender<ServerCommand>>,
}

#[pymethods]
impl ServerHandle {
    pub async fn stop(&self, graceful: Option<bool>) -> ServiceResult<()> {
        let (tx, rx) = oneshot::channel();
        let cmd = ServerCommand::Stop { graceful, tx };
        let result_send_cmd = self.tx_cmd.send(cmd).await;
        consume_and_log_err(result_send_cmd);
        rx.await?
    }
}
