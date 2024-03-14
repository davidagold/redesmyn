use crate::predictions::{Configurable, HandlerArgs, Service};
use crate::schema::Schema;

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use futures::{Future, TryFutureExt};
use pyo3::{PyResult, Python};
use serde::Deserialize;
use tracing::subscriber::set_default;
use std::collections::VecDeque;
use std::env;
use std::future::{ready, Ready};
use tokio::{signal, task::JoinHandle};
use tracing::{error, info, Dispatch};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

trait ResourceFactory: Sync + Send {
    fn new_resource(&mut self, path: &str) -> Result<Resource, ServiceError>;

    fn clone_boxed(&self) -> Box<dyn ResourceFactory>;
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
        self.run()?;
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
}

pub trait Serve {
    fn register<S, O>(&mut self, service: S) -> &Self
    where
        S: Service + Configurable + Clone + Sync + Send + 'static,
        S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
        S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
        S::H: Handler<HandlerArgs<S::R, S::T>, Output = O> + Sync + Send,
        O: Responder + 'static;

        fn serve(&mut self) -> Result<actix_web::dev::Server, ServiceError>;
}

#[derive(Default)]
pub struct Server {
    pub(crate) factories: VecDeque<BoxedResourceFactory>,
}

impl Clone for Server {
    fn clone(&self) -> Self {
        let mut server = Server::default();
        for factory in self.factories.iter() {
            server.factories.push_back(factory.clone());
        };
        server
    }
}

impl Serve for Server {
    fn register<S, O>(&mut self, service: S) -> &Self
    where
        S: Service + Configurable + Clone + Sync + Send + 'static,
        S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
        S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
        S::H: Handler<HandlerArgs<S::R, S::T>, Output = O> + Sync + Send,
        O: Responder + 'static,
    {
        info!("Registering endpoint with path: ...");
        let path = service.get_config().unwrap().path.to_string();
        self.factories.push_back(BoxedResourceFactory(Box::new(service), path));
        self
    }

    fn serve(&mut self) -> Result<actix_web::dev::Server, ServiceError> {
        let factories = self.factories.clone();
        let http_server = HttpServer::new(move || {
            match factories.clone().into_iter().fold(
                Result::<actix_web::App<_>, ServiceError>::Ok(actix_web::App::new()),
                |app, mut factory| {
                    let resource = factory.0.new_resource(&factory.1)?;
                    Ok(app?.service(resource))
            }) 
            {
                Ok(app) => app,
                Err(err) => {
                    panic!("{err}")
                }
            }
        })
        .disable_signals();

        let server = http_server.bind("127.0.0.1:8080")?.run();

        let subscribe_layer = tracing_subscriber::fmt::layer().json();
        let guard = tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(subscribe_layer)
            .set_default();

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
        // Ok(tokio::spawn(server))
        // server.map_err(|err| err.into())
        Ok(server)
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
