use crate::predictions::{Configurable, HandlerArgs, Service};
use crate::schema::Schema;

use super::error::ServiceError;
use super::schema::Relation;
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use pyo3::{PyResult, Python};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::env;
use tokio::{signal, task::JoinHandle};
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

trait ResourceFactory: Sync + Send {
    fn new_resource(&self, path: &str) -> Resource;

    fn clone_boxed(&self) -> Box<dyn ResourceFactory>;
}

struct BoxedResourceFactory(Box<dyn ResourceFactory>, String);

impl Clone for BoxedResourceFactory {
    fn clone(&self) -> Self {
        BoxedResourceFactory(self.0.clone_boxed(), self.1.clone())
    }
}

impl<R, H, O, Serv> ResourceFactory for Serv
where
    Self: Service<R = R, H = H> + Clone + Sync + Send + 'static,
    <Self as Service>::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
    R: Relation<Serialized = <Self as Service>::T> + Sync + Send + 'static,
    H: Handler<HandlerArgs<<Self as Service>::R, <Self as Service>::T>, Output = O>,
    O: Responder + 'static,
{
    fn new_resource(&self, path: &str) -> Resource {
        let handler = self.get_handler_fn();
        web::resource(path)
            .app_data(web::Data::<Self>::new(self.clone()))
            .app_data(web::Data::<Schema>::new(self.get_schema()))
            .route(web::post().to(handler))
    }

    fn clone_boxed(&self) -> Box<dyn ResourceFactory> {
        Box::new(self.clone())
    }
}

pub trait Serve {
    fn register<S, O>(self, service: S) -> Self
    where
        S: Service + Configurable + Clone + Sync + Send + 'static,
        S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
        S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
        S::H: Handler<HandlerArgs<<S as Service>::R, <S as Service>::T>, Output = O> + Sync + Send,
        O: Responder + 'static;

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError>;
}

#[derive(Default)]
pub struct Server {
    factories: VecDeque<BoxedResourceFactory>,
}

impl Serve for Server {
    fn register<S, O>(mut self, mut service: S) -> Self
    where
        S: Service + Configurable + Clone + Sync + Send + 'static,
        S::T: Sync + Send + for<'de> Deserialize<'de> + 'static,
        S::R: Relation<Serialized = S::T> + Sync + Send + 'static,
        S::H: Handler<HandlerArgs<<S as Service>::R, <S as Service>::T>, Output = O> + Sync + Send,
        O: Responder + 'static,
    {
        info!("Registering endpoint with path: ...");
        service.run();
        let path = service.get_config().unwrap().path.to_string();
        self.factories.push_back(BoxedResourceFactory(Box::new(service), path));
        self
    }

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError> {
        let http_server = HttpServer::new(move || {
            self.factories.clone().into_iter().fold(actix_web::App::new(), |app, factory| {
                app.service(factory.0.new_resource(&factory.1))
            })
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
        Ok(tokio::spawn(server))
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
