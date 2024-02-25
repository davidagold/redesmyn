use crate::predictions::{ModelSpec, Service};

use super::error::ServiceError;
use super::predictions::{BatchPredictor, Schema};
use actix_web::{dev::ServerHandle, web, HttpServer};
use actix_web::{Handler, Resource, Responder};
use pyo3::{PyResult, Python};
use redesmyn_macros::Schema;
use serde::Deserialize;
use std::collections::VecDeque;
use std::env;
use tokio::{signal, task::JoinHandle};
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

pub trait Serves {
    // type H;

    fn register<S, O>(&mut self, endpoint: S) -> &Self
    where
        S: Service + Clone + Sync + Send + 'static,
        S::R: Schema<S::R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
        S::H: Handler<
                (
                    web::Path<ModelSpec>,
                    web::Json<Vec<S::R>>,
                    web::Data<BatchPredictor<S::R>>,
                ),
                Output = O,
            > + Sync
            + Send,
        O: Responder + 'static;

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError>;
}

struct BoxedToResource(Box<dyn ToResource>);

impl Clone for BoxedToResource {
    fn clone(&self) -> Self {
        self.0.clone_boxed()
    }
}

pub struct Server {
    factories: VecDeque<BoxedToResource>,
}

impl Default for Server {
    fn default() -> Self {
        Server { factories: VecDeque::new() }
    }
}

use polars::prelude::*;
#[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}
#[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord2 {
    a: f64,
    b: f64,
    c: f64,
}

trait ToResource: Sync + Send {
    fn to_resource(&self) -> Resource;

    fn clone_boxed(&self) -> BoxedToResource;
}

impl<R, H, O, T: Service<R = R, H = H>> ToResource for T
where
    Self: Clone + Sync + Send + 'static,
    R: Schema<R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
    H: Handler<
        (
            web::Path<ModelSpec>,
            web::Json<Vec<R>>,
            web::Data<BatchPredictor<R>>,
        ),
        Output = O,
    >,
    O: Responder + 'static,
{
    fn to_resource(&self) -> Resource {
        let handler = self.get_handler();
        web::resource(self.path())
            .app_data(web::Data::new(self.clone()))
            .route(web::post().to(handler))
    }

    fn clone_boxed(&self) -> BoxedToResource {
        BoxedToResource(Box::new(self.clone()))
    }
}

impl Serves for Server {
    fn register<S, O>(&mut self, service: S) -> &Self
    where
        S: Service + Clone + Sync + Send + 'static,
        S::R: Schema<S::R> + Sync + Send + 'static + for<'a> Deserialize<'a>,
        S::H: Handler<
                (
                    web::Path<ModelSpec>,
                    web::Json<Vec<S::R>>,
                    web::Data<BatchPredictor<S::R>>,
                ),
                Output = O,
            > + Sync
            + Send,
        O: Responder + 'static,
    {
        info!("Registering endpoint with path: ...");
        self.factories.push_back(BoxedToResource(Box::new(service)));
        self
    }

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError> {
        let http_server = HttpServer::new(move || {
            self.factories
                .clone()
                .into_iter()
                .fold(actix_web::App::new(), |app, factory| {
                    app.service(factory.0.to_resource())
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
        Ok(tokio::spawn(async move { server.await }))
    }
}

async fn await_shutdown(server_handle: ServerHandle) {
    let _ = signal::ctrl_c().await;
    tracing::info!("Received shutdown signal.");
    server_handle.stop(true).await;
}
