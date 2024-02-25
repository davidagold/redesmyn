use crate::predictions::{invoke, Endpoint, ModelSpec, Service};

use super::error::ServiceError;
use super::predictions::{BatchPredictor, Schema};
use actix_web::{Handler, HttpMessage, HttpResponse, Responder};
use actix_web::{dev::ServerHandle, web, HttpServer};
use pyo3::{PyResult, Python};
use redesmyn_macros::Schema;
use serde::Deserialize;
use std::any::Any;
use std::collections::VecDeque;
use std::env;
use std::marker::PhantomData;
use tokio::{signal, task::JoinHandle};
use tracing::{error, info};
use tracing_subscriber::{self, layer::SubscriberExt, prelude::*, EnvFilter};

pub trait Serves {
    // type H;

    fn register<S>(&mut self, endpoint: S) -> &Self
    where
        S: ServiceFactory<R = dyn Any> + Clone + Send + 'static;

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError>;
}

pub trait ServiceFactory: Send {
    type R: Sync + Send + 'static;
    // type H;

    // fn new(&self) -> Box<dyn Service<H = Self::H>>;

    fn clone_boxed(&self) -> Box<dyn ServiceFactory<R = Self::R, H=Self::H>>;
}

// struct BoxedResourceFactory<R: ?Sized>(Box<dyn ServiceFactory<R = R>>);

// impl<R> Clone for BoxedResourceFactory<R> {
//     fn clone(&self) -> Self {
//         BoxedResourceFactory(self.0.clone_boxed())
//     }
// }

// struct BoxedService<H, R>(Box<dyn Service<H = H>>, PhantomData<R>)
// where
//     H: Handler<(
//         web::Path<ModelSpec>,
//         web::Json<Vec<R>>,
//         web::Data<BatchPredictor<R>>,
//     ), Output = HttpResponse>;



pub struct Server {
    // resource_factories: VecDeque<Box<dyn ServiceFactory<R = dyn Any>>>,
    // resource_factories: VecDeque<BoxedResourceFactory<dyn Any>>,
}

impl Default for Server {
    fn default() -> Self {
        // Server { resource_factories: VecDeque::new() }
        Server { }
    }
}

use polars::prelude::*;
#[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

impl Serves for Server {

    fn register<S>(&mut self, endpoint: S) -> &Self
    where
        S: ServiceFactory<R = dyn Any> + Clone + Send + 'static,
    {
        info!("Registering endpoint with path: ...");
        // self.resource_factories
        //     .push_back(BoxedResourceFactory(Box::new(endpoint)));
        self
    }

    fn serve(self) -> Result<JoinHandle<Result<(), std::io::Error>>, ServiceError> 
    {
        // let services: Vec<Box<dyn Service<R = dyn Any>>> = self
            // .resource_factories
            // .into_iter()
            // .map(|f| f.0.new())
            // .collect();

        // let services: Vec<Box<dyn Any>> = vec![
            // Box::new(BatchPredictor::new("predictions/{model_name}/{model_version}"))
        // ];

        // type BoxedService = Box<
        //     &dyn Service<R = dyn Any, H = <BatchPredictor<_> as Service>::H>
        // >;        

        let mut services: Vec<
            Box<
                &dyn Service<R = dyn Any, H = <BatchPredictor<_> as Service>::H>
                // &dyn Service<R = dyn Schema<<BatchPredictor<_> as Service>::R>, H = <BatchPredictor<_> as Service>::H>
            >
        > = vec![
            Box::new(
                &BatchPredictor::<ToyRecord>::new("predictions/{model_name}/{model_version}")
            )
            as Box<
                &dyn Service<R = dyn Any, H = <BatchPredictor<_> as Service>::H>
                // &dyn Service<R = dyn Schema<<BatchPredictor<_> as Service>::R>, H = <BatchPredictor<_> as Service>::H>
            >
        ];

        for service in services.iter_mut() {
            // .run();
        }

        let x = BatchPredictor::<ToyRecord>::new("predictions/{model_name}/{model_version}");

        let http_server = HttpServer::new(move || {
            let app = services.clone()
                .into_iter()
                .fold(actix_web::App::new(), |app, mut service| {
                    let handler = Box::new(*service).get_handler();
                    // service.run();
                    app.service(
                        web::resource("predictions/{model_name}/{model_version}")
                        .app_data(web::Data::new(*service))
                        .route(web::post().to(handler))
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
