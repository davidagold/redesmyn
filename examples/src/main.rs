use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Configurable, Schema},
    server::{Serve, Server},
};
use redesmyn_macros::Schema;
use serde::Deserialize;
use tracing::error;

#[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), ServiceError> {
    let service = BatchPredictor::<ToyRecord>::new()
        .path("predictions/{model_name}/{model_version}")
        .batch_max_capacity(50)
        .batch_max_delay_ms(10)
        .handler("handlers.model:handle");

    let server = Server::default().register(service);

    let handle = server.serve()?;
    handle.await?.map_err(|err| {
        error!("{err}");
        err.into()
    })
}
