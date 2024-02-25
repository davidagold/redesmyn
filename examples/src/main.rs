use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Schema},
    server::{Server, Serve},
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
    let service = BatchPredictor::<ToyRecord>::new();
    let mut server = Server::default();
    server.register("predictions/{model_name}/{model_version}", service);
    let handle = server.serve()?;

    handle.await?.map_err(|err| {
        error!("{err}");
        err.into()
    })
}
