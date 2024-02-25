use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Schema},
    server::{Server, Serves},
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
    let mut server = Server::default();

    let path = "predictions/{model_name}/{model_version}";
    let service = BatchPredictor::<ToyRecord>::new(path);
    server.register(service);
    let handle = server.serve()?;
    handle.await?.map_err(|err| {
        error!("{err}");
        err.into()
    })
}
