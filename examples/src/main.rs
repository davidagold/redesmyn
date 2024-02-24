use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Endpoint, Schema},
    server::{Server, Serves},
};
use serde::Deserialize;

#[derive(Debug, Deserialize, redesmyn_macros::Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), ServiceError> {
    let mut server = Server::default();

    let path = "predictions/{model_name}/{model_version}";
    let service = BatchPredictor::<ToyRecord>::new(path);
    let endpoint = Endpoint { service, path: path.to_string() };
    server.register(endpoint);
    Ok(server.serve()?.await?)
}
