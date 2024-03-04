use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Configurable},
    schema::{Relation, Schema},
    server::{Serve, Server},
};

use redesmyn_macros::Relation;
use serde::Deserialize;

// #[derive(Debug, Deserialize, Relation)]
// pub struct ToyRecord {
//     a: f64,
//     b: f64,
// }

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), ServiceError> {
    // let schema = <ToyRecord as Relation>::schema(None).unwrap();

    let schema = Schema::default()
        .add_field("a", datatypes::DataType::Float64)
        .add_field("b", datatypes::DataType::Float64);

    // let service = BatchPredictor::<String, ToyRecord>::new(schema)
    let service = BatchPredictor::<String, Schema>::new(schema)
        .path("predictions/{model_name}/{model_version}".into())
        .batch_max_size(50)
        .batch_max_delay_ms(10)
        .py_handler("handlers.model:handle".into());

    let server = Server::default().register(service);
    server.serve()?.await?;

    Ok(())
}
