use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::BatchPredictor,
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

    let mut schema = Schema::default();
    schema.add_field("a", datatypes::DataType::Float64);
    schema.add_field("b", datatypes::DataType::Float64);

    // let service = BatchPredictor::<String, ToyRecord>::new(schema)
    let service = BatchPredictor::<String, Schema>::new(schema)
        .path("predictions/{model_name}/{model_version}".into())
        .batch_max_size(100)
        .batch_max_delay_ms(5)
        .py_handler("handlers.model:handle".into());

    let mut server = Server::default();
    server.register(service);
    server.serve()?.await?;

    Ok(())
}
