use std::iter::repeat;
use std::{collections::HashMap, time::Instant};


use polars::prelude::*;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Configurable},
    schema::{self, dataframe_from_records},
    server::{Serve, Server},
};
use redesmyn_macros::Schema;

use serde::de::{DeserializeSeed, IntoDeserializer};
use serde::Deserialize;
use serde_json::Deserializer;

#[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> Result<(), ServiceError> {
    let service = BatchPredictor::<String>::new()
        .path("predictions/{model_name}/{model_version}")
        .batch_max_capacity(50)
        .batch_max_delay_ms(10)
        .py_handler("handlers.model:handle");

    let server = Server::default().register(service);
    server.serve()?.await?.map_err(Into::into);


    Ok(())
}
