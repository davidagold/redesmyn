use std::collections::HashMap;

use polars::prelude::*;
use redesmyn::schema;
use redesmyn::{
    error::ServiceError,
    predictions::{BatchPredictor, Configurable},
    server::{Serve, Server},
};
// use redesmyn_macros::Schema;
use serde::de::{DeserializeSeed, IntoDeserializer};
use serde::{Deserialize};
use serde_json::Deserializer;

// #[derive(Debug, Deserialize, Schema)]
pub struct ToyRecord {
    a: f64,
    b: f64,
}

// #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> Result<(), ServiceError> {
    // let service = BatchPredictor::<ToyRecord>::new()
    //     .path("predictions/{model_name}/{model_version}")
    //     .batch_max_capacity(50)
    //     .batch_max_delay_ms(10)
    //     .handler("handlers.model:handle");

    // let server = Server::default().register(service);
    // server.serve()?.await?.map_err(Into::into)

    // let Sche

    let json = r#"
        {
            "a": 1,
            "b": 2
        }
    "#;

    let schema = schema::Schema {
        fields: vec![
            schema::Field {name: "a".to_string(), data_type: schema::DataType::F64},
            schema::Field {name: "b".to_string(), data_type: schema::DataType::F64},
        ]
    };

    // let parsed: HashMap<String, usize> = serde_json::from_str(json)?;
    // print!("{:#?}", parsed);
    let mut de = Deserializer::from_str(json);
    let columns = schema::Columns(schema.columns()).deserialize(&mut de);
    print!("{:#?}", columns);
    Ok(())
}